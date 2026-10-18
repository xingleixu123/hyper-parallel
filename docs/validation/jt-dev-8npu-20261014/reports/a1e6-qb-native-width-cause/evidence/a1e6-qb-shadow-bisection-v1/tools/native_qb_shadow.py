# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Calibrated q_b N-width shadows; the original Tensor and training graph survive unchanged."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch

from qb_shadow_contract import (
    CPU_SHAPE, FEATURE_BLOCK, FROZEN_FILES, MODULE, NATIVE_SHAPE, SCHEMA, SOURCE_COMMIT,
    TARGET_BATCH, TARGET_RANK, TOKEN_BLOCK, WEIGHT_BLOCK, WEIGHT_CPU_ROWS, frozen_support,
)


SUPPORT = frozen_support()
FROZEN = SUPPORT.FROZEN
Topology = SUPPORT.Topology
MaterializationBinding = SUPPORT.MaterializationBinding
wrap_get_batch = SUPPORT.wrap_get_batch
batch_identity = SUPPORT.batch_identity
physical_metadata = SUPPORT.physical_metadata
CALLS = {"q_b_pre": 1, "q_b_post": 1}


def _digest(cpu: torch.Tensor) -> str:
    """Hash one bounded CPU tile, never a complete feature-wide slab."""
    if cpu.device.type != "cpu" or cpu.numel() * cpu.element_size() > TOKEN_BLOCK * FEATURE_BLOCK * 8:
        raise ValueError("Native bytes may only be serialized from one bounded CPU tile/RNG state")
    return hashlib.sha256(cpu.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def _copy_slab(matrix: torch.Tensor, begin: int, end: int, copies: dict[str, int]) -> torch.Tensor:
    if matrix.ndim != 2 or not matrix.is_contiguous():
        raise ValueError("Only complete contiguous feature rows may cross to CPU")
    if (end - begin) * matrix.shape[1] * matrix.element_size() > 24 * 2**20:
        raise ValueError("One native CPU slab may not exceed24MiB")
    cpu = torch.empty((end - begin, matrix.shape[1]), dtype=matrix.dtype, device="cpu")
    cpu.copy_(matrix[begin:end], non_blocking=False)
    size = cpu.numel() * cpu.element_size()
    copies["slab_copy_calls"] += 1
    copies["slab_copy_bytes"] += size
    copies["largest_slab_bytes"] = max(copies["largest_slab_bytes"], size)
    return cpu


def hash_matrix(value: torch.Tensor, *, weight: bool, copies: dict[str, int]) -> dict[str, Any]:
    """Stream X/W/originalY identities, with all native byte hashes and bounded CPU storage."""
    matrix = value.detach() if weight else value.detach()[0]
    rows, columns = matrix.shape
    row_tile = WEIGHT_BLOCK if weight else TOKEN_BLOCK
    slab_rows = WEIGHT_CPU_ROWS if weight else TOKEN_BLOCK
    blocks = []
    for begin in range(0, rows, slab_rows):
        end = min(begin + slab_rows, rows)
        cpu = _copy_slab(matrix, begin, end, copies)
        copies["cpu_native_bytes_bound"] = max(copies["cpu_native_bytes_bound"], cpu.numel() * cpu.element_size())
        for row in range(begin, end, row_tile):
            row_end = min(row + row_tile, end)
            for col in range(0, columns, FEATURE_BLOCK):
                col_end = min(col + FEATURE_BLOCK, columns)
                tile = cpu[row - begin:row_end - begin, col:col_end]
                if not bool(torch.isfinite(tile).all()):
                    raise ValueError("Nonfinite actual X/W/production output")
                blocks.append({"coordinates": [row, row_end, col, col_end], "sha256": _digest(tile)})
                del tile
        del cpu
    return {"shape": list(value.shape), "dtype": str(value.dtype), "blocks": blocks}


def _new_stats() -> dict[str, Any]:
    return {"compared_numel": 0, "left_finite": 0, "right_finite": 0, "both_finite": 0,
            "nonzero_count": 0, "max_abs": 0.0, "max_coordinate": None, "max_values": None,
            "different_native_blocks": 0, "_squares": []}


def _add_tile(stats: dict[str, Any], left: torch.Tensor, right: torch.Tensor,
              coordinate: list[int], native_equal: bool) -> None:
    if (left.device.type != "cpu" or right.device.type != "cpu" or left.shape != right.shape
            or left.numel() > TOKEN_BLOCK * FEATURE_BLOCK):
        raise ValueError("FP64 statistics are limited to one complete CPU tile")
    stats["compared_numel"] += left.numel()
    stats["different_native_blocks"] += not native_equal
    left_finite, right_finite = torch.isfinite(left), torch.isfinite(right)
    both_finite = left_finite & right_finite
    stats["left_finite"] += int(left_finite.sum())
    stats["right_finite"] += int(right_finite.sum())
    stats["both_finite"] += int(both_finite.sum())
    if not bool(both_finite.all()):
        return
    a, b = left.to(torch.float64), right.to(torch.float64)
    difference = a - b
    stats["nonzero_count"] += int(torch.count_nonzero(difference))
    magnitude = difference.abs()
    maximum = float(magnitude.max())
    if maximum > stats["max_abs"]:
        index = int(magnitude.reshape(-1).argmax())
        row, col = divmod(index, left.shape[1])
        stats["max_abs"] = maximum
        stats["max_coordinate"] = [coordinate[0] + row, coordinate[2] + col]
        stats["max_values"] = [float(a[row, col]), float(b[row, col])]
    stats["_squares"].append(float((difference * difference).sum()))


def _finish_stats(stats: dict[str, Any]) -> dict[str, Any]:
    squares = math.fsum(stats.pop("_squares"))
    count = stats["compared_numel"]
    finite = all(stats[key] == count for key in ("left_finite", "right_finite", "both_finite"))
    stats["finite"] = finite
    stats["squared_error_sum"] = squares if finite else None
    stats["mse"] = squares / count if finite else None
    if not finite:
        stats["max_abs"] = None
    return stats


def compare_outputs(operands: Mapping[str, tuple[torch.Tensor, int]], pairs: Mapping[str, tuple[str, str]],
                    shape: tuple[int, int, int], begin: int, end: int,
                    copies: dict[str, int]) -> dict[str, Any]:
    """Copy wide contiguous rows once, then hash/measure complete tiles only on CPU.

    Production always copies its full 24MiB slab even for a half comparison. This
    avoids an implicit NPU contiguous pack of the strided production half-view.
    """
    rows, _inner, width = shape
    summaries = {name: {"logical_shape": [rows, width], "bounds": [0, rows, begin, end],
                        "dtype": str(value.dtype), "physical": physical_metadata(value), "blocks": []}
                 for name, (value, _offset) in operands.items()}
    statistics = {name: _new_stats() for name in pairs}
    for row in range(0, rows, TOKEN_BLOCK):
        row_end = min(row + TOKEN_BLOCK, rows)
        if sum((row_end - row) * value.shape[-1] * value.element_size()
               for value, _offset in operands.values()) > 48 * 2**20:
            raise ValueError("Live native CPU slabs may not exceed48MiB")
        slabs = {name: _copy_slab(value.detach()[0], row, row_end, copies)
                 for name, (value, _offset) in operands.items()}
        resident = sum(cpu.numel() * cpu.element_size() for cpu in slabs.values())
        copies["cpu_native_bytes_bound"] = max(copies["cpu_native_bytes_bound"], resident)
        if not all(cpu.device.type == "cpu" for cpu in slabs.values()):
            raise ValueError("Statistics must run on bounded CPU tiles")
        for col in range(begin, end, FEATURE_BLOCK):
            col_end = min(col + FEATURE_BLOCK, end)
            coordinate = [row, row_end, col, col_end]
            tiles = {name: slabs[name][:, col - offset:col_end - offset]
                     for name, (_value, offset) in operands.items()}
            digests = {name: _digest(tile) for name, tile in tiles.items()}
            for name in operands:
                summaries[name]["blocks"].append({"coordinates": coordinate, "sha256": digests[name]})
            for name, (left, right) in pairs.items():
                _add_tile(statistics[name], tiles[left], tiles[right], coordinate, digests[left] == digests[right])
            del tiles
        slabs.clear()
    return {"operands": summaries, "comparisons": {name: _finish_stats(value)
                                                   for name, value in statistics.items()}}


def rng_digest(*, cpu_fixture: bool, device: Any = None) -> dict[str, str]:
    """Read RNG states without reseeding, initializing an NPU, or consuming randomness."""
    result = {"cpu": _digest(torch.get_rng_state())}
    if not cpu_fixture:
        backend = getattr(torch, "npu", None)
        if backend is None or not backend.is_initialized():
            raise ValueError("Shadow cannot initialize an NPU to query its RNG")
        result["npu"] = _digest(backend.get_rng_state(device))
    return result


class QBShadowTrace(FROZEN.NativeForwardTrace):
    """Keep the frozen actual-batch FIFO; run shadows on one proven FSDP use-point only."""

    def __init__(self, model: torch.nn.Module, output: Path, topology: Any, binding: Any,
                 runtime_layout_proof: Mapping[str, Any], *, cpu_fixture: bool = False,
                 fixture_target_batch: str | None = None) -> None:
        if (topology.tp_size != 1 or topology.dp_size != 8 or topology.rank != topology.dp_rank
                or topology.sequence_parallel or runtime_layout_proof.get("verified") is not True):
            raise ValueError("Only the fixed FSDP8 use-point is eligible")
        if not cpu_fixture and fixture_target_batch is not None:
            raise ValueError("A native run cannot override the selected real batch")
        self.topology, self.binding, self.cpu_fixture = topology, binding, cpu_fixture
        self.shape = CPU_SHAPE if cpu_fixture else NATIVE_SHAPE
        self.target_batch = fixture_target_batch if cpu_fixture else TARGET_BATCH
        if self.target_batch is None or len(self.target_batch) != 64:
            raise ValueError("An explicit CPU-fixture identity or the fixed native batch is required")
        self.path = output / f"qb-shadow-rank{topology.rank}.jsonl"
        self.pending, self.calls, self.seen_batches, self.handles = deque(), Counter(), set(), []
        self.batch, self.micro = None, -1
        self.enabled = self.first_step_started = self.first_step_complete = False
        self.failed = self.run_complete = self.target_seen = False
        self.pre_metadata = None
        self.diagnostic_status = "NOT_SELECTED"
        modules = dict(model.named_modules())
        if MODULE not in modules or type(modules[MODULE]) is not torch.nn.Linear or modules[MODULE].bias is not None:
            raise ValueError("Exactly the ordinary, bias-free layer0 q_b Linear is required")
        self._event("metadata", schema=SCHEMA, source_commit=SOURCE_COMMIT,
                    topology=asdict(topology), scope="cpu-fixture-only" if cpu_fixture else "native-candidate",
                    shape=list(self.shape), target_batch=self.target_batch, target_rank=TARGET_RANK,
                    frozen_inputs={name: digest for name, (_path, digest) in FROZEN_FILES.items()},
                    runtime_layout_proof=dict(runtime_layout_proof),
                    limitation="This is a shadow mechanism diagnostic, not a200step verdict or repaired forward")
        try:
            self.handles.append(modules[MODULE].register_forward_pre_hook(self._pre_hook, with_kwargs=True))
            self.handles.append(modules[MODULE].register_forward_hook(self._post_hook, with_kwargs=True))
        except BaseException:
            self._fail_closed()
            raise

    def _event(self, kind: str, **fields: Any) -> None:
        if kind == "metadata":
            if self.path.exists():
                raise FileExistsError(self.path)
            with self.path.open("x", encoding="utf-8") as handle:
                handle.write(json.dumps({"kind": kind, "rank": self.topology.rank, **fields},
                                        sort_keys=True, allow_nan=False) + "\n")
            return
        super()._event(kind, **fields)

    def _fail_closed(self) -> None:
        self.failed, self.enabled = True, False
        self.pre_metadata = None
        self.close()

    def abort(self) -> None:
        """Uninstall after an interrupted capture without writing run completion."""
        self._fail_closed()

    def _require_active(self) -> None:
        if self.failed:
            raise ValueError("Failed shadow cannot restart or complete")

    def start_step(self, completed_steps: int) -> None:
        """Arm the first step exactly once; later two steps have no q_b hooks."""
        self._require_active()
        try:
            if completed_steps == 0:
                if self.first_step_started:
                    raise ValueError("The first step cannot restart")
                self.first_step_started = True
            elif not self.first_step_complete:
                raise ValueError("Missing completed first-step observation")
            super().start_step(completed_steps)
        except BaseException:
            self._fail_closed()
            raise

    def queue_batch(self, model_inputs: Mapping[str, Any], loss_inputs: Mapping[str, Any]) -> None:
        """Reuse the pinned eager/deferred FIFO at actual consumption."""
        self._require_active()
        try:
            super().queue_batch(model_inputs, loss_inputs)
        except BaseException:
            self._fail_closed()
            raise

    def begin_microbatch(self, fields: Mapping[str, Any]) -> None:
        """Require the unchanged real target at its proven rank and microstep."""
        self._require_active()
        try:
            super().begin_microbatch(fields)
            if self.enabled and ((self.batch == self.target_batch) != (self.topology.rank == TARGET_RANK)):
                raise ValueError("Target batch/rank allocation differs from the observed original")
        except BaseException:
            self._fail_closed()
            raise

    def _check_previous_calls(self) -> None:
        if self.batch is not None and dict(self.calls) != CALLS:
            raise ValueError("Missing/repeated actual q_b pre/output boundary")

    def _region(self) -> dict[str, Any]:
        _rows, inner, width = self.shape
        return {"logical_shape": [width, inner], "bounds": [0, width, 0, inner],
                "physical_ndim": 2, "semantics": "weight"}

    def _checked(self, value: Any, shape: tuple[int, ...]) -> dict[str, Any]:
        if (type(value) not in (torch.Tensor, torch.nn.Parameter) or tuple(value.shape) != shape
                or value.dtype != (torch.float64 if self.cpu_fixture else torch.bfloat16)
                or value.device.type != ("cpu" if self.cpu_fixture else "npu") or not value.is_contiguous()
                or value.storage_offset() != 0):
            raise ValueError("Unknown actual plain q_b tensor shape/dtype/device/storage")
        meta = physical_metadata(value)
        if meta["storage_nbytes"] != value.numel() * value.element_size():
            raise ValueError("q_b storage capacity differs from the pinned contiguous buffer")
        return meta

    @staticmethod
    def _input(args: tuple, kwargs: Mapping[str, Any]) -> torch.Tensor:
        if len(args) == 1 and not kwargs:
            return args[0]
        if not args and set(kwargs) == {"input"}:
            return kwargs["input"]
        raise ValueError("Ambiguous or extra ordinary Linear arguments")

    def _pre_hook(self, module: Any, args: tuple, kwargs: Mapping[str, Any]) -> None:
        if not self.enabled:
            return
        try:
            if self.batch is None or self.micro != 0:
                raise ValueError("q_b ran outside its bound first microbatch")
            self.calls["q_b_pre"] += 1
            if self.calls["q_b_pre"] != 1:
                raise ValueError("Repeated q_b call/replay")
            rows, inner, width = self.shape
            value = self._input(args, kwargs)
            actual, proof = self.binding.check(module, "q_b_proj", self.topology, self._region(),
                                               cpu_fixture=self.cpu_fixture)
            self.pre_metadata = {"input": self._checked(value, (1, rows, inner)),
                                 "weight": self._checked(actual, (width, inner)),
                                 "input_object_id": id(value), "weight_object_id": id(module.weight),
                                 "input_version": value._version, "weight_version": module.weight._version,
                                 "materialization": proof}
            if self.batch == self.target_batch:
                copies = {"slab_copy_calls": 0, "slab_copy_bytes": 0, "largest_slab_bytes": 0,
                          "cpu_native_bytes_bound": 0}
                self.pre_metadata["rng_before"] = rng_digest(cpu_fixture=self.cpu_fixture, device=value.device)
                self.pre_metadata["input_native_identity"] = hash_matrix(value, weight=False, copies=copies)
                self.pre_metadata["weight_native_identity"] = hash_matrix(actual, weight=True, copies=copies)
                self.pre_metadata["copies"] = copies
            public = {key: item for key, item in self.pre_metadata.items()
                      if key not in ("input_native_identity", "weight_native_identity", "copies", "rng_before")}
            self._event("projection", step=1, micro=0, module=MODULE, stage="input",
                        batch_sha256=self.batch, **public)
        except BaseException:
            self._fail_closed()
            raise

    def _post_hook(self, module: Any, args: tuple, kwargs: Mapping[str, Any], output: Any) -> None:
        if not self.enabled:
            return
        try:
            self.calls["q_b_post"] += 1
            if self.calls["q_b_post"] != 1 or self.pre_metadata is None:
                raise ValueError("q_b output lacks its unique actual input")
            rows, inner, width = self.shape
            value = self._input(args, kwargs)
            actual, proof = self.binding.check(module, "q_b_proj", self.topology, self._region(),
                                               cpu_fixture=self.cpu_fixture)
            output_meta = self._checked(output, (1, rows, width))
            if (self._checked(value, (1, rows, inner)) != self.pre_metadata["input"]
                    or self._checked(actual, (width, inner)) != self.pre_metadata["weight"]
                    or id(value) != self.pre_metadata["input_object_id"]
                    or id(module.weight) != self.pre_metadata["weight_object_id"]
                    or value._version != self.pre_metadata["input_version"]
                    or module.weight._version != self.pre_metadata["weight_version"]):
                raise ValueError("q_b changed actual X/W or installed storage during forward")
            if self.batch == self.target_batch:
                if self.target_seen:
                    raise ValueError("Target shadow cannot repeat")
                self.target_seen = True
                result = self._shadow(module, value, output)
                self.diagnostic_status = result["status"]
                self._event("shadow", step=1, micro=0, module=MODULE, stage="output",
                            batch_sha256=self.batch, result=result)
            self._event("projection", step=1, micro=0, module=MODULE, stage="output",
                        batch_sha256=self.batch, physical=output_meta, materialization=proof)
            self.pre_metadata = None
        except BaseException:
            self._fail_closed()
            raise

    def _shadow(self, module: Any, value: torch.Tensor, output: torch.Tensor) -> dict[str, Any]:
        original_weight = module.weight
        original_output_id = id(output)
        original_linear = torch.nn.functional.linear
        original_forward = getattr(module.forward, "__func__", module.forward)
        grad_node = output.grad_fn
        original_versions = [tensor._version for tensor in (value, original_weight, output)]
        original_grad = [tensor.requires_grad for tensor in (value, original_weight, output)]
        original_grad_enabled = torch.is_grad_enabled()
        rng_before = self.pre_metadata["rng_before"]
        copies = self.pre_metadata["copies"]
        before_x = self.pre_metadata["input_native_identity"]
        before_w = self.pre_metadata["weight_native_identity"]
        before_y = hash_matrix(output, weight=False, copies=copies)
        halves = {}
        with torch.no_grad():
            x, weight = value.detach(), original_weight.detach()
            repeat = torch.nn.functional.linear(x, weight, None)
            try:
                self._checked(repeat, (1, self.shape[0], self.shape[2]))
                calibration = compare_outputs(
                    {"production": (output, 0), "repeat": (repeat, 0)},
                    {"production_vs_repeat": ("production", "repeat")}, self.shape, 0, self.shape[2], copies)
            finally:
                del repeat
            finite = calibration["comparisons"]["production_vs_repeat"]["finite"]
            if not finite:
                raise ValueError("Nonfinite full-N calibration")
            exact = calibration["operands"]["production"]["blocks"] == calibration["operands"]["repeat"]["blocks"]
            if exact:
                for side in range(2):
                    begin, end = side * self.shape[2] // 2, (side + 1) * self.shape[2] // 2
                    view = weight[begin:end]
                    first = torch.nn.functional.linear(x, view, None)
                    try:
                        second = torch.nn.functional.linear(x, view, None)
                        try:
                            self._checked(first, (1, self.shape[0], end - begin))
                            self._checked(second, (1, self.shape[0], end - begin))
                            half = compare_outputs(
                                {"production": (output, 0), "repeat1": (first, begin), "repeat2": (second, begin)},
                                {"production_vs_repeat1": ("production", "repeat1"),
                                 "production_vs_repeat2": ("production", "repeat2"),
                                 "repeat1_vs_repeat2": ("repeat1", "repeat2")},
                                self.shape, begin, end, copies)
                            if any(not record["finite"] for record in half["comparisons"].values()):
                                raise ValueError("Nonfinite half-N shadow")
                            half["global_features"] = [begin, end]
                            half["weight_view"] = physical_metadata(view)
                            halves[str(side)] = half
                        finally:
                            del second
                    finally:
                        del first, view
            del x, weight
        after_x = hash_matrix(value, weight=False, copies=copies)
        after_w = hash_matrix(original_weight, weight=True, copies=copies)
        after_y = hash_matrix(output, weight=False, copies=copies)
        _actual, final_materialization = self.binding.check(
            module, "q_b_proj", self.topology, self._region(), cpu_fixture=self.cpu_fixture)
        del _actual
        invariants = {
            "input_native_unchanged": before_x == after_x,
            "weight_native_unchanged": before_w == after_w,
            "production_native_unchanged": before_y == after_y,
            "rng_unchanged": rng_before == rng_digest(cpu_fixture=self.cpu_fixture, device=value.device),
            "production_object_preserved": id(output) == original_output_id and output.grad_fn is grad_node,
            "requires_grad_unchanged": original_grad == [tensor.requires_grad
                                                        for tensor in (value, original_weight, output)],
            "tensor_versions_unchanged": original_versions == [tensor._version
                                                               for tensor in (value, original_weight, output)],
            "grad_enabled_unchanged": original_grad_enabled == torch.is_grad_enabled(),
            "module_weight_object_unchanged": module.weight is original_weight,
            "functional_linear_unchanged": torch.nn.functional.linear is original_linear,
            "module_forward_unchanged": getattr(module.forward, "__func__", module.forward) is original_forward,
        }
        if not all(invariants.values()):
            raise ValueError("Shadow changed original input/weight/output/RNG/graph state")
        stable = all(half["operands"]["repeat1"]["blocks"] == half["operands"]["repeat2"]["blocks"]
                     for half in halves.values())
        status = ("CALIBRATION_FAILED_NO_N_INFERENCE" if not exact else
                  "CORE_SHADOW_COMPLETE" if stable else "HALF_REPEAT_UNSTABLE_NO_N_INFERENCE")
        return {"status": status, "calibration_native_exact": exact, "calibration": calibration, "halves": halves,
                "invariants": invariants, "rng_before": rng_before, "input_native_identity": before_x,
                "weight_native_identity": before_w, "final_materialization": final_materialization, "copies": copies,
                "input_and_weight_identity_capture": "actual_forward_pre_hook_before_original_linear",
                "production_identity_capture": "actual_output_hook_before_any_shadow_linear",
                "production_native_identity_before_shadow": before_y,
                "temporary_output_bytes_bound": output.element_size() * self.shape[0] * self.shape[2],
                "CPU_tile_scratch_budget_bytes": 4 * 2**20,
                "CPU_total_tensor_budget_bytes": copies["cpu_native_bytes_bound"] + 4 * 2**20,
                "unknown_backend_workspace_excluded": True,
                "limitation": "Calibration changes grad mode;failure isnot proof of identical-condition nondeterminism"}

    def finish_step(self, completed_steps: int) -> None:
        """Require one real q_b call and the target on rank2; unload after step one."""
        self._require_active()
        try:
            active = self.enabled
            if active and self.target_seen != (self.topology.rank == TARGET_RANK):
                raise ValueError("Missing or misplaced fixed target shadow")
            super().finish_step(completed_steps)
            if active:
                self.first_step_complete = True
        except BaseException:
            self._fail_closed()
            raise

    def finish_run(self, completed_steps: int) -> None:
        """Complete only after three unchanged normal steps;diagnostic failure stays failure."""
        self._require_active()
        try:
            if not self.first_step_complete or completed_steps != 3:
                raise ValueError("Three-step shadow run is incomplete")
            self.close()
            self._event("run_complete", completed_steps=3, diagnostic_status=self.diagnostic_status)
            self.run_complete = True
        except BaseException:
            self._fail_closed()
            raise


def cleanup_after_execution(trace: QBShadowTrace | None) -> None:
    """Clean outside the Torch dispatcher; never turn an incomplete capture into PASS."""
    if trace is not None:
        if not trace.run_complete:
            trace.abort()
        else:
            trace.close()
