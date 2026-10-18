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
"""Use-point projection/weight observation, without replay or state repair."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, Mapping
import weakref

import torch

from attention0_contract import (
    EVENT_CALLS, EVENT_ORDER, FROZEN_SHA256, PARENT, SCHEMA, SOURCE_COMMIT,
    expected_region, projection_shapes, sample_coordinates,
)


def _load_frozen() -> ModuleType:
    path = (Path("/work/evidence/a1e6-native-forward-bisection-v1/tools/native_forward_trace.py")
            if os.name != "nt" else Path(__file__).resolve().parents[2]
            / "a1e6-native-forward-bisection-v1/tools/native_forward_trace.py")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != FROZEN_SHA256:
        raise ValueError("Frozen batch FIFO source is not b8f")
    name = "_jt_attention0_frozen_native_b8f"
    module = ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    try:
        exec(compile(payload, str(path), "exec", dont_inherit=True), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


FROZEN = _load_frozen()
Topology = FROZEN.Topology
batch_identity = FROZEN.batch_identity
wrap_get_batch = FROZEN.wrap_get_batch


def mesh_metadata(mesh: Any) -> dict[str, Any]:
    """Read only already-created mesh coordinates; never slice/create a mesh."""
    names = tuple(mesh.mesh_dim_names or ())
    shape = tuple(mesh.mesh_shape)
    coordinate = mesh.get_coordinate()
    ranks = tuple(mesh.rank_list)
    if (not names or len(names) != len(shape) or coordinate is None or len(coordinate) != len(shape)
            or any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != len(names) or any(type(size) is not int or size <= 0 for size in shape)
            or len(ranks) != math.prod(shape) or len(set(ranks)) != len(ranks)
            or any(type(rank) is not int or not 0 <= rank < 8 for rank in ranks)):
        raise ValueError("Unproven source mesh")
    flat = 0
    for position, size in zip(coordinate, shape, strict=True):
        if type(position) is not int or not 0 <= position < size:
            raise ValueError("Invalid existing mesh coordinate")
        flat = flat * size + position
    if flat >= len(ranks):
        raise ValueError("Source mesh rank map is incomplete")
    return {"names": list(names), "shape": list(shape), "coordinate": list(coordinate),
            "ranks": list(ranks), "current_rank": ranks[flat]}


def placement_metadata(placement: Any) -> dict[str, Any]:
    """Reject noncontiguous/ragged or unknown placement contracts."""
    if hasattr(placement, "split_factor") or (callable(getattr(placement, "is_ragged_shard", None))
                                              and placement.is_ragged_shard()):
        raise ValueError("Strided/ragged placements are unsupported")
    if callable(getattr(placement, "is_partial", None)) and placement.is_partial():
        if getattr(placement, "reduce_op", None) != "sum":
            raise ValueError("Only explicitly tagged sum contributors are supported")
        return {"kind": "partial", "reduce_op": "sum"}
    if callable(getattr(placement, "is_shard", None)) and placement.is_shard():
        return {"kind": "shard", "dim": placement.dim}
    if callable(getattr(placement, "is_replicate", None)) and placement.is_replicate():
        return {"kind": "replicate"}
    raise ValueError("Unknown tensor placement")


def source_layout(parameter: Any, name: str, topology: Any) -> dict[str, Any]:
    """Validate source TP ownership independently of a local tensor's shape."""
    info = parameter.source_shard_info
    if info is None:
        if topology.tp_size != 1:
            raise ValueError("TP2 weight has no source-layout proof")
        return {"kind": "unsharded_tp1", "tp_coordinate": 0}
    mesh = mesh_metadata(info.mesh)
    placements = [placement_metadata(item) for item in info.placements]
    if len(placements) != len(mesh["names"]) or mesh["current_rank"] != topology.rank:
        raise ValueError("Source layout rank/mesh mismatch")
    if "tp" not in mesh["names"]:
        raise ValueError("Source layout has no TP axis")
    axis = mesh["names"].index("tp")
    expected = {"kind": "replicate"}
    if name in ("q_b_proj", "kv_b_proj"):
        expected = {"kind": "shard", "dim": 0}
    elif name == "o_proj":
        expected = {"kind": "shard", "dim": 1}
    if (mesh["shape"][axis] != topology.tp_size or mesh["coordinate"][axis] != topology.tp_rank
            or placements[axis] != expected
            or any(value != {"kind": "replicate"} for index, value in enumerate(placements) if index != axis)):
        raise ValueError("Source placement does not prove this contiguous TP weight slice")
    return {"kind": "source_shard_info", "mesh": mesh, "placements": placements,
            "origin_is_dtensor": info.origin_is_dtensor}


def _local_tensor(value: Any, region: Mapping[str, Any], topology: Any) -> tuple[torch.Tensor, dict[str, Any]]:
    distributed = callable(getattr(value, "to_local", None))
    tensor_layout = getattr(value, "layout", None)
    placements = getattr(value, "placements", None)
    if placements is None:
        placements = getattr(tensor_layout, "placements", None)
    declared_partial = callable(getattr(tensor_layout, "is_partial", None)) and tensor_layout.is_partial()
    allowed_partial = region["semantics"] == "k_partition_contribution"
    if declared_partial and not allowed_partial:
        raise ValueError("Partial is not a complete activation or weight")
    observed = {"kind": "plain_local_with_pinned_runtime_contract"}
    if distributed:
        if not placements:
            raise ValueError("Distributed value has no placement proof")
        mesh = mesh_metadata(value.device_mesh)
        kinds = [placement_metadata(item) for item in placements]
        if len(kinds) != len(mesh["names"]) or mesh["current_rank"] != topology.rank:
            raise ValueError("Distributed value mesh/rank mismatch")
        physical_ndim = region["physical_ndim"]
        rb, re_, cb, ce = region["bounds"]
        axis_kind = {"kind": "replicate"}
        if allowed_partial and topology.tp_size > 1:
            axis_kind = {"kind": "partial", "reduce_op": "sum"}
        elif re_ - rb != region["logical_shape"][0]:
            axis_kind = {"kind": "shard", "dim": physical_ndim - 2}
        elif ce - cb != region["logical_shape"][1]:
            axis_kind = {"kind": "shard", "dim": physical_ndim - 1}
        if "tp" not in mesh["names"]:
            raise ValueError("Distributed value has no TP axis")
        tp_axis = mesh["names"].index("tp")
        if (mesh["shape"][tp_axis] != topology.tp_size or mesh["coordinate"][tp_axis] != topology.tp_rank
                or kinds[tp_axis] != axis_kind
                or any(item != {"kind": "replicate"} for axis, item in enumerate(kinds) if axis != tp_axis)):
            raise ValueError("Distributed placement contradicts canonical activation coverage")
        logical_rows, logical_cols = region["logical_shape"]
        expected_global = ([logical_rows, logical_cols] if physical_ndim == 2 else
                           [1, logical_rows, logical_cols] if physical_ndim == 3 else
                           [1, 1, logical_rows, logical_cols])
        if list(value.shape) != expected_global:
            raise ValueError("Distributed global shape differs from this exact boundary")
        observed = {"kind": "distributed", "mesh": mesh, "placements": kinds}
    if not callable(getattr(value, "detach", None)):
        raise ValueError("Expected one actual Tensor")
    detached = value.detach()
    local = detached.to_local() if distributed else detached
    if not isinstance(local, torch.Tensor) or not local.is_floating_point():
        raise ValueError("Expected a floating local tensor")
    return local, observed


def physical_metadata(value: torch.Tensor) -> dict[str, Any]:
    """Record the actual physical storage contract without making a device copy."""
    storage = value.untyped_storage()
    if any(stride <= 0 for stride in value.stride()):
        raise ValueError("Unknown nonpositive physical stride")
    span = value.storage_offset() + 1 + sum((size - 1) * stride
                                          for size, stride in zip(value.shape, value.stride(), strict=True))
    if value.numel() == 0 or not storage.data_ptr() or span * value.element_size() > storage.nbytes():
        raise ValueError("Observed storage is empty, freed or too small for the actual view")
    return {"shape": list(value.shape), "stride": list(value.stride()), "dtype": str(value.dtype),
            "device": str(value.device), "storage_offset": value.storage_offset(),
            "data_ptr": value.data_ptr(), "storage_data_ptr": storage.data_ptr(),
            "storage_identity": storage._cdata, "storage_nbytes": storage.nbytes(),
            "contiguous": value.is_contiguous(), "last_dimension_stride_one": value.stride()[-1] == 1}


class MaterializationBinding:
    """Keep only weak references to existing HSDP owners; never cache their tensors."""

    def __init__(self, parameter: Any, state: Any) -> None:
        self.parameter = weakref.ref(parameter)
        self.state = weakref.ref(state)

    def check(self, module: Any, name: str, topology: Any, region: Mapping[str, Any], *,
              cpu_fixture: bool = False) -> tuple[torch.Tensor, dict[str, Any]]:
        """Require currently installed UNSHARDED storage at the actual Linear pre-hook."""
        parameter, state = self.parameter(), self.state()
        if parameter is None or state is None:
            raise ValueError("HSDP materialization owner expired")
        info = parameter._module_info
        if info.module is not module or info.param_name != "weight":
            raise ValueError("Managed parameter owner differs from the actual Linear")
        if (state.is_shard is not False or getattr(parameter.sharded_state, "name", None) != "UNSHARDED"
                or parameter.allgather_comm_ctx.allgather_handle is not None):
            raise ValueError("Weight is not a completed, installed UNSHARDED parameter")
        layout = source_layout(parameter, name, topology)
        actual, actual_layout = _local_tensor(module.weight, region, topology)
        installed, _installed_layout = _local_tensor(parameter.unsharded_param, region, topology)
        left, right = physical_metadata(actual), physical_metadata(installed)
        compared = ("shape", "stride", "dtype", "device", "storage_offset", "data_ptr",
                    "storage_data_ptr", "storage_identity", "storage_nbytes")
        if any(left[key] != right[key] for key in compared):
            raise ValueError("Actual module.weight is not the current installed unsharded storage")
        rb, re_, cb, ce = region["bounds"]
        shape = [re_ - rb, ce - cb]
        if (list(actual.shape) != shape or tuple(parameter._orig_size) != tuple(shape)
                or actual.ndim != 2 or actual.stride()[1] != 1
                or actual.stride()[0] < actual.shape[1] or actual.device.type != ("cpu" if cpu_fixture else "npu")
                or (not cpu_fixture and actual.dtype != torch.bfloat16)):
            raise ValueError("Unknown materialized weight shape/stride/device/dtype")
        buffers = parameter.unsharded_param_buffers
        if not buffers or not any(buffer.untyped_storage()._cdata == left["storage_identity"] for buffer in buffers):
            raise ValueError("Installed weight does not belong to its unsharded buffer storage")
        proof = {"verified": True, "state": "UNSHARDED", "owner_is_shard": False, "pending_handle": False,
                 "installed_storage_equal": True, "actual_object_id": id(module.weight),
                 "installed_object_id": id(parameter.unsharded_param), "source_layout": layout,
                 "actual_tensor_layout": actual_layout, "physical": left}
        return actual, proof


def summarize(value: Any, region: Mapping[str, Any], topology: Any, *, cpu_fixture: bool = False) -> dict[str, Any]:
    """Hash one bounded CPU tile at a time; preserve all NPU input/storage strides."""
    local, observed = _local_tensor(value, region, topology)
    rb, re_, cb, ce = region["bounds"]
    ndim = region["physical_ndim"]
    expected = ([re_ - rb, ce - cb] if ndim == 2 else
                [1, re_ - rb, ce - cb] if ndim == 3 else [1, 1, re_ - rb, ce - cb])
    if (list(local.shape) != expected or local.device.type != ("cpu" if cpu_fixture else "npu")
            or any(stride <= 0 for stride in local.stride())):
        raise ValueError("Unknown actual physical activation/weight shape, stride or device")
    physical = physical_metadata(local)
    view = local if ndim == 2 else local[0] if ndim == 3 else local[0, 0]
    tile = (128, 128) if region["semantics"] == "weight" else (512, 128)
    sample_rows = sample_coordinates(rb, re_, tile[0], region["logical_shape"][0])
    sample_cols = sample_coordinates(cb, ce, tile[1], region["logical_shape"][1])
    samples = {row: {} for row in sample_rows}
    blocks = []
    for row in range(rb, re_, tile[0]):
        row_end = min(row + tile[0], re_)
        for col in range(cb, ce, tile[1]):
            col_end = min(col + tile[1], ce)
            cpu = view[row - rb:row_end - rb, col - cb:col_end - cb].cpu().contiguous()
            if not bool(torch.isfinite(cpu).all()):
                raise ValueError("Nonfinite actual observed values")
            digest = hashlib.sha256(cpu.view(torch.uint8).numpy().tobytes()).hexdigest()
            blocks.append({"coordinates": [row, row_end, col, col_end], "sha256": digest})
            for global_row in sample_rows:
                if row <= global_row < row_end:
                    for global_col in sample_cols:
                        if col <= global_col < col_end:
                            samples[global_row][global_col] = float(cpu[global_row - row, global_col - col])
            del cpu
    result = {"logical_shape": region["logical_shape"], "bounds": region["bounds"],
              "semantics": region["semantics"], "physical": physical,
              "actual_tensor_layout": observed, "blocks": blocks,
              "grid": {"rows": sample_rows, "columns": sample_cols,
                       "values": [[samples[row][col] for col in sample_cols] for row in sample_rows]}}
    if "contributor_k" in region:
        result["contributor_k"] = region["contributor_k"]
    return result


class Attention0ForwardTrace(FROZEN.NativeForwardTrace):
    """Observe thirteen actual events per microbatch using the frozen batch FIFO."""

    def __init__(self, model: torch.nn.Module, output: Path, topology: Any,
                 bindings: Mapping[str, MaterializationBinding], runtime_layout_proof: Mapping[str, Any], *,
                 cpu_fixture: bool = False) -> None:
        if topology.hidden_size != (128 if cpu_fixture else 5120):
            raise ValueError("Wrong native/CPU-fixture hidden contract")
        if runtime_layout_proof.get("verified") is not True:
            raise ValueError("Actual runtime layout is unproven")
        self.topology, self.cpu_fixture = topology, cpu_fixture
        self.path = output / f"attention0-forward-rank{topology.rank}.jsonl"
        self.bindings = dict(bindings)
        self.enabled, self.first_step_started, self.first_step_complete = False, False, False
        self.failed, self.run_complete = False, False
        self.pending, self.calls, self.seen_batches, self.handles = deque(), Counter(), set(), []
        self.batch, self.micro = None, -1
        modules = dict(model.named_modules())
        names = {name for name, _stage in EVENT_ORDER}
        if names - set(modules) or len({id(modules[name]) for name in names}) != len(names):
            raise ValueError("Missing or aliased actual attention projection modules")
        if set(bindings) != {f"{PARENT}.{name}" for name in projection_shapes(cpu_fixture)}:
            raise ValueError("Each actual projection needs one materialization binding")
        if any(type(modules[name]) is not torch.nn.Linear for name in bindings):
            raise ValueError("Expected unchanged ordinary torch Linear projections")
        meta = {"kind": "metadata", "schema": SCHEMA, "source_commit": SOURCE_COMMIT,
                "frozen_source_sha256": FROZEN_SHA256, "topology": asdict(topology),
                "event_order": [list(key) for key in EVENT_ORDER],
                "projection_shapes": {key: list(value) for key, value in projection_shapes(cpu_fixture).items()},
                "scope": "cpu-fixture" if cpu_fixture else "native-candidate",
                "runtime_layout_proof": dict(runtime_layout_proof),
                "limitation": "Completion alone is not own-prefix transparency or GEMM root-cause acceptance"}
        with self.path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(meta, sort_keys=True, allow_nan=False) + "\n")
        try:
            for name in bindings:
                self.handles.append(modules[name].register_forward_pre_hook(self._pre_hook(name), with_kwargs=True))
                self.handles.append(modules[name].register_forward_hook(self._output_hook(name)))
            self.handles.append(modules[PARENT].register_forward_hook(self._output_hook(PARENT)))
        except BaseException:
            self._fail_closed()
            raise

    def _fail_closed(self) -> None:
        self.failed = True
        self.enabled = False
        self.close()

    def _require_not_failed(self) -> None:
        if self.failed:
            raise ValueError("Observation previously failed and cannot be restarted or completed")

    def abort(self) -> None:
        """Unload observation after an interrupted run without writing completion."""
        self._fail_closed()

    def start_step(self, completed_steps: int) -> None:
        """Arm step one once; later steps have no observation hooks."""
        self._require_not_failed()
        try:
            if completed_steps == 0:
                if self.first_step_started:
                    raise ValueError("Step one cannot be restarted")
                self.first_step_started = True
            elif not self.first_step_complete:
                raise ValueError("Cannot resume without completed first-step observation")
            super().start_step(completed_steps)
        except BaseException:
            self._fail_closed()
            raise

    def queue_batch(self, model_inputs: Mapping[str, Any], loss_inputs: Mapping[str, Any]) -> None:
        """Keep the frozen FIFO behavior, closing on an invalid actual batch."""
        self._require_not_failed()
        try:
            super().queue_batch(model_inputs, loss_inputs)
        except BaseException:
            self._fail_closed()
            raise

    def begin_microbatch(self, fields: Mapping[str, Any]) -> None:
        """Bind the frozen FIFO identity, closing on ambiguity or incomplete calls."""
        self._require_not_failed()
        try:
            super().begin_microbatch(fields)
        except BaseException:
            self._fail_closed()
            raise

    def _check_previous_calls(self) -> None:
        if self.batch is not None and dict(self.calls) != EVENT_CALLS:
            raise ValueError("Missing or repeated attention projection/weight boundaries")

    def _record(self, name: str, stage: str, value: Any, materialization: dict | None = None) -> None:
        if self.batch is None:
            raise ValueError("Attention projection ran before actual batch binding")
        key = f"{name}::{stage}"
        self.calls[key] += 1
        if self.calls[key] != 1:
            raise ValueError("Repeated attention projection boundary or replay")
        region = expected_region(name, stage, asdict(self.topology), cpu_fixture=self.cpu_fixture)
        summary = summarize(value, region, self.topology, cpu_fixture=self.cpu_fixture)
        if materialization is not None:
            summary["materialization"] = materialization
        self._event("boundary", step=1, micro=self.micro, dp=self.topology.dp_rank, tp=self.topology.tp_rank,
                    batch_sha256=self.batch, module=name, stage=stage, call=0, **summary)

    def _pre_hook(self, name: str) -> Any:
        def observe(module: Any, args: tuple, kwargs: Mapping[str, Any]) -> None:
            if not self.enabled:
                return
            try:
                if self.batch is None:
                    raise ValueError("Attention projection ran before actual batch binding")
                if args and "input" in kwargs:
                    raise ValueError("Ambiguous Linear input")
                value = args[0] if args else kwargs.get("input")
                region = expected_region(name, "weight", asdict(self.topology), cpu_fixture=self.cpu_fixture)
                actual, proof = self.bindings[name].check(
                    module, name.rsplit(".", 1)[1], self.topology, region, cpu_fixture=self.cpu_fixture)
                self._record(name, "weight", actual, proof)
                del actual
                self._record(name, "input", value)
            except BaseException:
                self._fail_closed()
                raise
        return observe

    def _output_hook(self, name: str) -> Any:
        def observe(_module: Any, _args: Any, output: Any) -> None:
            if not self.enabled:
                return
            try:
                if name == PARENT:
                    if not isinstance(output, tuple) or len(output) != 2 or output[1] is not None:
                        raise ValueError("Expected the original native (attention_output, None) tuple")
                    value = output[0]
                else:
                    value = output
                self._record(name, "output", value)
            except BaseException:
                self._fail_closed()
                raise
        return observe

    def finish_step(self, completed_steps: int) -> None:
        """Check first-step FIFO completeness and remove all hooks immediately."""
        self._require_not_failed()
        try:
            active = self.enabled
            super().finish_step(completed_steps)
            if active:
                self.first_step_complete = True
        except BaseException:
            self._fail_closed()
            raise

    def finish_run(self, completed_steps: int) -> None:
        """Require observed first step and exactly three normal optimizer steps."""
        self._require_not_failed()
        try:
            if not self.first_step_complete:
                raise ValueError("Incomplete first-step observation")
            super().finish_run(completed_steps)
            self.run_complete = True
        except BaseException:
            self._fail_closed()
            raise


def cleanup_after_execution(trace: Attention0ForwardTrace | None) -> None:
    """Clean up outside the module hook dispatcher, preserving any original error."""
    if trace is not None:
        if not trace.run_complete:
            trace.abort()
        else:
            trace.close()
