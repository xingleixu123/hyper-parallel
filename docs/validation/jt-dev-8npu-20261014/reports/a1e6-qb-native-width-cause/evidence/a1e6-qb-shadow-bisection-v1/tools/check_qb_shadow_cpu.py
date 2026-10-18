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

"""Real small Torch CPU fixtures; no JT model, NPU, or TP collective is executed."""

from __future__ import annotations

import ast
import copy
import gc
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from typing import Any
import unittest
from unittest import mock
import weakref

import torch

from compare_qb_shadow import compare_to_existing
from native_qb_shadow import (
    MaterializationBinding, QBShadowTrace, Topology, _copy_slab, _digest,
    batch_identity, cleanup_after_execution, hash_matrix, wrap_get_batch,
)
from qb_shadow_contract import CPU_SHAPE, MODULE, TARGET_RANK, read_rank_files, validate_run


def batch(index: int) -> tuple[dict, dict]:
    """Build synthetic CPU-only batch identities using the unchanged FIFO input contract."""
    ids = (torch.arange(4096, dtype=torch.int64).reshape(1, 4096) + index * 19) % 100003
    return {"input_ids": ids}, {"shift_labels": ids + 1}


def topology(rank: int = TARGET_RANK) -> Any:
    """Return a CPU-fixture FSDP8 coordinate; it creates no distributed state."""
    return Topology(rank=rank, dp_rank=rank, tp_rank=0, tp_size=1, dp_size=8,
                    sequence_parallel=False, hidden_size=128)


class FixtureManaged:
    """Weak-referenceable stand-in for a currently installed HSDP parameter owner."""

    def __init__(self, module: Any) -> None:
        self._module_info = SimpleNamespace(module=module, param_name="weight")
        self.sharded_state = SimpleNamespace(name="UNSHARDED")
        self.allgather_comm_ctx = SimpleNamespace(allgather_handle=None)
        self._orig_size = tuple(module.weight.shape)
        self.unsharded_param = module.weight
        self.unsharded_param_buffers = [module.weight.detach().view(-1)]
        self.source_shard_info = None


class FixtureState:
    """Fake CPU owner only; never an actual FSDP state or process group."""

    def __init__(self) -> None:
        self.is_shard = False


class Toy(torch.nn.Module):
    """One ordinary small Linear behind the exact diagnostic FQN."""

    def __init__(self) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        layer = torch.nn.Module()
        layer.self_attn = torch.nn.Module()
        layer.self_attn.q_b_proj = torch.nn.Linear(CPU_SHAPE[1], CPU_SHAPE[2], bias=False, dtype=torch.float64)
        self.model.layers = torch.nn.ModuleList([layer])
        with torch.no_grad():
            layer.self_attn.q_b_proj.weight.copy_(
                torch.arange(CPU_SHAPE[1] * CPU_SHAPE[2], dtype=torch.float64).reshape(
                    CPU_SHAPE[2], CPU_SHAPE[1]) / 4096 - 0.25)

    @property
    def projection(self) -> torch.nn.Linear:
        """Return this toy's existing ordinary Linear."""
        return self.model.layers[0].self_attn.q_b_proj

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Return the very same Tensor object produced by the Linear hooks."""
        return self.projection(value)


def make_trace(model: Toy, output: Path, rank: int = TARGET_RANK) -> QBShadowTrace:
    """Mark all fake topology/materialization as CPU-fixture-only."""
    owner, state = FixtureManaged(model.projection), FixtureState()
    model.fixture_owners = owner, state
    binding = MaterializationBinding(owner, state)
    target = batch_identity(*batch(TARGET_RANK))
    return QBShadowTrace(model, output, topology(rank), binding, {"verified": True, "scope": "CPU_FAKE_ONLY"},
                         cpu_fixture=True, fixture_target_batch=target)


def arm(trace: QBShadowTrace, rank: int = TARGET_RANK) -> None:
    """Bind one eagerly consumed synthetic CPU batch."""
    inputs = batch(rank)
    trace.queue_batch(*inputs)
    trace.begin_microbatch({**inputs[0], **inputs[1]})


def hidden(step: int = 0) -> torch.Tensor:
    """Return a deterministic small contiguous leaf, with no RNG consumed."""
    value = torch.arange(CPU_SHAPE[0] * CPU_SHAPE[1], dtype=torch.float64).reshape(1, *CPU_SHAPE[:2])
    return (value / 512 + step / 128).requires_grad_()


def records(directory: Path, rank: int = TARGET_RANK) -> list[dict]:
    """Parse the actual toy observer output."""
    return [json.loads(line) for line in (directory / f"qb-shadow-rank{rank}.jsonl").read_text().splitlines()]


def run_three(model: Toy, trace: QBShadowTrace | None = None, rank: int = TARGET_RANK) -> list[dict]:
    """Run real forward/backward and three SGD updates, with no diagnostic tensors in the loss."""
    outputs = []
    for step in range(3):
        if trace is not None:
            trace.start_step(step)
            arm(trace, rank)
        value = hidden(step)
        result = model(value)
        loss = result.square().mean() + result.sum() / 8192
        loss.backward()
        outputs.append({"Y": result.detach().clone(), "Xgrad": value.grad.detach().clone(),
                        "Wgrad": model.projection.weight.grad.detach().clone(), "loss": loss.detach().clone()})
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(parameter.grad, alpha=-0.0001)
                parameter.grad = None
        if trace is not None:
            trace.finish_step(step + 1)
    if trace is not None:
        trace.finish_run(3)
    return outputs


def fixture_run(output: Path) -> tuple[dict[int, list[dict]], dict]:
    """Generate eight real small CPU rank files, not a real distributed runtime."""
    for rank in range(8):
        toy = Toy()
        trace = make_trace(toy, output, rank)
        try:
            run_three(toy, trace, rank)
        finally:
            cleanup_after_execution(trace)
    rows, _sha = read_rank_files(output)
    return rows, validate_run(rows, cpu_fixture=True)


def old_states(validated: dict) -> tuple[dict, dict]:
    """Make explicitly synthetic old native-hash states from the actual CPU fixture output."""
    target = validated["target"]
    result, digest = target["result"], target["batch_sha256"]
    left, right = {"complete": {}}, {"complete": {}}
    for stage, field in (("input", "input_native_identity"), ("weight", "weight_native_identity")):
        value = result[field]
        state = {"dtype": value["dtype"], "logical_shape": value["shape"][-2:],
                 "blocks": {tuple(block["coordinates"]): block["sha256"] for block in value["blocks"]}}
        left["complete"][(digest, MODULE, stage)] = state
        right["complete"][(digest, MODULE, stage)] = copy.deepcopy(state)
    full = result["calibration"]["operands"]["production"]
    left["complete"][(digest, MODULE, "output")] = {
        "dtype": full["dtype"], "logical_shape": full["logical_shape"],
        "blocks": {tuple(block["coordinates"]): block["sha256"] for block in full["blocks"]}}
    merged = {}
    for half in result["halves"].values():
        merged.update({tuple(block["coordinates"]): block["sha256"]
                       for block in half["operands"]["repeat1"]["blocks"]})
    right["complete"][(digest, MODULE, "output")] = {
        "dtype": full["dtype"], "logical_shape": full["logical_shape"], "blocks": merged}
    return left, right


class QBShadowCPUTests(unittest.TestCase):
    """Bounded real CPU tensor checks; none constitutes NPU/JT/compiled-plan acceptance."""

    def assert_closed(self, trace: QBShadowTrace, model: Toy) -> None:
        """Require that failed own observation cannot silently restart."""
        self.assertTrue(trace.failed)
        self.assertFalse(trace.enabled)
        self.assertEqual(trace.handles, [])
        self.assertFalse(model.projection._forward_pre_hooks)
        self.assertFalse(model.projection._forward_hooks)
        with self.assertRaises(ValueError):
            trace.start_step(0)
        with self.assertRaises(ValueError):
            trace.finish_run(3)

    def test_three_step_forward_backward_and_updates_are_exact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            baseline, observed = Toy(), Toy()
            before = run_three(baseline)
            trace = make_trace(observed, Path(directory))
            try:
                after = run_three(observed, trace)
            finally:
                cleanup_after_execution(trace)
            for left, right in zip(before, after, strict=True):
                for key in left:
                    self.assertTrue(torch.equal(left[key], right[key]), key)
            self.assertTrue(torch.equal(baseline.projection.weight, observed.projection.weight))
            self.assertTrue(trace.run_complete)
            self.assertEqual(trace.handles, [])

    def test_original_object_graph_rng_versions_and_temporary_lifetimes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = Toy()
            saved = []
            external = model.projection.register_forward_hook(lambda _module, _args, out: saved.append(out))
            trace = make_trace(model, Path(directory))
            trace.start_step(0)
            arm(trace)
            native = torch.nn.functional.linear
            temporary = []

            def watch(*args: Any, **kwargs: Any) -> torch.Tensor:
                out = native(*args, **kwargs)
                if not torch.is_grad_enabled():
                    temporary.append(weakref.ref(out))
                return out

            try:
                with mock.patch.object(torch.nn.functional, "linear", side_effect=watch):
                    out = model(hidden())
                self.assertIs(out, saved[0])
                self.assertIsNotNone(out.grad_fn)
                out.sum().backward()
                result = next(row["result"] for row in records(Path(directory)) if row["kind"] == "shadow")
                self.assertTrue(all(result["invariants"].values()))
                self.assertEqual(result["production_identity_capture"],
                                 "actual_output_hook_before_any_shadow_linear")
                self.assertEqual(result["production_native_identity_before_shadow"]["blocks"],
                                 result["calibration"]["operands"]["production"]["blocks"])
                self.assertEqual(len(temporary), 5)
                gc.collect()
                self.assertTrue(all(ref() is None for ref in temporary))
                self.assertIsNone(trace.pre_metadata)
            finally:
                external.remove()
                cleanup_after_execution(trace)

    def test_full_calibration_failure_never_calls_half_width(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = Toy()
            trace = make_trace(model, Path(directory))
            native, widths = torch.nn.functional.linear, []

            def different_mode(value: torch.Tensor, weight: torch.Tensor, bias: Any = None) -> torch.Tensor:
                out = native(value, weight, bias)
                if not torch.is_grad_enabled():
                    widths.append(weight.shape[0])
                    return out + 0.25
                return out

            try:
                with mock.patch.object(torch.nn.functional, "linear", side_effect=different_mode):
                    run_three(model, trace)
                result = next(row["result"] for row in records(Path(directory)) if row["kind"] == "shadow")
                self.assertEqual(widths, [CPU_SHAPE[2]])
                self.assertEqual(result["status"], "CALIBRATION_FAILED_NO_N_INFERENCE")
                self.assertEqual(result["halves"], {})
                self.assertTrue(all(result["invariants"].values()))
            finally:
                cleanup_after_execution(trace)

    def test_full_numeric_half_statistics_and_global_offsets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = Toy()
            trace = make_trace(model, Path(directory))
            native = torch.nn.functional.linear

            def one_difference(value: torch.Tensor, weight: torch.Tensor, bias: Any = None) -> torch.Tensor:
                out = native(value, weight, bias)
                if not torch.is_grad_enabled() and weight.shape[0] == CPU_SHAPE[2] // 2:
                    out[0, 3, 77] += 0.125
                return out

            try:
                with mock.patch.object(torch.nn.functional, "linear", side_effect=one_difference):
                    run_three(model, trace)
                result = next(row["result"] for row in records(Path(directory)) if row["kind"] == "shadow")
                self.assertTrue(result["calibration_native_exact"])
                for side, half in result["halves"].items():
                    statistic = half["comparisons"]["production_vs_repeat1"]
                    self.assertEqual(statistic["nonzero_count"], 1)
                    self.assertEqual(statistic["max_abs"], 0.125)
                    self.assertEqual(statistic["max_coordinate"], [3, int(side) * CPU_SHAPE[2] // 2 + 77])
                    self.assertEqual(statistic["mse"], 0.125**2 / (CPU_SHAPE[0] * CPU_SHAPE[2] // 2))
                    self.assertTrue(statistic["finite"])
                    self.assertEqual(half["comparisons"]["repeat1_vs_repeat2"]["max_abs"], 0)
            finally:
                cleanup_after_execution(trace)

    def test_half_repeat_instability_stays_an_unaccepted_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = Toy()
            trace = make_trace(model, Path(directory))
            native, calls = torch.nn.functional.linear, 0

            def unstable(value: torch.Tensor, weight: torch.Tensor, bias: Any = None) -> torch.Tensor:
                nonlocal calls
                out = native(value, weight, bias)
                if not torch.is_grad_enabled() and weight.shape[0] == CPU_SHAPE[2] // 2:
                    calls += 1
                    out[0, 0, 0] += (calls % 2) * 0.25
                return out

            try:
                with mock.patch.object(torch.nn.functional, "linear", side_effect=unstable):
                    run_three(model, trace)
                self.assertEqual(calls, 4)
                self.assertEqual(trace.diagnostic_status, "HALF_REPEAT_UNSTABLE_NO_N_INFERENCE")
                self.assertTrue(trace.run_complete)
            finally:
                cleanup_after_execution(trace)

    def test_slab_and_native_serialization_budget_guards(self) -> None:
        counters = {"slab_copy_calls": 0, "slab_copy_bytes": 0, "largest_slab_bytes": 0}
        with self.assertRaisesRegex(ValueError, "24MiB"):
            _copy_slab(torch.zeros(512, 8192, dtype=torch.float64), 0, 512, counters)
        with self.assertRaisesRegex(ValueError, "bounded"):
            _digest(torch.zeros(512, 1024, dtype=torch.float64))
        self.assertEqual(counters["slab_copy_calls"], 0)

    def test_fifo_missing_duplicate_and_deferred_inputs_close_cleanly(self) -> None:
        for failure in ("wrong_fifo", "missing_output", "duplicate", "deferred"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                model = Toy()
                trace = make_trace(model, Path(directory))
                trace.start_step(0)
                try:
                    if failure == "wrong_fifo":
                        trace.queue_batch(*batch(2))
                        with self.assertRaises(ValueError):
                            inputs = batch(3)
                            trace.begin_microbatch({**inputs[0], **inputs[1]})
                    else:
                        if failure == "deferred":
                            def get_batch(_iterator: Any) -> Any:
                                return batch(2)

                            def prepare(_iterator: Any, _steps: int) -> Any:
                                return {}, iter([batch(2)])

                            get_batch.prepare_step = prepare
                            wrapped = wrap_get_batch(get_batch, trace)
                            _, batches = wrapped.prepare_step(None, 1)
                            self.assertFalse(trace.pending)
                            inputs = next(batches)
                            self.assertEqual(len(trace.pending), 1)
                            trace.begin_microbatch({**inputs[0], **inputs[1]})
                        else:
                            arm(trace)
                        if failure in ("missing_output", "deferred"):
                            with self.assertRaises(ValueError):
                                trace.finish_step(1)
                        else:
                            model(hidden())
                            with self.assertRaises(ValueError):
                                model(hidden())
                    self.assert_closed(trace, model)
                finally:
                    cleanup_after_execution(trace)

    def test_materialization_layout_and_nonfinite_fail_closed(self) -> None:
        for failure in ("sharded", "storage", "wrong_shape", "input_nonfinite", "weight_nonfinite",
                        "production_changed_during_repeat"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                model = Toy()
                production = []
                external = model.projection.register_forward_hook(lambda _module, _args, out: production.append(out))
                trace = make_trace(model, Path(directory))
                trace.start_step(0)
                arm(trace)
                value = hidden()
                if failure == "sharded":
                    model.fixture_owners[1].is_shard = True
                elif failure == "storage":
                    model.projection.weight = torch.nn.Parameter(model.projection.weight.detach().clone())
                elif failure == "wrong_shape":
                    value = value[:, :-1].contiguous()
                elif failure == "input_nonfinite":
                    value = torch.full_like(value, float("nan"), requires_grad=True)
                elif failure == "weight_nonfinite":
                    with torch.no_grad():
                        model.projection.weight.fill_(float("inf"))
                native = torch.nn.functional.linear

                def corrupt_original(*args: Any, **kwargs: Any) -> torch.Tensor:
                    out = native(*args, **kwargs)
                    if failure == "production_changed_during_repeat" and not torch.is_grad_enabled():
                        production[0].data[0, 0, 0] += 0.25
                    return out

                try:
                    with mock.patch.object(torch.nn.functional, "linear", side_effect=corrupt_original):
                        with self.assertRaises(ValueError):
                            model(value)
                    external.remove()
                    self.assert_closed(trace, model)
                finally:
                    external.remove()
                    cleanup_after_execution(trace)

    def test_original_forward_exception_preserves_object_and_outer_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = Toy()
            trace = make_trace(model, Path(directory))
            trace.start_step(0)
            arm(trace)
            original = RuntimeError("original toy Linear failure")
            with mock.patch.object(model.projection, "forward", side_effect=original):
                with self.assertRaises(RuntimeError) as caught:
                    try:
                        model(hidden())
                    finally:
                        cleanup_after_execution(trace)
            self.assertIs(caught.exception, original)
            self.assert_closed(trace, model)
            self.assertFalse(any(row["kind"] == "run_complete" for row in records(Path(directory))))

    def test_complete_json_and_synthetic_old_tp_native_tiles(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _rows, validated = fixture_run(Path(directory))
            left, right = old_states(validated)
            report = compare_to_existing(validated, left, right, cpu_fixture=True)
            self.assertEqual(report["status"], "N_WIDTH_SHADOW_REPRODUCES_OLD_TP2_NATIVE_BYTES")
            self.assertEqual(report["scope"], "cpu-fixture-only")
            self.assertTrue(all(report["old_and_new_actual_X_W_native_equal"].values()))
            self.assertEqual(len(report["comparisons_to_old_TP2"]), 4)

    def test_json_missing_overlap_target_and_same_hash_numeric_conflicts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            baseline, _validated = fixture_run(Path(directory))
            for failure in ("missing_rank", "missing_half", "missing_tile", "wrong_offset", "native_conflict",
                            "wrong_target", "missing_completion"):
                with self.subTest(failure=failure):
                    rows = copy.deepcopy(baseline)
                    shadow = next(row["result"] for row in rows[2] if row["kind"] == "shadow")
                    if failure == "missing_rank":
                        del rows[7]
                    elif failure == "missing_half":
                        del shadow["halves"]["1"]
                    elif failure == "missing_tile":
                        shadow["halves"]["0"]["operands"]["repeat1"]["blocks"].pop()
                    elif failure == "wrong_offset":
                        shadow["halves"]["1"]["global_features"][0] = 0
                    elif failure == "native_conflict":
                        shadow["calibration"]["comparisons"]["production_vs_repeat"]["max_abs"] = 0.25
                    elif failure == "wrong_target":
                        next(row for row in rows[2] if row["kind"] == "shadow")["batch_sha256"] = "0" * 64
                    else:
                        rows[0] = [row for row in rows[0] if row["kind"] != "run_complete"]
                    with self.assertRaises(ValueError):
                        validate_run(rows, cpu_fixture=True)

    def test_runtime_scope_restart_and_exclusive_output_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = Toy()
            trace = make_trace(model, Path(directory))
            try:
                with self.assertRaises(FileExistsError):
                    make_trace(Toy(), Path(directory))
                trace.start_step(0)
                with self.assertRaises(ValueError):
                    trace.start_step(0)
                self.assert_closed(trace, model)
            finally:
                cleanup_after_execution(trace)
        with tempfile.TemporaryDirectory() as directory:
            model = Toy()
            owner, state = FixtureManaged(model.projection), FixtureState()
            with self.assertRaises(ValueError):
                QBShadowTrace(model, Path(directory), topology(), MaterializationBinding(owner, state),
                              {"verified": False}, cpu_fixture=True,
                              fixture_target_batch=batch_identity(*batch(2)))
            with self.assertRaises(ValueError):
                QBShadowTrace(model, Path(directory), topology(), MaterializationBinding(owner, state),
                              {"verified": True}, fixture_target_batch=batch_identity(*batch(2)))
            self.assertFalse(list(Path(directory).iterdir()))

    def test_hook_returns_and_actual_capture_finally_are_explicit(self) -> None:
        source = Path(__file__).with_name("native_qb_shadow.py")
        parsed = ast.parse(source.read_text("utf-8"))
        hook_names = {"_pre_hook", "_post_hook"}
        found = set()
        for node in ast.walk(parsed):
            if isinstance(node, ast.FunctionDef) and node.name in hook_names:
                found.add(node.name)
                self.assertTrue(all(item.value is None for item in ast.walk(node) if isinstance(item, ast.Return)))
        self.assertEqual(found, hook_names)
        capture = ast.parse(Path(__file__).with_name("capture_qb_shadow_training.py").read_text("utf-8"))
        main = next(node for node in capture.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        self.assertTrue(any(isinstance(node, ast.Try) and node.finalbody for node in ast.walk(main)))
        self.assertNotIn("always_call", source.read_text("utf-8"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
