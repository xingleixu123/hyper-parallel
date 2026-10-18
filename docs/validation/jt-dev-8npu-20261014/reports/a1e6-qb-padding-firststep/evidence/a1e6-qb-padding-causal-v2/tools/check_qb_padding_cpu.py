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

"""Real small CPU tensor tests; fake topology/owners are not a JT/NPU acceptance."""

from __future__ import annotations

import ast
import contextvars
import copy
import functools
import gc
import inspect
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from typing import Any
import unittest
from unittest import mock
import weakref

import torch

from compare_qb_padding import METRICS, RECIPE_FIELDS, compare_native, prefix_transparency
from native_qb_padding import (
    MaterializationBinding, QBPaddingTrace, Topology, actual_forward_proof, batch_identity, cleanup_after_execution,
    effective_linear, live_weight, wrap_get_batch,
)
from qb_padding_contract import CPU_SHAPE, MODULE, read_run, source_snapshot, validate_run


def _source_fragment(name: str, function_name: str, *, class_name: str | None = None) -> Any:
    path, raw = source_snapshot(name)
    tree = ast.parse(raw)
    scope = tree.body if class_name is None else next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name).body
    function = next(node for node in scope if isinstance(node, ast.FunctionDef) and node.name == function_name)
    namespace = {"Any": Any, "Tensor": torch.Tensor, "functools": functools, "DTensor": FixtureWeight,
                 "_LOCAL_COMPUTE_ACTIVE": contextvars.ContextVar("CPU_local_only", default=True)}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[function_name]


class FixturePlacement:
    """Read-only contiguous fake placement, without distributed execution."""

    dim = 0

    @staticmethod
    def is_partial() -> bool:
        """No fixture is a Partial contribution."""
        return False

    @staticmethod
    def is_shard() -> bool:
        """The two fake peers own disjoint output rows."""
        return True

    @staticmethod
    def is_replicate() -> bool:
        """The declared source weight is not replicated across fake peers."""
        return False


class FixtureMesh:
    """Only source-layout metadata; no process group is created."""

    def __init__(self, rank: int) -> None:
        self.mesh_dim_names, self.mesh_shape = ("tp",), (2,)
        self.rank_list = (rank // 2 * 2, rank // 2 * 2 + 1)
        self.coordinate = [rank % 2]

    def get_coordinate(self) -> list[int]:
        """Return the existing simulated coordinate."""
        return list(self.coordinate)


class FixtureWeight:
    """Explicit CPU-only holder around an unchanged differentiable local Parameter."""

    _padding_CPU_fixture_only = True

    def __init__(self, local: torch.Tensor, rank: int) -> None:
        self._local_tensor, self.rank = local, rank
        self.device_mesh, self.placements = FixtureMesh(rank), (FixturePlacement(),)
        self.layout = SimpleNamespace(is_partial=lambda: False)
        self.shape = (CPU_SHAPE[2], CPU_SHAPE[1])

    def detach(self) -> FixtureWeight:
        """Detach only the materialization observer, not the effective calculation."""
        return FixtureWeight(self._local_tensor.detach(), self.rank)

    def to_local(self) -> torch.Tensor:
        """Return the same local object; a test also executes the actual fixed source method."""
        return self._local_tensor


class FixtureManaged:
    """Keep weak-referenceable fake installed-buffer metadata alive."""

    def __init__(self, module: Any, rank: int, leaf: torch.Tensor) -> None:
        self._module_info = SimpleNamespace(module=module, param_name="weight")
        self.sharded_state = SimpleNamespace(name="UNSHARDED")
        self.allgather_comm_ctx = SimpleNamespace(allgather_handle=None)
        self._orig_size = tuple(leaf.shape)
        self.unsharded_param = module.weight
        self.unsharded_param_buffers = [leaf.detach().view(-1)]
        self.source_shard_info = SimpleNamespace(mesh=FixtureMesh(rank), placements=(FixturePlacement(),),
                                                origin_is_dtensor=isinstance(module.weight, FixtureWeight))


class FixtureState:
    """Weak-referenceable CPU-only state for the existing materialization checker."""

    def __init__(self) -> None:
        self.is_shard = False


class FixtureScheduler:
    """Simulate ancestor ordering only; this is not an HSDP implementation."""

    def __init__(self, owner: Any, state: Any, parameter: Any) -> None:
        self.hsdp_state, self.parameter, self.modules = state, parameter, (owner,)
        self.order = []
        owner.register_forward_pre_hook(self._forward_pre_hook, with_kwargs=True)
        owner.register_forward_hook(self._forward_hook)

    def _forward_pre_hook(self, _module: Any, _args: Any, _kwargs: Any) -> None:
        self.order.append("ancestor_pre")
        self.hsdp_state.is_shard = False
        self.parameter.sharded_state.name = "UNSHARDED"

    def _forward_hook(self, _module: Any, _args: Any, _output: Any) -> None:
        self.order.append("ancestor_post")
        self.hsdp_state.is_shard = True
        self.parameter.sharded_state.name = "SHARDED"


class FixtureLayer(torch.nn.Module):
    """Provide a real Module ancestor around the real small Linear."""

    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.self_attn = torch.nn.Module()
        self.self_attn.q_b_proj = torch.nn.Linear(CPU_SHAPE[1], CPU_SHAPE[2] // 2,
                                                bias=False, dtype=torch.float64)
        with torch.no_grad():
            self.self_attn.q_b_proj.weight.copy_(weight)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Return the projection result through ordinary Module hook dispatch."""
        return self.self_attn.q_b_proj(value)


class Toy(torch.nn.Module):
    """Use real CPU autograd with exact FQNs and explicitly simulated TP metadata."""

    def __init__(self, rank: int = 4, wrapped: bool = False) -> None:
        super().__init__()
        self.rank = rank
        width, inner = CPU_SHAPE[2], CPU_SHAPE[1]
        full = torch.arange(width * inner, dtype=torch.float64).reshape(width, inner) / 131072 - 0.125
        self.model = torch.nn.Module()
        begin = rank % 2 * width // 2
        self.model.layers = torch.nn.ModuleList([FixtureLayer(full[begin:begin + width // 2])])
        object.__setattr__(self, "local_parameter", self.projection.weight)
        if wrapped:
            self.projection._parameters["weight"] = FixtureWeight(self.local_parameter, rank)
            _source_fragment("rewriter", "_wrap_module_forward")(self.projection)
        self.fixture_state = FixtureState()
        self.fixture_managed = FixtureManaged(self.projection, rank, self.local_parameter)
        self.model.layers[0].hsdp_scheduler = FixtureScheduler(
            self.model.layers[0], self.fixture_state, self.fixture_managed)

    @property
    def projection(self) -> torch.nn.Linear:
        """Return the existing ordinary Linear, without adapting its class."""
        return self.model.layers[0].self_attn.q_b_proj

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Run the simulated ancestor and unmodified original projection forward."""
        return self.model.layers[0](value)


def batch(index: int) -> tuple[dict, dict]:
    """Build fake input identity separately from the small CPU hidden Tensor."""
    ids = (torch.arange(4096, dtype=torch.int64).reshape(1, 4096) + index * 19) % 100003
    return {"input_ids": ids}, {"shift_labels": ids + 1}


def topology(rank: int) -> Any:
    """Use rank parity only to construct this explicit fake metadata, never native routing."""
    return Topology(rank=rank, dp_rank=rank // 2, tp_rank=rank % 2, tp_size=2, dp_size=4,
                    sequence_parallel=True, hidden_size=128)


def make_trace(model: Toy, output: Path, arm: str = "T") -> QBPaddingTrace:
    """Bind weak fake owners while retaining actual CPU tensors and gradients."""
    return QBPaddingTrace(model, output, topology(model.rank),
                          MaterializationBinding(model.fixture_managed, model.fixture_state),
                          {"verified": True, "scope": "CPU_FAKE_OWNER_ONLY"}, arm, model.model.layers[0],
                          cpu_fixture=True, fixture_target_batch=batch_identity(*batch(4)))


def arm_micro(trace: QBPaddingTrace, micro: int) -> None:
    """Queue/bind the actual fake DP/micro logical identity."""
    inputs = batch(trace.topology.dp_rank * 2 + micro)
    trace.queue_batch(*inputs)
    trace.begin_microbatch({**inputs[0], **inputs[1]})


def hidden(step: int = 0, micro: int = 0) -> torch.Tensor:
    """Return deterministic small CPU values, with no RNG consumed."""
    values = torch.arange(CPU_SHAPE[0] * CPU_SHAPE[1], dtype=torch.float64).reshape(1, *CPU_SHAPE[:2])
    return (values / 8192 + step / 16 + micro / 8).requires_grad_()


def run_three(model: Toy, trace: QBPaddingTrace | None = None) -> list[dict]:
    """Execute real CPU forward/backward and three explicit SGD updates."""
    values = []
    for step in range(3):
        if trace is not None:
            trace.start_step(step)
        for micro in range(2):
            if trace is not None:
                arm_micro(trace, micro)
            x = hidden(step, micro)
            y = model(x)
            loss = y.square().mean() / 2
            loss.backward()
            values.append({"Y": y.detach().clone(), "Xgrad": x.grad.detach().clone(),
                           "Wgrad": model.local_parameter.grad.detach().clone(), "loss": loss.detach().clone()})
        values.append({"grad_norm": model.local_parameter.grad.norm().detach().clone()})
        with torch.no_grad():
            model.local_parameter.add_(model.local_parameter.grad, alpha=-1 / 16384)
            model.local_parameter.grad = None
        if trace is not None:
            trace.finish_step(step + 1)
    if trace is not None:
        trace.finish_run(3)
    return values


def fixture_run(output: Path, arm: str = "T") -> dict:
    """Eight serial CPU models are not an actual distributed or TP execution."""
    for rank in range(8):
        model = Toy(rank)
        trace = make_trace(model, output, arm)
        try:
            run_three(model, trace)
        finally:
            cleanup_after_execution(trace)
    return validate_run(read_run(output)[0], cpu_fixture=True)


class QBPaddingCPUTests(unittest.TestCase):
    """Real small tensor math and lifecycle checks; no JT compiled-plan/NPU claim."""

    def assert_closed(self, trace: QBPaddingTrace) -> None:
        """Require failed hooks removed without touching ancestor hooks."""
        self.assertTrue(trace.failed)
        self.assertFalse(trace.enabled)
        self.assertEqual(trace.handles, [])
        self.assertFalse(trace.module._forward_pre_hooks)
        self.assertFalse(trace.module._forward_hooks)
        self.assertTrue(trace.owner()._forward_hooks)
        with self.assertRaises(ValueError):
            trace.finish_run(3)

    def test_native_false_proof_accepts_real_exact_bare_linear_and_autograd(self) -> None:
        module = torch.nn.Linear(128, 128, bias=False, dtype=torch.float64)
        original_weight = module.weight
        proof = actual_forward_proof(module, cpu_fixture=False)
        self.assertEqual(proof["route"], "bare_nn_linear")
        self.assertTrue(proof["same_bound_module"])
        self.assertTrue(proof["plain_installed_weight"])
        self.assertFalse(proof["torch_linear_source_identity"]["source_bytes_sha_checked"])
        value = hidden()
        output = module(value)
        output.sum().backward()
        self.assertIs(module.weight, original_weight)
        self.assertIs(type(output), torch.Tensor)
        self.assertIsNotNone(value.grad)
        self.assertIsNotNone(original_weight.grad)
        self.assertEqual(actual_forward_proof(module, cpu_fixture=False), proof)

    def test_native_false_proof_rejects_unknown_subclass_foreign_binding_wrapper_and_bare_dtensor(self) -> None:
        class UnknownLinear(torch.nn.Linear):
            """A subclass is deliberately outside the fixed ordinary Linear route."""

        for mode in ("subclass", "foreign", "wrapper", "unbound", "bare_dtensor", "bias", "changed_class"):
            with self.subTest(mode=mode):
                module = (UnknownLinear(128, 128, bias=False, dtype=torch.float64) if mode == "subclass"
                          else torch.nn.Linear(128, 128, bias=mode == "bias", dtype=torch.float64))
                if mode == "foreign":
                    other = torch.nn.Linear(128, 128, bias=False, dtype=torch.float64)
                    module.forward = other.forward
                elif mode == "wrapper":
                    original = module.forward

                    @functools.wraps(original)
                    def unknown(*args: Any, **kwargs: Any) -> Any:
                        return original(*args, **kwargs)

                    module.forward = unknown
                elif mode == "unbound":
                    module.forward = torch.nn.Linear.forward
                elif mode == "bare_dtensor":
                    module._parameters["weight"] = FixtureWeight(module.weight, 4)
                if mode == "changed_class":
                    with mock.patch.object(torch.nn.Linear, "forward", lambda _self, value: value):
                        with self.assertRaises(ValueError):
                            actual_forward_proof(module, cpu_fixture=False)
                else:
                    with self.assertRaises(ValueError):
                        actual_forward_proof(module, cpu_fixture=False)

    def test_native_false_proof_accepts_actual_fixed_adapter_ast_and_rejects_context_or_double_wrap(self) -> None:
        for mode in ("valid", "inactive_context", "double"):
            with self.subTest(mode=mode):
                model = Toy(wrapped=True)
                module = model.projection
                context = module.forward.__globals__["_LOCAL_COMPUTE_ACTIVE"]
                token = context.set(mode != "inactive_context")
                if mode == "double":
                    _source_fragment("rewriter", "_wrap_module_forward")(module)
                try:
                    if mode == "valid":
                        proof = actual_forward_proof(module, cpu_fixture=False)
                        self.assertEqual(proof["route"], "fixed_single_local_param")
                        self.assertTrue(proof["closure_module_is_actual"])
                        self.assertIs(inspect.getclosurevars(module.forward).nonlocals["module"], module)
                    else:
                        with self.assertRaises(ValueError):
                            actual_forward_proof(module, cpu_fixture=False)
                finally:
                    context.reset(token)

    def test_route_failure_has_no_cpu_fixture_bypass_and_removes_only_own_hooks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = Toy()
            original = model.projection.forward

            @functools.wraps(original)
            def unknown(*args: Any, **kwargs: Any) -> Any:
                return original(*args, **kwargs)

            model.projection.forward = unknown
            trace = make_trace(model, Path(directory))
            trace.start_step(0)
            arm_micro(trace, 0)
            try:
                with self.assertRaises(ValueError):
                    model(hidden())
                self.assert_closed(trace)
            finally:
                cleanup_after_execution(trace)

    def test_both_halves_forward_and_complete_dx_dw_match_mathematics(self) -> None:
        rows, inner, width = CPU_SHAPE
        for tp in (0, 1):
            with self.subTest(tp=tp):
                begin, end = tp * width // 2, (tp + 1) * width // 2
                x = hidden()
                w = torch.nn.Parameter(torch.arange(width // 2 * inner, dtype=torch.float64).reshape(
                    width // 2, inner) / 65536)
                g = torch.arange(rows * width // 2, dtype=torch.float64).reshape(1, rows, width // 2) / 8192
                for arm in ("T", "P"):
                    actual = effective_linear(x, w, arm, begin, end, width)
                    dx, dw = torch.autograd.grad(actual, (x, w), g, retain_graph=True)
                    self.assertTrue(torch.equal(actual, x @ w.t()))
                    self.assertTrue(torch.equal(dx, g @ w))
                    self.assertTrue(torch.equal(dw, g[0].t() @ x[0]))
                    self.assertEqual(tuple(actual.shape), (1, rows, width // 2))
                    self.assertTrue(actual.is_contiguous())
                    self.assertEqual(actual.storage_offset(), 0)

    def test_t_three_step_outputs_gradients_norms_and_updates_are_exact(self) -> None:
        for rank in (0, 4, 5):
            with self.subTest(rank=rank), tempfile.TemporaryDirectory() as directory:
                baseline, observed = Toy(rank), Toy(rank)
                before = run_three(baseline)
                trace = make_trace(observed, Path(directory))
                try:
                    after = run_three(observed, trace)
                finally:
                    cleanup_after_execution(trace)
                for left, right in zip(before, after, strict=True):
                    for key in left:
                        self.assertTrue(torch.equal(left[key], right[key]), key)
                self.assertTrue(torch.equal(baseline.local_parameter, observed.local_parameter))
                self.assertTrue(trace.run_complete)

    def test_restored_to_local_is_the_actual_original_compute_tensor(self) -> None:
        for rank in (4, 5):
            with self.subTest(rank=rank), tempfile.TemporaryDirectory() as directory:
                model = Toy(rank, wrapped=True)
                installed = model.projection.weight
                trace = make_trace(model, Path(directory))
                trace.start_step(0)
                arm_micro(trace, 0)
                native, weights, outputs = torch.nn.functional.linear, [], []
                intervene = trace._intervene

                def watched(x: torch.Tensor, weight: torch.Tensor, bias: Any = None) -> torch.Tensor:
                    weights.append(weight)
                    result = native(x, weight, bias)
                    outputs.append(result)
                    return result

                def at_posthook(module: Any, value: torch.Tensor, original: Any) -> Any:
                    self.assertIs(module.weight, installed)
                    self.assertIs(original, outputs[0])
                    self.assertIs(type(original), torch.Tensor)
                    self.assertEqual(model.model.layers[0].hsdp_scheduler.order, ["ancestor_pre"])
                    returned, record = intervene(module, value, original)
                    self.assertIs(type(returned), torch.Tensor)
                    self.assertTrue(record["return_contract"]["parameter_restore_before_posthook"])
                    return returned, record

                try:
                    with mock.patch.object(torch.nn.functional, "linear", side_effect=watched), \
                            mock.patch.object(trace, "_intervene", side_effect=at_posthook):
                        y = model(hidden())
                    self.assertIs(model.projection.weight, installed)
                    self.assertIs(type(y), torch.Tensor)
                    self.assertIs(y, outputs[1])
                    self.assertEqual(model.model.layers[0].hsdp_scheduler.order, ["ancestor_pre", "ancestor_post"])
                    self.assertIs(weights[0], model.local_parameter)
                    self.assertIs(weights[1], weights[0])
                    y.sum().backward()
                    self.assertIsNotNone(model.local_parameter.grad)
                    self.assertTrue(model.local_parameter.grad.abs().sum() > 0)
                finally:
                    cleanup_after_execution(trace)

    def test_actual_fixed_to_local_source_keeps_object_and_gradient_path(self) -> None:
        method = _source_fragment("dtensor", "to_local", class_name="DTensor")
        local = torch.nn.Parameter(torch.ones(2, 3, dtype=torch.float64))
        holder = SimpleNamespace(_local_tensor=local)
        result = method(holder)
        self.assertIs(result, local)
        result.square().sum().backward()
        self.assertTrue(torch.equal(local.grad, torch.full_like(local, 2)))

    def test_target_replacement_and_nontarget_original_object_are_explicit(self) -> None:
        for rank in (0, 4):
            with self.subTest(rank=rank), tempfile.TemporaryDirectory() as directory:
                model = Toy(rank)
                trace = make_trace(model, Path(directory))
                trace.start_step(0)
                arm_micro(trace, 0)
                native, outputs = torch.nn.functional.linear, []

                def watched(*args: Any, **kwargs: Any) -> torch.Tensor:
                    out = native(*args, **kwargs)
                    outputs.append(out)
                    return out

                try:
                    with mock.patch.object(torch.nn.functional, "linear", side_effect=watched):
                        returned = model(hidden())
                    self.assertEqual(len(outputs), 2 if rank == 4 else 1)
                    if rank == 4:
                        self.assertIs(returned, outputs[1])
                        self.assertIsNot(returned, outputs[0])
                        self.assertTrue(torch.equal(returned, outputs[0]))
                    else:
                        self.assertIs(returned, outputs[0])
                finally:
                    cleanup_after_execution(trace)

    def test_only_first_target_is_changed_and_hooks_remove_after_step_one(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = Toy()
            trace = make_trace(model, Path(directory), "P")
            run_three(model, trace)
            records, _sha = read_run_file(Path(directory), 4)
            self.assertEqual(sum(row["kind"] == "intervention" for row in records), 1)
            self.assertEqual(sum(row["kind"] == "call" for row in records), 2)
            self.assertEqual(trace.target_calls, 1)
            self.assertEqual(trace.handles, [])
            self.assertFalse(model.projection._forward_hooks)
            self.assertTrue(model.model.layers[0]._forward_hooks)

    def test_full_native_output_coordinates_cover_left_and_right(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            validated = fixture_run(Path(directory), "P")
            original, effective = validated["complete"]["original"], validated["complete"]["effective"]
            self.assertEqual(set(effective["blocks"]), {(0, 32, 0, 128), (0, 32, 128, 256)})
            self.assertEqual(original["blocks"], effective["blocks"])
            self.assertEqual(len(validated["selected"]), 2)
            self.assertEqual(validated["scope"], "cpu-fixture-only")

    def test_original_parameter_input_versions_rng_and_graph_are_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = Toy()
            installed = model.projection.weight
            rng = torch.get_rng_state().clone()
            trace = make_trace(model, Path(directory), "P")
            run_three(model, trace)
            records, _sha = read_run_file(Path(directory), 4)
            result = next(row["result"] for row in records if row["kind"] == "intervention")
            self.assertTrue(all(result["invariants"].values()))
            self.assertIs(model.projection.weight, installed)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertGreater(result["explicit_forward_extra_tensor_bytes_bound"], 0)

    def test_padding_values_and_tensor_temporaries_do_not_escape_the_hook(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = Toy()
            trace = make_trace(model, Path(directory), "P")
            trace.start_step(0)
            arm_micro(trace, 0)
            native, references = torch.nn.functional.linear, []

            def watched(x: torch.Tensor, weight: torch.Tensor, bias: Any = None) -> torch.Tensor:
                y = native(x, weight, bias)
                references.append(weakref.ref(y))
                return y

            try:
                with mock.patch.object(torch.nn.functional, "linear", side_effect=watched):
                    y = model(hidden())
                y.sum().backward()
                del y
                gc.collect()
                self.assertIsNone(trace.pre_metadata)
                self.assertIsNone(trace.compute_weight_ref)
                self.assertTrue(all(reference() is None for reference in references))
            finally:
                cleanup_after_execution(trace)

    def test_detached_to_local_and_missing_autograd_conditions_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = Toy(wrapped=True)
            model.projection.weight.to_local = lambda: model.local_parameter.detach()
            with self.assertRaises(ValueError):
                live_weight(model.projection, cpu_fixture=True)
        x, w = hidden(), torch.nn.Parameter(torch.ones(128, 128, dtype=torch.float64))
        for mode in ("no_grad", "input_detached", "weight_detached", "bad_offset"):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                if mode == "no_grad":
                    with torch.no_grad():
                        effective_linear(x, w, "P", 0, 128, 256)
                else:
                    effective_linear(x.detach() if mode == "input_detached" else x,
                                     w.detach() if mode == "weight_detached" else w, "P",
                                     1 if mode == "bad_offset" else 0, 128, 256)

    def test_target_layout_finiteness_and_installed_storage_fail_closed(self) -> None:
        for mode in ("nonfinite_input", "nonfinite_weight", "wrong_shape", "no_input_grad", "storage",
                     "non_plain_result"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                model = Toy()
                trace = make_trace(model, Path(directory))
                trace.start_step(0)
                arm_micro(trace, 0)
                value = hidden()
                if mode == "nonfinite_input":
                    value = torch.full_like(value, float("nan"), requires_grad=True)
                elif mode == "nonfinite_weight":
                    with torch.no_grad():
                        model.local_parameter.fill_(float("inf"))
                elif mode == "wrong_shape":
                    value = value[:, :-1].contiguous()
                elif mode == "no_input_grad":
                    value = value.detach()
                elif mode == "storage":
                    model.projection.weight = torch.nn.Parameter(model.local_parameter.detach().clone())
                try:
                    if mode == "non_plain_result":
                        with mock.patch.object(torch.nn.functional, "linear",
                                               return_value=FixtureWeight(model.local_parameter, model.rank)):
                            with self.assertRaisesRegex(ValueError, "plain torch.Tensor"):
                                model(value)
                    else:
                        with self.assertRaises(ValueError):
                            model(value)
                    self.assert_closed(trace)
                finally:
                    cleanup_after_execution(trace)

    def test_wrong_target_repeated_missing_and_deferred_fifo_are_rejected(self) -> None:
        for mode in ("wrong_target", "repeat", "missing", "deferred"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                model = Toy()
                trace = make_trace(model, Path(directory))
                trace.start_step(0)
                try:
                    if mode == "wrong_target":
                        wrong = batch(0)
                        trace.queue_batch(*wrong)
                        with self.assertRaises(ValueError):
                            trace.begin_microbatch({**wrong[0], **wrong[1]})
                    else:
                        if mode == "deferred":
                            def get_batch(_iterator: Any) -> Any:
                                return batch(4)

                            def prepare(_iterator: Any, _count: int) -> Any:
                                return {}, iter([batch(4), batch(5)])

                            get_batch.prepare_step = prepare
                            wrapped = wrap_get_batch(get_batch, trace)
                            _, iterator = wrapped.prepare_step(None, 2)
                            self.assertFalse(trace.pending)
                            fields = next(iterator)
                            trace.begin_microbatch({**fields[0], **fields[1]})
                        else:
                            arm_micro(trace, 0)
                        if mode == "repeat":
                            model(hidden())
                            with self.assertRaises(ValueError):
                                model(hidden())
                        else:
                            with self.assertRaises(ValueError):
                                trace.finish_step(1)
                    self.assert_closed(trace)
                finally:
                    cleanup_after_execution(trace)

    def test_unknown_projection_hooks_reject_and_cleanup_keeps_others(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = Toy()
            existing = model.projection.register_forward_hook(lambda _m, _a, _out: None)
            with self.assertRaises(ValueError):
                make_trace(model, Path(directory))
            existing.remove()
            trace = make_trace(model, Path(directory))
            extra = model.projection.register_forward_hook(lambda _m, _a, _out: None)
            trace.abort()
            self.assertEqual(set(model.projection._forward_hooks), {extra.id})
            extra.remove()

    def test_original_forward_exception_and_parameter_restore_are_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = Toy(wrapped=True)
            installed = model.projection.weight
            trace = make_trace(model, Path(directory))
            trace.start_step(0)
            arm_micro(trace, 0)
            error = RuntimeError("original Linear failure")
            with mock.patch.object(torch.nn.functional, "linear", side_effect=error):
                with self.assertRaises(RuntimeError) as caught:
                    try:
                        model(hidden())
                    finally:
                        cleanup_after_execution(trace)
            self.assertIs(caught.exception, error)
            self.assertIs(model.projection.weight, installed)
            self.assert_closed(trace)

    def test_complete_eight_rank_json_and_negative_contracts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture_run(Path(directory))
            good, _sha = read_run(Path(directory))
            for mode in ("missing_rank", "missing_half", "duplicate_tile", "wrong_interval", "no_graph", "bad_stats"):
                with self.subTest(mode=mode):
                    records = copy.deepcopy(good)
                    result = next(row["result"] for row in records[4] if row["kind"] == "intervention")
                    if mode == "missing_rank":
                        del records[7]
                    elif mode == "missing_half":
                        records[5] = [row for row in records[5] if row["kind"] != "intervention"]
                    elif mode == "duplicate_tile":
                        result["effective"]["blocks"].append(result["effective"]["blocks"][0])
                    elif mode == "wrong_interval":
                        result["effective"]["bounds"][2] = 128
                    elif mode == "no_graph":
                        result["invariants"]["effective_graph_present"] = False
                    else:
                        result["original_vs_effective"]["nonzero_count"] = 1
                    with self.assertRaises(ValueError):
                        validate_run(records, cpu_fixture=True)
            with self.assertRaises(ValueError):
                validate_run(good)

    def test_capture_finally_and_effective_math_have_no_detached_branch(self) -> None:
        path = Path(__file__).with_name("native_qb_padding.py")
        tree = ast.parse(path.read_bytes())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == "effective_linear")
        calls = [node for node in ast.walk(function) if isinstance(node, ast.Attribute)]
        self.assertFalse(any(node.attr in ("detach", "data", "no_grad", "copy_", "all_reduce") for node in calls))
        capture = ast.parse(path.with_name("capture_qb_padding_training.py").read_bytes())
        main = next(node for node in capture.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        self.assertTrue(any(isinstance(node, ast.Try) and node.finalbody for node in ast.walk(main)))
        self.assertNotIn("always_call", path.read_text(encoding="utf-8"))

    def test_t_native_and_three_step_gates_reject_p_inference_on_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            left, right = directory / "T", directory / "P"
            left.mkdir()
            right.mkdir()
            control, padded = fixture_run(left), fixture_run(right, "P")
            key = control["target_batch"]
            old = {"complete": {(key, MODULE, stage): control["complete"][stage]
                                 for stage in ("input", "weight")}}
            old["complete"][(key, MODULE, "output")] = control["complete"]["original"]
            report = compare_native(control, padded, old, old)
            self.assertEqual(report["status"], "PAD_TARGET_NATIVE_MATCHES_OLD_FSDP")
            bad = copy.deepcopy(control)
            first = next(iter(bad["complete"]["effective"]["blocks"]))
            bad["complete"]["effective"]["blocks"][first] = "f" * 64
            self.assertFalse(compare_native(bad, padded, old, old)["native_T_control_passed"])
            scalar = [{step: {**{name: 1.0 for name in METRICS}, "input_identity": {"micro_steps": 2}}
                       for step in (1, 2, 3)} for _rank in range(8)]
            states = {f"{prefix}-rank{rank}.json": {} for rank in range(8)
                      for prefix in ("initial-state", "runtime-flags")}
            states["config.resolved.json"] = {name: {} for name in RECIPE_FIELDS}
            states["config.resolved.json"]["training"] = {"train_iters": 3}
            self.assertTrue(prefix_transparency(scalar, scalar, states, states)["passed"])
            bad_scalar = copy.deepcopy(scalar)
            bad_scalar[7][3]["training/grad_norm"] += 1
            self.assertFalse(prefix_transparency(bad_scalar, scalar, states, states)["passed"])
            bad_states = copy.deepcopy(states)
            bad_states["config.resolved.json"]["optimizer"] = {"changed": True}
            self.assertFalse(prefix_transparency(scalar, scalar, bad_states, states)["passed"])


def read_run_file(directory: Path, rank: int) -> tuple[list[dict], str]:
    """Read one real CPU fixture output without claiming eight-rank completeness."""
    path = directory / f"qb-padding-rank{rank}.jsonl"
    return [json.loads(line) for line in path.read_bytes().splitlines()], str(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
