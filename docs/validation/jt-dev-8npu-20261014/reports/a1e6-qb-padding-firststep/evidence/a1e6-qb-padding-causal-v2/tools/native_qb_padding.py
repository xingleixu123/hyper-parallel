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

"""Active target-only T/P posthooks; original TP/FSDP forwards remain installed."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import asdict
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any, Mapping
import weakref

import torch

from qb_padding_contract import (
    CPU_SHAPE, FROZEN_FILES, INVARIANTS, MODULE, NATIVE_SHAPE, SCHEMA, SOURCE_COMMIT,
    SOURCE_FILES, TARGET_BATCH, TARGET_RANKS, frozen_support, region,
)


SUPPORT = frozen_support(tensor=True)
FROZEN = SUPPORT.FROZEN
Topology = SUPPORT.Topology
MaterializationBinding = SUPPORT.MaterializationBinding
wrap_get_batch = SUPPORT.wrap_get_batch
batch_identity = SUPPORT.batch_identity
physical_metadata = SUPPORT.physical_metadata
CALLS = {"q_b_pre": 1, "q_b_post": 1}
ORIGINAL_LINEAR_FORWARD = torch.nn.Linear.forward


def function_identity(function: Any) -> tuple[int, int]:
    """Identify a stored function or freshly bound method without comparing method wrappers."""
    return id(getattr(function, "__func__", function)), id(getattr(function, "__self__", None))


def _source_digest(function: Any) -> str:
    path = inspect.getsourcefile(function)
    if path is None:
        raise ValueError("The actual callable has no fixed source file")
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def live_weight(module: Any, *, cpu_fixture: bool = False) -> tuple[torch.Tensor, dict[str, Any]]:
    """Use module.weight itself, never a detached observation helper's returned Tensor."""
    installed = module.weight
    if type(installed) in (torch.Tensor, torch.nn.Parameter):
        local = installed
        kind = "same_plain_installed_tensor"
    else:
        method = getattr(installed, "to_local", None)
        fixture = cpu_fixture and getattr(installed, "_padding_CPU_fixture_only", False) is True
        if (not callable(method) or (not fixture and (
                type(installed).__module__ != "hyper_parallel.core.dtensor.dtensor"
                or type(installed).__name__ != "DTensor"
                or _source_digest(method) != SOURCE_FILES["dtensor"][1]))):
            raise ValueError("Unknown autograd-preserving installed weight API")
        local = method()
        if local is not getattr(installed, "_local_tensor", None):
            raise ValueError("to_local did not return the original local Tensor object")
        kind = "fixed_to_local_direct_reference" if not fixture else "CPU_fixture_direct_reference"
    if type(local) not in (torch.Tensor, torch.nn.Parameter) or local.requires_grad is not True:
        raise ValueError("The actual local weight must retain its existing autograd path")
    return local, {"kind": kind, "installed_object_id": id(installed), "local_tensor_id": id(local),
                   "requires_grad": local.requires_grad, "is_leaf": local.is_leaf,
                   "physical": physical_metadata(local)}


def _linear_source_identity() -> dict[str, Any]:
    """Check the installed original function's file identity, not a source-byte SHA."""
    function = ORIGINAL_LINEAR_FORWARD
    path = inspect.getsourcefile(function)
    source_module = torch.nn.modules.linear
    code = getattr(function, "__code__", None)
    if (torch.nn.Linear.forward is not function or source_module.Linear is not torch.nn.Linear
            or function.__module__ != "torch.nn.modules.linear" or function.__qualname__ != "Linear.forward"
            or path is None or code is None or not getattr(source_module, "__file__", None)
            or Path(path).resolve() != Path(source_module.__file__).resolve()
            or Path(code.co_filename).resolve() != Path(path).resolve()):
        raise ValueError("The original installed torch.nn.Linear.forward identity changed or is unproven")
    return {"module": function.__module__, "qualname": function.__qualname__,
            "source_file": str(Path(path).resolve()), "code_filename": str(Path(code.co_filename).resolve()),
            "file_identity_checked": True, "source_bytes_sha_checked": False}


def actual_forward_proof(module: Any, *, cpu_fixture: bool = False) -> dict[str, Any]:
    """Prove the exact bare or fixed single-adapter route; CPU fixtures use the same checks."""
    del cpu_fixture
    if type(module) is not torch.nn.Linear or module.bias is not None:
        raise ValueError("The actual target must be the exact original bias-free Linear class")
    function = module.forward
    source = _linear_source_identity()
    if (getattr(function, "__self__", None) is module
            and getattr(function, "__func__", None) is ORIGINAL_LINEAR_FORWARD
            and getattr(function, "__wrapped__", None) is None):
        if (type(module.weight) not in (torch.Tensor, torch.nn.Parameter)
                or module.weight.requires_grad is not True):
            raise ValueError("A bare Linear forward requires its actual plain differentiable installed weight")
        return {"verified": True, "route": "bare_nn_linear", "original_linear": "torch.nn.Linear.forward",
                "same_bound_module": True, "class_forward_unchanged": True, "plain_installed_weight": True,
                "installed_weight_object_id": id(module.weight), "torch_linear_source_identity": source,
                "forward_identity": function_identity(function)}
    if (_source_digest(function) != SOURCE_FILES["rewriter"][1]
            or getattr(function, "__code__", None) is None
            or function.__code__.co_name != "local_param_forward"):
        raise ValueError("Actual projection lacks the fixed local-parameter adapter")
    closure = inspect.getclosurevars(function)
    original = closure.nonlocals.get("original_forward")
    context = function.__globals__.get("_LOCAL_COMPUTE_ACTIVE")
    if (closure.nonlocals.get("module") is not module or getattr(original, "__self__", None) is not module
            or getattr(original, "__func__", None) is not ORIGINAL_LINEAR_FORWARD
            or getattr(function, "__wrapped__", None) is not original
            or context is None or context.get() is not True):
        raise ValueError("Original Linear module/context/parameter-unwrapping route is unproven")
    return {"verified": True, "route": "fixed_single_local_param", "rewriter_sha256": SOURCE_FILES["rewriter"][1],
            "original_linear": "torch.nn.Linear.forward", "local_compute_active": True,
            "closure_module_is_actual": True, "torch_linear_source_identity": source,
            "forward_identity": function_identity(function)}


def effective_linear(value: torch.Tensor, weight: torch.Tensor, arm: str,
                     begin: int, end: int, full_width: int) -> torch.Tensor:
    """Run only ordinary differentiable Linear/cat/slice; do not detach the effective graph."""
    if (arm not in ("T", "P") or not torch.is_grad_enabled() or not value.requires_grad
            or not weight.requires_grad or begin < 0 or end > full_width or end - begin != weight.shape[0]
            or full_width != 2 * weight.shape[0] or begin not in (0, weight.shape[0])
            or value.shape[-1] != weight.shape[1]):
        raise ValueError("Invalid active q_b arm, global interval, or autograd condition")
    if arm == "T":
        return torch.nn.functional.linear(value, weight, None)
    zeros = weight.new_zeros((weight.shape[0], weight.shape[1]), requires_grad=False)
    padded = torch.cat((weight, zeros) if begin == 0 else (zeros, weight), dim=0)
    whole = torch.nn.functional.linear(value, padded, None)
    return whole[..., begin:end].contiguous()


def native_identity(value: torch.Tensor, stage: str, topology: Any, copies: dict[str, int], *,
                    cpu_fixture: bool = False) -> dict[str, Any]:
    """Stream native bytes through bounded slabs and map every tile to global coordinates."""
    expected = region(stage, asdict(topology), cpu_fixture=cpu_fixture)
    record = SUPPORT.hash_matrix(value, weight=stage == "weight", copies=copies)
    rb, _re, cb, _ce = expected["bounds"]
    for block in record["blocks"]:
        coordinates = block["coordinates"]
        block["coordinates"] = [coordinates[0] + rb, coordinates[1] + rb,
                                coordinates[2] + cb, coordinates[3] + cb]
    record.update(expected)
    record["physical"] = physical_metadata(value)
    return record


class QBPaddingTrace(FROZEN.NativeForwardTrace):
    """Bind the frozen batch FIFO; replace only the selected two-peer q_b returns."""

    def __init__(self, model: Any, output: Path, topology: Any, binding: Any,
                 runtime_proof: Mapping[str, Any], arm: str, owner_module: Any, *,
                 cpu_fixture: bool = False, fixture_target_batch: str | None = None) -> None:
        if (arm not in ("T", "P") or topology.tp_size != 2 or topology.dp_size != 4
                or not topology.sequence_parallel or topology.cp_size != 1 or topology.pp_size != 1
                or topology.ep_size != 1 or runtime_proof.get("verified") is not True):
            raise ValueError("Only the fixed TP2 three-step active arm is supported")
        if not cpu_fixture and fixture_target_batch is not None:
            raise ValueError("A native arm cannot change the selected actual batch")
        self.topology, self.binding, self.arm, self.cpu_fixture = topology, binding, arm, cpu_fixture
        self.shape = CPU_SHAPE if cpu_fixture else NATIVE_SHAPE
        self.target_batch = fixture_target_batch if cpu_fixture else TARGET_BATCH
        if self.target_batch is None or len(self.target_batch) != 64:
            raise ValueError("An explicit target identity is required")
        self.module = dict(model.named_modules())[MODULE]
        if type(self.module) is not torch.nn.Linear or self.module.bias is not None:
            raise ValueError("Exactly the original bias-free ordinary Linear is required")
        if self.module._forward_pre_hooks or self.module._forward_hooks:
            raise ValueError("The active hook must install before any projection observer hooks")
        global_hooks = torch.nn.modules.module
        if global_hooks._global_forward_pre_hooks or global_hooks._global_forward_hooks:
            raise ValueError("Unknown global hooks could change the active ordering")
        self.owner = weakref.ref(owner_module)
        self.owner_hooks = self._owner_hooks()
        self.forward_identity = function_identity(self.module.forward)
        self.path = output / f"qb-padding-rank{topology.rank}.jsonl"
        self.pending, self.calls, self.seen_batches, self.handles = deque(), Counter(), set(), []
        self.batch, self.micro = None, -1
        self.enabled = self.first_step_started = self.first_step_complete = False
        self.failed = self.run_complete = False
        self.target_calls = 0
        self.pre_metadata = None
        self.compute_weight_ref = None
        self._event("metadata", schema=SCHEMA, source_commit=SOURCE_COMMIT, arm=arm,
                    scope="cpu-fixture-only" if cpu_fixture else "native-candidate", topology=asdict(topology),
                    target_batch=self.target_batch, runtime_proof=dict(runtime_proof),
                    frozen_inputs={name: digest for name, (_path, digest) in FROZEN_FILES.items()},
                    source_inputs={name: digest for name, (_path, digest) in SOURCE_FILES.items()},
                    original_forward_preserved=True, active_hook_before_passive_observers=True,
                    limitation="Active causal short arm; not a repaired or accepted200step run")
        try:
            self.handles.append(self.module.register_forward_pre_hook(self._pre_hook, with_kwargs=True))
            self.handles.append(self.module.register_forward_hook(self._post_hook, with_kwargs=True, prepend=True))
        except BaseException:
            self._fail_closed()
            raise

    def _owner_hooks(self) -> tuple[tuple, tuple]:
        owner = self.owner()
        if owner is None:
            raise ValueError("The actual ancestor owner expired")
        return (tuple((key, function_identity(value)) for key, value in owner._forward_pre_hooks.items()),
                tuple((key, function_identity(value)) for key, value in owner._forward_hooks.items()))

    def _check_hooks(self) -> None:
        if (set(self.module._forward_pre_hooks) != {self.handles[0].id}
                or set(self.module._forward_hooks) != {self.handles[1].id}
                or self.module.bias is not None
                or self._owner_hooks() != self.owner_hooks
                or function_identity(self.module.forward) != self.forward_identity):
            raise ValueError("Projection/ancestor hooks or original forward changed after active installation")

    def _event(self, kind: str, **fields: Any) -> None:
        if kind == "metadata":
            with self.path.open("x", encoding="utf-8") as handle:
                handle.write(json.dumps({"kind": kind, "rank": self.topology.rank, **fields},
                                        sort_keys=True, allow_nan=False) + "\n")
            return
        super()._event(kind, **fields)

    def _fail_closed(self) -> None:
        self.failed, self.enabled = True, False
        self.pre_metadata = self.compute_weight_ref = None
        self.close()

    def abort(self) -> None:
        """Remove only this diagnostic's handles without declaring completion."""
        self._fail_closed()

    def _require_active(self) -> None:
        if self.failed:
            raise ValueError("Failed active arm cannot restart or complete")

    def start_step(self, completed_steps: int) -> None:
        """Arm once; later optimizer steps have no diagnostic hooks or replacements."""
        self._require_active()
        try:
            if completed_steps == 0:
                if self.first_step_started:
                    raise ValueError("First step cannot restart")
                self.first_step_started = True
            elif not self.first_step_complete:
                raise ValueError("Missing first-step completion")
            super().start_step(completed_steps)
        except BaseException:
            self._fail_closed()
            raise

    def queue_batch(self, model_inputs: Mapping[str, Any], loss_inputs: Mapping[str, Any]) -> None:
        """Use the fixed actual-consumption eager/deferred FIFO."""
        self._require_active()
        try:
            super().queue_batch(model_inputs, loss_inputs)
        except BaseException:
            self._fail_closed()
            raise

    def begin_microbatch(self, fields: Mapping[str, Any]) -> None:
        """Route by actual SHA, then require the frozen target micro/rank contract."""
        self._require_active()
        try:
            super().begin_microbatch(fields)
            expected = self.topology.rank in TARGET_RANKS and self.micro == 0
            if self.enabled and (self.batch == self.target_batch) is not expected:
                raise ValueError("Actual target batch moved from its fixed two-peer micro0 use-point")
        except BaseException:
            self._fail_closed()
            raise

    def _check_previous_calls(self) -> None:
        if self.batch is not None and dict(self.calls) != CALLS:
            raise ValueError("Missing/repeated q_b pre/output call")

    @staticmethod
    def _input(args: tuple, kwargs: Mapping[str, Any]) -> torch.Tensor:
        if len(args) == 1 and not kwargs:
            return args[0]
        if not args and set(kwargs) == {"input"}:
            return kwargs["input"]
        raise ValueError("Ambiguous original Linear arguments")

    def _checked(self, tensor: Any, shape: tuple[int, ...]) -> dict[str, Any]:
        if (type(tensor) not in (torch.Tensor, torch.nn.Parameter) or tuple(tensor.shape) != shape
                or tensor.dtype != (torch.float64 if self.cpu_fixture else torch.bfloat16)
                or tensor.device.type != ("cpu" if self.cpu_fixture else "npu") or not tensor.is_contiguous()
                or tensor.storage_offset() != 0 or tensor.requires_grad is not True):
            raise ValueError("Unproven actual q_b shape/layout/dtype/gradient path")
        physical = physical_metadata(tensor)
        if physical["storage_nbytes"] != tensor.numel() * tensor.element_size():
            raise ValueError("Actual q_b storage capacity differs from the compact contract")
        return physical

    def _pre_hook(self, module: Any, args: tuple, kwargs: Mapping[str, Any]) -> None:
        if not self.enabled:
            return
        try:
            self._check_hooks()
            if self.batch is None:
                raise ValueError("Original q_b ran before batch binding")
            self.calls["q_b_pre"] += 1
            if self.calls["q_b_pre"] != 1:
                raise ValueError("Repeated q_b call")
            if self.batch != self.target_batch:
                return
            rows, inner, width = self.shape
            value = self._input(args, kwargs)
            if not torch.is_grad_enabled():
                raise ValueError("The active arm must use the original grad-enabled context")
            weight, live = live_weight(module, cpu_fixture=self.cpu_fixture)
            _observation_only, proof = self.binding.check(
                module, "q_b_proj", self.topology, region("weight", asdict(self.topology),
                                                       cpu_fixture=self.cpu_fixture), cpu_fixture=self.cpu_fixture)
            del _observation_only
            self.compute_weight_ref = weakref.ref(weight)
            copies = {"slab_copy_calls": 0, "slab_copy_bytes": 0, "largest_slab_bytes": 0,
                      "cpu_native_bytes_bound": 0}
            self.pre_metadata = {
                "input_physical": self._checked(value, (1, rows, inner)),
                "weight_physical": self._checked(weight, (width // 2, inner)),
                "input_object_id": id(value), "installed_weight_object_id": id(module.weight),
                "versions": [value._version, weight._version], "live_weight": live,
                "forward_proof": actual_forward_proof(module, cpu_fixture=self.cpu_fixture),
                "materialization": proof, "copies": copies,
                "rng_before": SUPPORT.rng_digest(cpu_fixture=self.cpu_fixture, device=value.device),
                "input": native_identity(value, "input", self.topology, copies, cpu_fixture=self.cpu_fixture),
                "weight": native_identity(weight, "weight", self.topology, copies, cpu_fixture=self.cpu_fixture),
            }
        except BaseException:
            self._fail_closed()
            raise

    def _intervene(self, module: Any, value: torch.Tensor, original: torch.Tensor) -> tuple[torch.Tensor, dict]:
        rows, inner, width = self.shape
        before = self.pre_metadata
        copies = before["copies"]
        weight, current = live_weight(module, cpu_fixture=self.cpu_fixture)
        self._checked(value, (1, rows, inner))
        self._checked(weight, (width // 2, inner))
        if type(original) is not torch.Tensor:
            raise ValueError("The fixed local Linear posthook must return a plain torch.Tensor, not a DTensor")
        self._checked(original, (1, rows, width // 2))
        if (id(module.weight) != before["installed_weight_object_id"] or id(value) != before["input_object_id"]
                or self.compute_weight_ref() is not weight or current != before["live_weight"]
                or [value._version, weight._version] != before["versions"]
                or actual_forward_proof(module, cpu_fixture=self.cpu_fixture) != before["forward_proof"]):
            raise ValueError("Original local_param_forward did not restore the exact original live weight")
        original_identity = native_identity(original, "output", self.topology, copies, cpu_fixture=self.cpu_fixture)
        original_version, original_graph = original._version, original.grad_fn
        rng_before = before["rng_before"]
        grad_mode = torch.is_grad_enabled()
        begin, end = region("output", asdict(self.topology), cpu_fixture=self.cpu_fixture)["bounds"][2:]
        effective = effective_linear(value, weight, self.arm, begin, end, width)
        if type(effective) is not torch.Tensor:
            raise ValueError("The active branch changed the original plain Tensor output contract")
        self._checked(effective, (1, rows, width // 2))
        pair = SUPPORT.compare_outputs(
            {"original": (original, begin), "effective": (effective, begin)},
            {"original_vs_effective": ("original", "effective")}, self.shape, begin, end, copies)
        if not pair["comparisons"]["original_vs_effective"]["finite"]:
            raise ValueError("Nonfinite effective Linear output")
        output_identity = native_identity(effective, "output", self.topology, copies, cpu_fixture=self.cpu_fixture)
        current_x = native_identity(value, "input", self.topology, copies, cpu_fixture=self.cpu_fixture)
        current_w = native_identity(weight, "weight", self.topology, copies, cpu_fixture=self.cpu_fixture)
        current_original = native_identity(original, "output", self.topology, copies, cpu_fixture=self.cpu_fixture)
        _observation_only, final_proof = self.binding.check(
            module, "q_b_proj", self.topology, region("weight", asdict(self.topology),
                                                   cpu_fixture=self.cpu_fixture), cpu_fixture=self.cpu_fixture)
        del _observation_only
        invariants = {
            "input_native_unchanged": before["input"] == current_x,
            "weight_native_unchanged": before["weight"] == current_w,
            "original_native_unchanged": original_identity == current_original,
            "rng_unchanged": rng_before == SUPPORT.rng_digest(cpu_fixture=self.cpu_fixture, device=value.device),
            "input_object_preserved": id(value) == before["input_object_id"],
            "weight_object_restored": id(module.weight) == before["installed_weight_object_id"],
            "same_live_compute_weight": self.compute_weight_ref() is weight,
            "versions_unchanged": [value._version, weight._version] == before["versions"]
            and original._version == original_version,
            "original_graph_preserved": original.grad_fn is original_graph,
            "effective_graph_present": effective.requires_grad and effective.grad_fn is not None,
            "grad_mode_unchanged": grad_mode is True and torch.is_grad_enabled() is True,
            "forward_unchanged": function_identity(module.forward) == self.forward_identity,
        }
        if set(invariants) != set(INVARIANTS) or not all(invariants.values()):
            raise ValueError("Active short arm changed original input/weight/output/RNG/graph state")
        exact = original_identity["blocks"] == output_identity["blocks"]
        result = {"input": before["input"], "weight": before["weight"], "original": original_identity,
                  "effective": output_identity, "original_vs_effective": pair["comparisons"]["original_vs_effective"],
                  "original_equals_effective_native": exact, "invariants": invariants,
                  "compute_weight_proof": {"same_original_local_tensor": True,
                                           "at_pre": before["live_weight"], "after_restore": current,
                                           "actual_forward_route": before["forward_proof"]},
                  "return_contract": {"original_type": "torch.Tensor", "effective_type": "torch.Tensor",
                                      "same_shape_dtype_device": original.shape == effective.shape
                                      and original.dtype == effective.dtype and original.device == effective.device,
                                      "parameter_restore_before_posthook": True, "DTensor_rewrap": False},
                  "initial_materialization": before["materialization"], "final_materialization": final_proof,
                  "copies": copies, "global_features": [begin, end], "grad_enabled": True,
                  "status": ("T_NATIVE_CONTROL_MATCH" if exact else "T_NATIVE_CONTROL_MISMATCH")
                  if self.arm == "T" else "P_ACTIVE_DIAGNOSTIC_ONLY",
                  "explicit_forward_extra_tensor_bytes_bound": ((width // 2) * inner + width * inner
                                                                  + rows * width + rows * (width // 2))
                  * weight.element_size() if self.arm == "P" else effective.numel() * effective.element_size(),
                  "backward_and_backend_workspace_bound": "NOT_ESTABLISHED; measure actual lifetime peaks",
                  "CPU_tensor_bytes_bound": copies["cpu_native_bytes_bound"] + 4 * 2**20,
                  "not_a_performance_recommendation": True}
        return effective, result

    def _post_hook(self, module: Any, args: tuple, kwargs: Mapping[str, Any], original: Any) -> Any:
        if not self.enabled:
            return None
        try:
            self._check_hooks()
            self.calls["q_b_post"] += 1
            if self.calls["q_b_post"] != 1 or self.calls["q_b_pre"] != 1:
                raise ValueError("Missing/repeated original q_b call")
            selected = self.batch == self.target_batch
            self._event("call", step=1, micro=self.micro, module=MODULE, batch_sha256=self.batch,
                        intervened=selected)
            if not selected:
                return None
            if self.pre_metadata is None or self.target_calls:
                raise ValueError("Missing or repeated selected use-point")
            effective, result = self._intervene(module, self._input(args, kwargs), original)
            self.target_calls += 1
            self._event("intervention", step=1, micro=self.micro, module=MODULE,
                        batch_sha256=self.batch, result=result)
            self.pre_metadata = self.compute_weight_ref = None
            return effective
        except BaseException:
            self._fail_closed()
            raise

    def finish_step(self, completed_steps: int) -> None:
        """Remove active hooks after step one; subsequent updates keep the original path."""
        self._require_active()
        try:
            active = self.enabled
            if active and self.target_calls != int(self.topology.rank in TARGET_RANKS):
                raise ValueError("Missing selected two-peer target call")
            super().finish_step(completed_steps)
            if active:
                self.first_step_complete = True
        except BaseException:
            self._fail_closed()
            raise

    def finish_run(self, completed_steps: int) -> None:
        """Completion records execution, not T transparency or P causal acceptance."""
        self._require_active()
        if not self.first_step_complete or completed_steps != 3:
            self._fail_closed()
            raise ValueError("Three normal training steps are required")
        self.close()
        self._event("run_complete", completed_steps=3, target_calls=self.target_calls)
        self.run_complete = True


def cleanup_after_execution(trace: QBPaddingTrace | None) -> None:
    """Use an outer finally for original model errors; never install always-call cleanup hooks."""
    if trace is not None:
        if trace.run_complete:
            trace.close()
        else:
            trace.abort()
