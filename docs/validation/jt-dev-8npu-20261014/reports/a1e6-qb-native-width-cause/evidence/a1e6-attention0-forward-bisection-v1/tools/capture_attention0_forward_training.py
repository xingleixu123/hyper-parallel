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
"""Attach use-point attention observation only after native TP/FSDP construction."""

import hashlib
import inspect
import os
from pathlib import Path
from typing import Any

import torch

import capture_training as cap
from capture_a1e6_training import RuntimeEvidenceCallback
from hyper_parallel.trainer.callbacks.base import Callback

from attention0_contract import MODEL_SOURCE_SHA256, PARENT, REWRITER_SHA256, SOURCE_COMMIT, projection_shapes
from native_attention0_trace import (
    Attention0ForwardTrace, MaterializationBinding, Topology, cleanup_after_execution,
    placement_metadata, wrap_get_batch,
)


ORIGINAL_TRAINER = cap.TextTrainer
ACTIVE_TRACE: Attention0ForwardTrace | None = None


def _source_sha(value: Any) -> str:
    path = inspect.getsourcefile(value)
    if path is None:
        raise ValueError("Runtime callable has no readable fixed source file")
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _boundary(module: Any, tp_size: int) -> dict[str, Any] | None:
    function = module.forward
    found, seen = [], set()
    while function is not None and id(function) not in seen:
        seen.add(id(function))
        if len(seen) > 8:
            raise ValueError("Unknown runtime wrapper depth")
        values = inspect.getclosurevars(function).nonlocals
        if "boundary" in values:
            code = function.__func__.__code__ if inspect.ismethod(function) else function.__code__
            if (code.co_name not in ("production_forward", "local_region_forward")
                    or values.get("validate_mode", False) is not False or _source_sha(function) != REWRITER_SHA256):
                raise ValueError("Unknown or validate-mode compiled boundary")
            boundary = values["boundary"]
            names = tuple(boundary.mesh_dim_names)
            if "tp" not in names:
                raise ValueError("Compiled boundary has no explicit TP dimension")
            axis = names.index("tp")
            plans = {}
            for direction, operations in (("input", boundary.in_plan), ("output", boundary.out_plan)):
                plans[direction] = [{"src": placement_metadata(operation.src_placements[axis]),
                                     "dst": placement_metadata(operation.dst_placements[axis])}
                                    for operation in operations]
            found.append(plans)
        function = getattr(function, "__wrapped__", None)
    if not found and tp_size == 1:
        return None
    if len(found) != 1:
        raise ValueError("Missing or ambiguous actual compiled TP boundary")
    return found[0]


def _runtime_layout(model: Any, topology: Any) -> dict[str, Any]:
    modules = dict(model.named_modules())
    parent = modules[PARENT]
    original = [cls for cls in type(parent).__mro__ if cls.__name__ == "JTDeepseekV3MLAAttention"]
    if len(original) != 1 or _source_sha(original[0]) != MODEL_SOURCE_SHA256:
        raise ValueError("Actual attention class/source differs from fixed JT MLA")
    if (parent.num_heads != 128 // topology.tp_size or parent.q_lora_rank != 1536
            or parent.kv_lora_rank != 512 or parent.qk_head_dim != 192 or parent.v_head_dim != 128):
        raise ValueError("Unknown MLA local-head/latent contract")
    shard = {"kind": "shard", "dim": 1}
    replicate = {"kind": "replicate"}
    parent_plan = _boundary(parent, topology.tp_size)
    expected_input = {"src": shard, "dst": shard}
    expected_output = {"src": {"kind": "partial", "reduce_op": "sum"}, "dst": shard}
    plans = {PARENT: parent_plan}
    if topology.tp_size == 2:
        if parent_plan != {"input": [expected_input], "output": [expected_output]}:
            raise ValueError("Actual parent TP/SP boundary differs from the proven contract")
        for suffix in ("q_a_layernorm", "kv_a_layernorm", "key_rope_gather"):
            name = f"{PARENT}.{suffix}"
            plan = _boundary(modules[name], topology.tp_size)
            if plan != {"input": [expected_input], "output": [{"src": shard, "dst": replicate}]}:
                raise ValueError("Actual latent/rope gather boundary is unproven")
            plans[name] = plan
    for suffix in projection_shapes():
        module = modules[f"{PARENT}.{suffix}"]
        if type(module) is not torch.nn.Linear:
            raise ValueError("Projection is no longer an ordinary Linear")
        function = module.forward
        while getattr(function, "__wrapped__", None) is not None:
            code = function.__func__.__code__ if inspect.ismethod(function) else function.__code__
            if code.co_name != "local_param_forward" or _source_sha(function) != REWRITER_SHA256:
                raise ValueError("Unexpected inner projection wrapper/collective")
            function = function.__wrapped__
        if getattr(function, "__func__", function) is not torch.nn.Linear.forward:
            raise ValueError("Projection forward is not the fixed ordinary Linear implementation")
    return {"verified": True, "source_commit": SOURCE_COMMIT, "model_source_sha256": MODEL_SOURCE_SHA256,
            "rewriter_sha256": REWRITER_SHA256, "compiled_tp_plans": plans,
            "kv_b_physical_contract": "[1,1,4096,feature]", "o_proj_output": "K-partition contribution"}


def _bindings(model: Any) -> dict[str, MaterializationBinding]:
    modules = dict(model.named_modules())
    targets = {f"{PARENT}.{name}": modules[f"{PARENT}.{name}"] for name in projection_shapes()}
    found, visited = {}, set()
    for owner in model.modules():
        scheduler = getattr(owner, "hsdp_scheduler", None)
        state = getattr(scheduler, "hsdp_state", None)
        if state is None or id(state) in visited:
            continue
        visited.add(id(state))
        for parameter in state.hsdp_params:
            info = parameter._module_info
            for name, module in targets.items():
                if info.module is module and info.param_name == "weight":
                    if name in found:
                        raise ValueError("Projection weight has duplicate HSDP owners")
                    found[name] = MaterializationBinding(parameter, state)
    if set(found) != set(targets):
        raise ValueError("Each projection needs its actual HSDP managed-parameter owner")
    return found


class Attention0Callback(Callback):
    """Keep the native callback order and normal three-step training lifecycle."""

    def __init__(self, base: Any, trace: Attention0ForwardTrace) -> None:
        super().__init__(base)
        self.trace = trace

    def on_train_begin(self, state: Any, **_kwargs: Any) -> None:
        """Require a fresh initial step."""
        if state.global_step != 0:
            raise ValueError("Attention observation requires step zero")

    def on_step_begin(self, state: Any, **_kwargs: Any) -> None:
        """Arm only step one."""
        self.trace.start_step(state.global_step)

    def on_micro_step_begin(self, state: Any, micro_batch: Any, **_kwargs: Any) -> None:
        """Bind actual batch identity at consumption time."""
        self.trace.begin_microbatch(micro_batch)

    def on_step_end(self, state: Any, **_kwargs: Any) -> None:
        """Complete coverage and remove hooks after the normal update."""
        self.trace.finish_step(state.global_step)

    def on_train_end(self, state: Any, **_kwargs: Any) -> None:
        """Finalize exactly three normal optimizer steps."""
        self.trace.finish_run(state.global_step)


def build_observed_trainer(*args: Any, **kwargs: Any) -> Any:
    """Bind current native HSDP parameters without unsharding or gathering them."""
    global ACTIVE_TRACE
    if ACTIVE_TRACE is not None:
        raise ValueError("One capture process must build exactly one observed trainer")
    trainer = ORIGINAL_TRAINER(*args, **kwargs)
    base = trainer.base
    mesh, config, model_config = base.mesh, base.config, base.model.config
    if (base.train_iters != 3 or base.num_micro_batches != mesh.tp_size
            or config.activation_checkpoint.mode != "off" or config.activation_checkpoint.swap_inputs
            or model_config.hidden_size != 5120 or model_config.num_hidden_layers != 2
            or model_config.num_nextn_predict_layers != 1):
        raise ValueError("Expected the unchanged three-step native validation contract")
    topology = Topology(rank=cap.RANK, dp_rank=mesh.dp_rank, tp_rank=mesh.tp_rank,
                        tp_size=mesh.tp_size, dp_size=mesh.dp_size, sequence_parallel=mesh.sequence_parallel,
                        hidden_size=model_config.hidden_size, cp_size=mesh.cp_size, pp_size=mesh.pp_size,
                        ep_size=mesh.ep_size, micro_batch_size=config.training.micro_batch_size)
    trace = Attention0ForwardTrace(base.model, cap.OUTPUT, topology, _bindings(base.model),
                                   _runtime_layout(base.model, topology))
    ACTIVE_TRACE = trace
    base.get_batch = wrap_get_batch(base.get_batch, trace)
    # The pinned trainer exposes the list without a public callback-registration API.
    base._callbacks.append(Attention0Callback(base, trace))  # pylint: disable=protected-access
    return trainer


def main() -> None:
    """Keep runtime evidence and reject additional mathematical instrumentation."""
    if (os.environ.get("JT_RECOMPUTE_TARGETS") or os.environ.get("JT_MODULE_PROFILE") == "1"
            or os.environ.get("JT_EXPERT_PROBES") == "1"):
        raise ValueError("Other recompute/profile/expert probes are outside this passive case")
    cap.EvidenceCallback = RuntimeEvidenceCallback
    cap.TextTrainer = build_observed_trainer
    try:
        cap.main()
    finally:
        cleanup_after_execution(ACTIVE_TRACE)


if __name__ == "__main__":
    main()
