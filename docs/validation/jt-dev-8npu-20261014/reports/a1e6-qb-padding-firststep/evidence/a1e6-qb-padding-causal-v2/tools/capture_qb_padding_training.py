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

"""Install the active short arm after native TP/FSDP and before any q_b observer."""

from __future__ import annotations

import hashlib
import inspect
import os
from pathlib import Path
from typing import Any

import capture_training as cap
from capture_a1e6_training import RuntimeEvidenceCallback
from hyper_parallel.trainer.callbacks.base import Callback

from native_qb_padding import QBPaddingTrace, Topology, cleanup_after_execution, wrap_get_batch
from qb_padding_contract import MODULE, OWNER, SOURCE_FILES, frozen_support, source_snapshot


OLD_CAPTURE = frozen_support(capture=True)
ORIGINAL_TRAINER = cap.TextTrainer
ACTIVE_TRACE: QBPaddingTrace | None = None


def owner_hook_proof(model: Any, binding: Any) -> tuple[Any, dict[str, Any]]:
    """Require the actual single ancestor FSDP owner and its unchanged ordered hooks."""
    modules = dict(model.named_modules())
    owner, target = modules[OWNER], modules[MODULE]
    scheduler = getattr(owner, "hsdp_scheduler", None)
    if (scheduler is None or binding.state() is not scheduler.hsdp_state
            or tuple(scheduler.modules) != (owner,) or target._forward_pre_hooks or target._forward_hooks):
        raise ValueError("Expected the original single layer0 owner and an observer-free descendant Linear")
    result = {}
    for stage, hooks, expected in (("pre", owner._forward_pre_hooks, "_forward_pre_hook"),
                                   ("post", owner._forward_hooks, "_forward_hook")):
        if len(hooks) != 1:
            raise ValueError("Unknown ancestor FSDP hook count/order")
        hook = next(iter(hooks.values()))
        path = inspect.getsourcefile(hook)
        if (getattr(hook, "__self__", None) is not scheduler or hook.__name__ != expected or path is None
                or hashlib.sha256(Path(path).read_bytes()).hexdigest() != SOURCE_FILES["scheduler"][1]):
            raise ValueError("Unknown ancestor materialization/reshard hook")
        result[stage] = {"name": expected, "id": next(iter(hooks)), "source_sha256": SOURCE_FILES["scheduler"][1]}
    return owner, {"verified": True, "owner": OWNER, "hooks": result,
                   "ordering": "ancestor pre -> original descendant forward+restore -> active hook -> ancestor post",
                   "passive_attention_observer_installed": False}


class QBPaddingCallback(Callback):
    """Keep the unchanged three-step optimizer loop and actual batch-consumption FIFO."""

    def __init__(self, base: Any, trace: QBPaddingTrace) -> None:
        super().__init__(base)
        self.trace = trace

    def on_train_begin(self, state: Any, **_kwargs: Any) -> None:
        """Reject continuation/restoration into this first-step-only arm."""
        if state.global_step != 0:
            raise ValueError("The active arm requires the original initial state")

    def on_step_begin(self, state: Any, **_kwargs: Any) -> None:
        """Arm only the first optimizer step."""
        self.trace.start_step(state.global_step)

    def on_micro_step_begin(self, state: Any, micro_batch: Any, **_kwargs: Any) -> None:
        """Bind the actual forward-consumed batch identity."""
        self.trace.begin_microbatch(micro_batch)

    def on_step_end(self, state: Any, **_kwargs: Any) -> None:
        """Unload the active hooks after normal first-step backward/update."""
        self.trace.finish_step(state.global_step)

    def on_train_end(self, state: Any, **_kwargs: Any) -> None:
        """Record three-step completion without claiming transparency or repair."""
        self.trace.finish_run(state.global_step)


def build_active_trainer(*args: Any, **kwargs: Any) -> Any:
    """Retain native construction and install no other projection observer hooks."""
    global ACTIVE_TRACE
    if ACTIVE_TRACE is not None:
        raise ValueError("One process may build only one active trainer")
    arm = os.environ.get("JT_QB_PADDING_ARM")
    if arm not in ("T", "P"):
        raise ValueError("Set the explicit new-case arm to T or P")
    trainer = ORIGINAL_TRAINER(*args, **kwargs)
    base = trainer.base
    mesh, config, model_config = base.mesh, base.config, base.model.config
    if (base.train_iters != 3 or base.num_micro_batches != 2 or mesh.tp_size != 2 or mesh.dp_size != 4
            or not mesh.sequence_parallel or mesh.cp_size != 1 or mesh.pp_size != 1 or mesh.ep_size != 1
            or config.activation_checkpoint.mode != "off" or config.activation_checkpoint.swap_inputs
            or config.checkpoint.save_ckpt is not False or config.checkpoint.restore_from is not None
            or model_config.hidden_size != 5120 or model_config.num_hidden_layers != 2
            or model_config.num_nextn_predict_layers != 1):
        raise ValueError("Expected the unchanged three-step TP2/CP1/EP1/PP1 configuration")
    for name in SOURCE_FILES:
        source_snapshot(name)
    topology = Topology(rank=cap.RANK, dp_rank=mesh.dp_rank, tp_rank=mesh.tp_rank,
                        tp_size=2, dp_size=4, sequence_parallel=True, hidden_size=5120,
                        cp_size=1, pp_size=1, ep_size=1, micro_batch_size=config.training.micro_batch_size)
    runtime = OLD_CAPTURE._runtime_layout(base.model, topology)  # pylint: disable=protected-access
    binding = OLD_CAPTURE._bindings(base.model)[MODULE]  # pylint: disable=protected-access
    owner, hooks = owner_hook_proof(base.model, binding)
    trace = QBPaddingTrace(base.model, cap.OUTPUT, topology, binding,
                          {"verified": True, "runtime": runtime, "hook_order": hooks}, arm, owner)
    ACTIVE_TRACE = trace
    base.get_batch = wrap_get_batch(base.get_batch, trace)
    base._callbacks.append(QBPaddingCallback(base, trace))  # pylint: disable=protected-access
    return trainer


def main() -> None:
    """Preserve original model errors and unload only this arm's hooks in outer finally."""
    if (os.environ.get("JT_RECOMPUTE_TARGETS") or os.environ.get("JT_MODULE_PROFILE") == "1"
            or os.environ.get("JT_EXPERT_PROBES") == "1"):
        raise ValueError("Other diagnostics are outside this minimal active arm")
    cap.EvidenceCallback = RuntimeEvidenceCallback
    cap.TextTrainer = build_active_trainer
    try:
        cap.main()
    finally:
        cleanup_after_execution(ACTIVE_TRACE)


if __name__ == "__main__":
    main()
