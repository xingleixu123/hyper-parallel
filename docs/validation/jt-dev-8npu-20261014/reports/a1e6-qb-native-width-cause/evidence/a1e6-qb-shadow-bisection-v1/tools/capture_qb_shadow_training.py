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

"""Attach the new q_b core shadow only after unchanged native FSDP8 construction."""

from __future__ import annotations

import os
from typing import Any

import capture_training as cap
from capture_a1e6_training import RuntimeEvidenceCallback
from hyper_parallel.trainer.callbacks.base import Callback

from native_qb_shadow import QBShadowTrace, Topology, cleanup_after_execution, wrap_get_batch
from qb_shadow_contract import MODULE, frozen_support


OLD_CAPTURE = frozen_support(capture=True)
ORIGINAL_TRAINER = cap.TextTrainer
ACTIVE_TRACE: QBShadowTrace | None = None


class QBShadowCallback(Callback):
    """Preserve normal native callback order and three complete optimizer steps."""

    def __init__(self, base: Any, trace: QBShadowTrace) -> None:
        super().__init__(base)
        self.trace = trace

    def on_train_begin(self, state: Any, **_kwargs: Any) -> None:
        """Reject resumed execution before the first observed forward."""
        if state.global_step != 0:
            raise ValueError("q_b shadow requires the original initial state")

    def on_step_begin(self, state: Any, **_kwargs: Any) -> None:
        """Arm step one exactly once."""
        self.trace.start_step(state.global_step)

    def on_micro_step_begin(self, state: Any, micro_batch: Any, **_kwargs: Any) -> None:
        """Bind the original batch FIFO when the actual forward consumes it."""
        self.trace.begin_microbatch(micro_batch)

    def on_step_end(self, state: Any, **_kwargs: Any) -> None:
        """Unload hooks after completing step one and preserve later updates."""
        self.trace.finish_step(state.global_step)

    def on_train_end(self, state: Any, **_kwargs: Any) -> None:
        """Require all three normal training steps, even if calibration declined inference."""
        self.trace.finish_run(state.global_step)


def build_observed_trainer(*args: Any, **kwargs: Any) -> Any:
    """Verify the frozen source/runtime contract without modifying production forwards."""
    global ACTIVE_TRACE
    if ACTIVE_TRACE is not None:
        raise ValueError("One process may build only one shadow trainer")
    trainer = ORIGINAL_TRAINER(*args, **kwargs)
    base = trainer.base
    mesh, config, model_config = base.mesh, base.config, base.model.config
    if (base.train_iters != 3 or base.num_micro_batches != 1 or mesh.tp_size != 1 or mesh.dp_size != 8
            or mesh.sequence_parallel or config.activation_checkpoint.mode != "off"
            or config.activation_checkpoint.swap_inputs or model_config.hidden_size != 5120
            or model_config.num_hidden_layers != 2 or model_config.num_nextn_predict_layers != 1):
        raise ValueError("Expected the unchanged three-step FSDP8 native contract")
    topology = Topology(rank=cap.RANK, dp_rank=mesh.dp_rank, tp_rank=mesh.tp_rank,
                        tp_size=mesh.tp_size, dp_size=mesh.dp_size, sequence_parallel=mesh.sequence_parallel,
                        hidden_size=model_config.hidden_size, cp_size=mesh.cp_size, pp_size=mesh.pp_size,
                        ep_size=mesh.ep_size, micro_batch_size=config.training.micro_batch_size)
    proof = OLD_CAPTURE._runtime_layout(base.model, topology)  # pylint: disable=protected-access
    binding = OLD_CAPTURE._bindings(base.model)[MODULE]  # pylint: disable=protected-access
    trace = QBShadowTrace(base.model, cap.OUTPUT, topology, binding, proof)
    ACTIVE_TRACE = trace
    base.get_batch = wrap_get_batch(base.get_batch, trace)
    base._callbacks.append(QBShadowCallback(base, trace))  # pylint: disable=protected-access
    return trainer


def main() -> None:
    """Keep runtime evidence and clean hooks outside the dispatcher on every exit."""
    if (os.environ.get("JT_RECOMPUTE_TARGETS") or os.environ.get("JT_MODULE_PROFILE") == "1"
            or os.environ.get("JT_EXPERT_PROBES") == "1"):
        raise ValueError("Other mathematical/profile probes are outside the core shadow case")
    cap.EvidenceCallback = RuntimeEvidenceCallback
    cap.TextTrainer = build_observed_trainer
    try:
        cap.main()
    finally:
        cleanup_after_execution(ACTIVE_TRACE)


if __name__ == "__main__":
    main()
