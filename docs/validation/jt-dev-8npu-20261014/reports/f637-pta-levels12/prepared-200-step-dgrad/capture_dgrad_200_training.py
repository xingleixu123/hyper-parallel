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

"""Matched 200-step FSDP8/CP2 diagnostic; not a production performance benchmark."""

import json
from typing import Any

import capture_training
from linear_dgrad_long_observer import LongLinearDgrad

from hyper_parallel.trainer.callbacks.base import Callback


NATIVE_TRAINER = capture_training.TextTrainer
OBSERVER = None


class DerivativeCoverage(Callback):
    """Persist actual FP32 input-gradient coverage without tensor dumps."""

    def on_step_end(self, state: object, **kwargs: object) -> None:
        """Require a real derivative branch and write one bounded row per optimizer step."""
        result = OBSERVER.finish_step()
        if result['counts'].get('fp32_dgrad', 0) <= 0:
            raise RuntimeError("No FP32 Linear input gradient was computed in this step")
        result.update(step=state.global_step, rank=capture_training.RANK)
        payload = json.dumps(result, allow_nan=False) + '\n'
        if len(payload.encode()) > 64 * 1024:
            raise RuntimeError("Derivative coverage exceeded the per-step metadata limit")
        path = capture_training.OUTPUT / f'linear-dgrad-step-coverage-rank{capture_training.RANK}.jsonl'
        with path.open('a', encoding='utf-8') as handle:
            handle.write(payload)


def build_trainer(*args: Any, **kwargs: Any) -> Any:
    """Install the diagnostic after normal model construction and before training."""
    global OBSERVER  # pylint: disable=global-statement
    trainer = NATIVE_TRAINER(*args, **kwargs)
    base = trainer.base
    if (base.train_iters != 200 or base.state.global_step != 0 or base.mesh.tp_size != 1
            or base.mesh.ep_size != 1 or base.mesh.cp_size not in (1, 2) or base.mesh.pp_size != 1):
        raise ValueError("This diagnostic requires a fresh 200-step TP1/EP1/PP1 FSDP8 or CP2 run")
    if base.mesh.dp_size * base.mesh.cp_size != 8:
        raise ValueError("The diagnostic must retain the matched eight-device topology")
    OBSERVER = LongLinearDgrad()
    base._callbacks.append(DerivativeCoverage(base))  # pylint: disable=protected-access
    capture_training.write_json(f'linear-dgrad-long-experiment-rank{capture_training.RANK}.json', {
        'numeric_recipe_changed': True, 'production_source_unchanged': True,
        'steps': 200, 'cp_size': base.mesh.cp_size,
        'forward': 'native BF16', 'dgrad': 'FP32 Linear then BF16 cast',
        'wgrad': 'native BF16 autograd replay; expert FP32 GMM remains configured separately',
        'comparison': 'only against the FSDP8 reference with this same changed numerical recipe',
        'scope': 'diagnostic only; the native wgrad replay adds overhead, so no production speedup claim',
    })
    return trainer


def main() -> None:
    """Run the ordinary evidence collector and remove only this diagnostic's patch."""
    capture_training.TextTrainer = build_trainer
    try:
        capture_training.main()
    finally:
        if OBSERVER is not None:
            OBSERVER.close()


if __name__ == '__main__':
    main()
