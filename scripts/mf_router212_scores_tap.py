# Copyright 2026 Huawei Technologies Co., Ltd.
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

"""Observe native routing outputs and derivative branches alongside verified MoE boundaries."""
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import mindspore as ms
from mindspore import nn, ops

import mf_moe212_branches_tap as base

CONTEXT: dict[str, Any] = {}
SEEN: set[Path] = set()
KINDS = ['logits', 'scores', 'scores_dy', 'coefficients', 'coefficients_dy',
         'auxiliary', 'auxiliary_dy', 'indices']


class ObservedSigmoid(nn.Cell):
    """Preserve the original primitive and observe its shared derivative."""

    def __init__(self, original: Any, path: str) -> None:
        super().__init__()
        self.original, self.path = original, path
        self.probe_dy = ops.InsertGradientOf(base.gradient_sink(path + '_scores_dy_value'))

    def construct(self, values: Any) -> Any:
        result = self.original(values)
        ops.TensorDump()(self.path + '_scores_value', result.astype(ms.float32))
        return self.probe_dy(result)


class ObservedRouting(base.ObservedRouter):
    """Keep original routing and expose the two normalization branches."""

    def gating(self, inputs: Any) -> Any:
        result = super().gating(inputs)
        ops.TensorDump()(self.score_probe_path + '_logits_value', result.astype(ms.float32))
        return result

    def routing(self, logits: Any) -> Any:
        coefficients, indices, auxiliary = super().routing(logits)
        ops.TensorDump()(self.score_probe_path + '_coefficients_value', coefficients.astype(ms.float32))
        ops.TensorDump()(self.score_probe_path + '_indices_value', indices)
        ops.TensorDump()(self.score_probe_path + '_auxiliary_value', auxiliary.astype(ms.float32))
        return self.probe_coefficients(coefficients), indices, self.probe_auxiliary(auxiliary)


def attach(network: Any, directory: Path) -> None:
    """Add scalar-width routing observations to the existing main-MoE observer."""
    base.attach(network, directory)
    module = dict(network.cells_and_names())['model.decoder.layers.1.mlp.router']
    if type(module) is not base.ObservedRouter:
        raise ValueError('Unexpected original router observer')
    module.__class__ = ObservedRouting
    module.score_probe_path = str(directory / 'router_scores')
    module.gating_activation = ObservedSigmoid(module.gating_activation, module.score_probe_path)
    module.probe_coefficients = ops.InsertGradientOf(base.gradient_sink(module.score_probe_path + '_coefficients_dy_value'))
    module.probe_auxiliary = ops.InsertGradientOf(base.gradient_sink(module.score_probe_path + '_auxiliary_dy_value'))
    CONTEXT['directory'] = directory


def observed_end(self: Any, run_context: Any) -> None:
    """Retain212 only and validate every native dump completes before reading."""
    base.observed_end(self, run_context)
    step = int(run_context.original_args().cur_step_num) + self.state.get('offset', 0)
    records = []
    for kind in KINDS:
        deadline = time.monotonic() + 45
        while True:
            paths = sorted(set(CONTEXT['directory'].glob('router_scores_' + kind + '_value_*.npy')) - SEEN)
            try:
                values = [np.load(path, mmap_mode='r') for path in paths]
                ready = paths and all(path.stat().st_size == value.offset + value.nbytes
                                      for path, value in zip(paths, values))
            except (ValueError, EOFError, OSError):
                ready = False
            if ready:
                break
            if time.monotonic() > deadline:
                raise TimeoutError('Routing boundary: ' + kind)
            time.sleep(.05)
        hashes = {hashlib.sha256(value.tobytes()).hexdigest() for value in values}
        if len(hashes) != 1:
            raise ValueError('Recomputed routing boundary changed')
        for path, value in zip(paths, values):
            records.append({'name': kind, 'file': path.name, 'shape': list(value.shape),
                            'dtype': str(value.dtype), 'sha256': next(iter(hashes)), 'retained': step == 212})
            SEEN.add(path)
        del values, value
        if step != 212:
            for path in paths:
                if path.is_symlink() or path.resolve().parent != CONTEXT['directory'].resolve():
                    raise ValueError('Unsafe routing temporary path')
                path.unlink()
    (self.output / ('ROUTER_SCORES-' + str(step) + '.json')).write_text(json.dumps({
        'step': step, 'records': records, 'readonly': True,
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2))


base.capture_mf_muon.attach_taps = attach
base.capture_mf_muon.ObserveMetrics.on_train_step_end = observed_end
