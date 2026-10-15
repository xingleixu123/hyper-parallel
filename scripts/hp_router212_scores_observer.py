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

"""Observe actual routing normalization and sigmoid inputs/derivatives at212."""
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from capture_hp_native import Observer
import hp_moe212_branches_factor as base
import muon_dense_adjacent_factor
from router_normalization_join_factor import RoutingNormalization
from router_sigmoid_arithmetic_factor import ArithmeticSigmoid

CONTEXT: dict[str, Any] = {'records': []}


def save(name: str, value: Any) -> None:
    """Retain one native boundary without recomputing or replacing it."""
    if base.CONTEXT['step'] != 212:
        return
    array = value.detach().cpu().numpy()
    path = CONTEXT['directory'] / (name + '.npy')
    if path.exists():
        raise ValueError('Unexpected repeated routing boundary')
    np.save(path, array, allow_pickle=False)
    CONTEXT['records'].append({'name': name, 'file': path.name, 'dtype': str(array.dtype),
                               'shape': list(array.shape), 'sha256': hashlib.sha256(array.tobytes()).hexdigest()})


def install() -> None:
    """Wrap existing functions, preserving their exact returned values."""
    norm_forward, norm_backward = RoutingNormalization.forward, RoutingNormalization.backward
    sigmoid_forward, sigmoid_backward = ArithmeticSigmoid.forward, ArithmeticSigmoid.backward
    original_end, original_finalize = Observer.on_step_end, muon_dense_adjacent_factor.install

    def norm_fwd(ctx: Any, scores: Any, indices: Any, scale: Any) -> Any:
        ctx.score_probe_main = base.CONTEXT['active'] == 'main'
        result = norm_forward(ctx, scores, indices, scale)
        if ctx.score_probe_main:
            save('indices', indices)
            save('coefficients', result[0])
            save('auxiliary', result[1])
        return result

    def norm_bwd(ctx: Any, coefficient_gradient: Any, auxiliary_gradient: Any) -> Any:
        if ctx.score_probe_main:
            _, _, selected, selected_sum, score_sum = ctx.saved_tensors
            for name, value in [('selected', selected), ('selected_sum', selected_sum), ('score_sum', score_sum),
                                ('coefficients_dy', coefficient_gradient), ('auxiliary_dy', auxiliary_gradient)]:
                save(name, value)
        result = norm_backward(ctx, coefficient_gradient, auxiliary_gradient)
        if ctx.score_probe_main:
            save('normalization_dx', result[0])
        return result

    def sigmoid_fwd(ctx: Any, values: Any, mode: Any) -> Any:
        ctx.score_probe_main = base.CONTEXT['active'] == 'main'
        result = sigmoid_forward(ctx, values, mode)
        if ctx.score_probe_main:
            save('logits', values)
            save('scores', result)
        return result

    def sigmoid_bwd(ctx: Any, gradient: Any) -> Any:
        result = sigmoid_backward(ctx, gradient)
        if ctx.score_probe_main:
            save('scores_dy', gradient)
            save('logits_dy', result[0])
        return result

    def end(self: Any, state: Any, **kwargs: Any) -> None:
        original_end(self, state, **kwargs)
        if int(state.global_step) == 212:
            if len(CONTEXT['records']) != 13:
                raise ValueError('Expected thirteen routing boundaries')
            (self.output / 'ROUTER_SCORES-212.json').write_text(json.dumps({
                'step': 212, 'records': CONTEXT['records'],
                'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'readonly': True, 'teacher_values_used': False}, indent=2))

    def finalize() -> None:
        original_finalize()
        complete_begin = Observer.on_train_begin

        def begin(self: Any, state: Any, **kwargs: Any) -> None:
            complete_begin(self, state, **kwargs)
            directory = Path('/state/jt-recipe-20261014') / self.output.name / 'scores'
            directory.mkdir(parents=True, exist_ok=False)
            CONTEXT['directory'] = directory

        Observer.on_train_begin = begin

    RoutingNormalization.forward, RoutingNormalization.backward = staticmethod(norm_fwd), staticmethod(norm_bwd)
    ArithmeticSigmoid.forward, ArithmeticSigmoid.backward = staticmethod(sigmoid_fwd), staticmethod(sigmoid_bwd)
    Observer.on_step_end, muon_dense_adjacent_factor.install = end, finalize
