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

"""Retain the actual expert-weight product operands and native derivative at212."""
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from capture_hp_native import Observer
import expert_token_order_factor
import hp_moe212_branches_factor as base
import muon_dense_adjacent_factor
from moe_backward_builder import WeightedProductGradient

CONTEXT: dict[str, Any] = {'records': []}


def save(name: str, value: Any) -> None:
    """Observe one complete tensor without changing the returned training value."""
    if base.CONTEXT['step'] != 212:
        return
    array = value.detach().float().cpu().numpy() if value.is_floating_point() else value.detach().cpu().numpy()
    path = CONTEXT['directory'] / (name + '.npy')
    if path.exists():
        raise ValueError('Repeated weighted boundary: ' + name)
    np.save(path, array, allow_pickle=False)
    CONTEXT['records'].append({'name': name, 'file': path.name, 'shape': list(array.shape),
                               'dtype': str(array.dtype), 'native_dtype': str(value.dtype),
                               'sha256': hashlib.sha256(array.tobytes()).hexdigest()})


def install() -> None:
    """Wrap the frozen weighted operation and preserve its original outputs."""
    original_forward, original_backward = WeightedProductGradient.forward, WeightedProductGradient.backward
    original_order, original_end = expert_token_order_factor.expert_order, Observer.on_step_end
    original_finalize = muon_dense_adjacent_factor.install

    def forward(ctx: Any, outputs: Any, weights: Any, round_outputs: bool, round_weights: bool) -> Any:
        ctx.weighted_probe_main = base.CONTEXT['active'] == 'main'
        return original_forward(ctx, outputs, weights, round_outputs, round_weights)

    def backward(ctx: Any, gradient: Any) -> Any:
        result = original_backward(ctx, gradient)
        if ctx.weighted_probe_main:
            outputs, weights = ctx.saved_tensors
            for name, value in [('outputs', outputs), ('weights', weights), ('dy', gradient),
                                ('weights_dy', result[1])]:
                save(name, value)
            if base.CONTEXT['step'] == 212:
                CONTEXT['round_outputs'], CONTEXT['round_weights'] = ctx.round_outputs, ctx.round_weights
        return result

    def order(indices: Any) -> Any:
        result = original_order(indices)
        if base.CONTEXT['active'] == 'main':
            save('order', result)
        return result

    def end(self: Any, state: Any, **kwargs: Any) -> None:
        original_end(self, state, **kwargs)
        if int(state.global_step) == 212:
            if len(CONTEXT['records']) != 5:
                raise ValueError('Expected five weighted-product observations')
            report = {key: value for key, value in CONTEXT.items() if key != 'directory'}
            report.update(step=212, readonly=True, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
            (self.output / 'WEIGHTED_PRODUCT-212.json').write_text(json.dumps(report, indent=2))

    def finalize() -> None:
        original_finalize()
        complete_begin = Observer.on_train_begin

        def begin(self: Any, state: Any, **kwargs: Any) -> None:
            complete_begin(self, state, **kwargs)
            directory = Path('/state/jt-recipe-20261014') / self.output.name / 'weighted'
            directory.mkdir(parents=True, exist_ok=False)
            CONTEXT['directory'] = directory

        Observer.on_train_begin = begin

    WeightedProductGradient.forward, WeightedProductGradient.backward = staticmethod(forward), staticmethod(backward)
    expert_token_order_factor.expert_order, Observer.on_step_end = order, end
    muon_dense_adjacent_factor.install = finalize
