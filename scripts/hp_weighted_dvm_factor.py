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

"""Use a measured independent final-axis reduction tree on live weighted gradients."""
import hashlib
import json
import os
from pathlib import Path
from types import MethodType
from typing import Any

import torch

from capture_hp_native import Observer
import muon_dense_adjacent_factor
from moe_backward_builder import WeightedProductGradient
from muon_q_norm_probe import half_tree

CONTEXT: dict[str, Any] = {'active': None, 'calls': {'main': 0, 'mtp': 0}}


def sum_fp32(product: torch.Tensor) -> torch.Tensor:
    """Reduce80 rows by halves with odd tail, then64 lanes by adjacent FP32 additions."""
    if product.shape[-1] != 5120 or product.dtype != torch.bfloat16:
        raise ValueError('Measured tree supports the actual BF16 hidden width5120 only')
    values = product.float().reshape(-1, 80, 64).transpose(-1, -2)
    lanes = half_tree(values)
    while lanes.shape[-1] > 1:
        lanes = lanes[..., ::2] + lanes[..., 1::2]
    return lanes.reshape(*product.shape[:-1], 1)


def install() -> None:
    """Change one live reduction at every step, without saved values or coordinate conditions."""
    scope = os.environ['P12_WEIGHTED_DVM_SCOPE']
    if scope not in ('main', 'both'):
        raise ValueError('Unknown weighted reduction scope')
    original_forward, original_backward = WeightedProductGradient.forward, WeightedProductGradient.backward
    original_end, original_finalize = Observer.on_step_end, muon_dense_adjacent_factor.install

    def forward(ctx: Any, outputs: Any, weights: Any, round_outputs: bool, round_weights: bool) -> Any:
        ctx.weighted_dvm_role = CONTEXT['active']
        return original_forward(ctx, outputs, weights, round_outputs, round_weights)

    def backward(ctx: Any, gradient: Any) -> Any:
        result = original_backward(ctx, gradient)
        role = ctx.weighted_dvm_role
        if role not in ('main', 'mtp'):
            raise ValueError('Unclassified weighted derivative')
        if scope == 'main' and role != 'main':
            return result
        outputs, weights = ctx.saved_tensors
        if not ctx.round_weights:
            raise ValueError('Required existing probability-gradient rounding is disabled')
        product = gradient.to(outputs.dtype) * outputs
        weight_gradient = sum_fp32(product).to(outputs.dtype).to(weights.dtype)
        CONTEXT['calls'][role] += 1
        return result[0], weight_gradient, result[2], result[3]

    def begin(self: Any, state: Any, **kwargs: Any) -> None:
        CONTEXT['initial_step'] = int(state.global_step)
        modules = dict(self.trainer.model.named_modules())
        for role, name in [('main', 'model.layers.1.mlp'), ('mtp', 'mtp.layers.0.transformer_layer.mlp')]:
            module = modules[name]
            original = module.forward

            def wrapped(self: Any, values: Any, original: Any = original, role: str = role) -> Any:
                del self
                CONTEXT['active'] = role
                try:
                    return original(values)
                finally:
                    CONTEXT['active'] = None

            module.forward = MethodType(wrapped, module)
        report = {'scope': scope, 'teacher_values_used': False, 'coordinate_or_step_special_case': False,
                  'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'helper_sha256': hashlib.sha256((Path(__file__).parent / 'muon_q_norm_probe.py').read_bytes()).hexdigest(),
                  'policy': 'Unchanged BF16 multiply; FP32 80-row odd-tail half tree then64 adjacent lanes; BF16 sum output.',
                  'classification': 'Archive graph reduction compatibility ablation; not a universal correctness fix.'}
        (self.output / 'WEIGHTED_DVM_FACTOR.json').write_text(json.dumps(report, indent=2))

    def end(self: Any, state: Any, **kwargs: Any) -> None:
        original_end(self, state, **kwargs)
        steps = int(state.global_step) - CONTEXT['initial_step']
        expected = {'main': steps, 'mtp': steps if scope == 'both' else 0}
        if CONTEXT['calls'] != expected:
            raise ValueError('Unexpected live weighted-reduction call count')
        with (self.output / 'WEIGHTED_DVM_CALLS.jsonl').open('a') as stream:
            stream.write(json.dumps({'step': int(state.global_step), 'calls': CONTEXT['calls']}) + '\n')

    def finalize() -> None:
        original_finalize()
        complete_begin = Observer.on_train_begin

        def after_factors(self: Any, state: Any, **kwargs: Any) -> None:
            complete_begin(self, state, **kwargs)
            begin(self, state, **kwargs)

        Observer.on_train_begin = after_factors

    WeightedProductGradient.forward, WeightedProductGradient.backward = staticmethod(forward), staticmethod(backward)
    Observer.on_step_end, muon_dense_adjacent_factor.install = end, finalize
