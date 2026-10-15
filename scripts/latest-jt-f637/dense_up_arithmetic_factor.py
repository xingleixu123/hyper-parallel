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

"""Isolate the first dense MLP up-derivative arithmetic without changing its gate."""
import json
from types import MethodType

import torch
from torch.nn import functional

import capture_hp_native
from silu_gate_factor import SiluGateGradient


class DenseUpGradient(SiluGateGradient):
    """Keep the measured gate derivative and compute the up derivative in FP32."""

    @staticmethod
    def backward(ctx, gradient):
        gate_gradient, _, _ = SiluGateGradient.backward(ctx, gradient)
        pair, = ctx.saved_tensors
        gate, _ = pair.chunk(2, dim=-1)
        up_gradient = (gradient.float() * functional.silu(gate.float())).to(gate.dtype)
        return gate_gradient, up_gradient, None


def install() -> None:
    """Change only the dense MLP after the established composition is installed."""
    original = capture_hp_native.Observer.on_train_begin

    def begin(observer, state, **kwargs):
        original(observer, state, **kwargs)
        name = 'model.layers.0.mlp'
        module = dict(observer.trainer.model.named_modules())[name]

        def forward(self, inputs):
            weight = torch.stack((self.gate_proj.weight, self.up_proj.weight), dim=1).flatten(0, 1)
            projected = functional.linear(inputs, weight)
            activated = DenseUpGradient.apply(projected[..., 0::2], projected[..., 1::2], False)
            return self.down_proj(activated)

        module.forward = MethodType(forward, module)
        report = {'modules': [name], 'teacher_values_used': False, 'forward_unchanged': True,
                  'gate_derivative_unchanged': True,
                  'up_formula': '(dy.float() * silu(gate.float())).to(gate.dtype)',
                  'classification': 'Archive arithmetic compatibility experiment; correctness evaluated separately'}
        (observer.output / 'DENSE_UP_ARITHMETIC.json').write_text(json.dumps(report, indent=2))
        print('[P12-DENSE-UP] FP32 SiLU then upstream product', name, flush=True)

    capture_hp_native.Observer.on_train_begin = begin
