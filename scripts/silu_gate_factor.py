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

"""Independent gate-derivative arithmetic matching the measured MF O1 SiLUGrad composition."""
import json
import os
from types import MethodType

import torch
from torch.nn import functional
import torch_npu

import capture_hp_native
from moe_backward_builder import InterleavedProjectionMLP, InterleavedRoundedUpMLP


class SiluGateGradient(torch.autograd.Function):
    """Change only the gate derivative; preserve the existing forward and up derivative."""

    @staticmethod
    def forward(ctx, gate, up, rounded_up):
        pair = torch.cat((gate, up), dim=-1)
        ctx.save_for_backward(pair)
        ctx.rounded_up = rounded_up
        return torch_npu.npu_swiglu(pair, dim=-1)

    @staticmethod
    def backward(ctx, gradient):
        pair, = ctx.saved_tensors
        gate, up = pair.chunk(2, dim=-1)
        gate32 = gate.float()
        sigmoid = gate32.sigmoid()
        silu = functional.silu(gate32)
        derivative = (sigmoid + silu) - silu * sigmoid
        gate_gradient = ((gradient.float() * up.float()) * derivative).to(gate.dtype)
        if ctx.rounded_up:
            up_gradient = gradient * functional.silu(gate)
        else:
            fused = torch.ops.npu.npu_swiglu_backward(gradient, pair, -1)
            _, up_gradient = fused.chunk(2, dim=-1)
        return gate_gradient, up_gradient, None


def attach(module: torch.nn.Module, rounded_up: bool) -> None:
    """Keep combined interleaved FC1 arithmetic and separate parameter ownership."""
    if not isinstance(module, (InterleavedProjectionMLP, InterleavedRoundedUpMLP)):
        raise ValueError('Gate-only factor requires the already verified combined FC1')

    def forward(self, inputs):
        weights = torch.stack((self.gate_proj.weight, self.up_proj.weight), dim=1).flatten(0, 1)
        projected = functional.linear(inputs, weights)
        activated = SiluGateGradient.apply(projected[..., 0::2], projected[..., 1::2], rounded_up)
        return self.down_proj(activated)

    module.forward = MethodType(forward, module)


def install() -> None:
    """Record the selected independent arithmetic factor before the restored model executes."""
    mode = os.environ.get('P12_SILU_GATE_SCOPE', 'all')
    scopes = {
        'all': ['model.layers.0.mlp', 'model.layers.1.mlp.shared_experts',
                'mtp.layers.0.transformer_layer.mlp.shared_experts'],
        'mtp': ['mtp.layers.0.transformer_layer.mlp.shared_experts'],
        'shared': ['model.layers.1.mlp.shared_experts', 'mtp.layers.0.transformer_layer.mlp.shared_experts'],
    }
    if mode not in scopes:
        raise ValueError('Unknown gate arithmetic scope')
    original = capture_hp_native.Observer.on_train_begin

    def begin(observer, state, **kwargs):
        original(observer, state, **kwargs)
        modules = dict(observer.trainer.model.named_modules())
        for name in scopes[mode]:
            attach(modules[name], name.startswith('mtp.'))
        report = {'scope': mode, 'modules': scopes[mode], 'teacher_values_used': False,
                  'gate_formula': '(dy.float()*up.float())*((sigmoid(x)+silu(x))-silu(x)*sigmoid(x))',
                  'forward_unchanged': True, 'up_derivative_unchanged': True,
                  'classification': 'Archive graph arithmetic compatibility experiment; no mathematical-bug claim'}
        (observer.output/'SILU_GATE_ARITHMETIC.json').write_text(json.dumps(report, indent=2))
        print('[P12-SILU-GATE] independent arithmetic factor', mode, scopes[mode], flush=True)

    capture_hp_native.Observer.on_train_begin = begin
