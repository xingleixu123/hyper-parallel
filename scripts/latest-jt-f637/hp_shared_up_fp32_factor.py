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

"""Independently compute only main shared-expert up gradients with FP32 SiLU."""
import hashlib
import json
from pathlib import Path
from types import MethodType

import torch
from torch.nn import functional
import silu_gate_factor
import muon_dense_adjacent_factor
from capture_hp_native import Observer


class Fp32SiluUpGradient(torch.autograd.Function):
    """Keep existing forward/gate derivative; round the up derivative after FP32 SiLU and multiply."""

    @staticmethod
    def forward(ctx, gate, up):
        return silu_gate_factor.SiluGateGradient.forward(ctx, gate, up, False)

    @staticmethod
    def backward(ctx, gradient):
        pair, = ctx.saved_tensors
        gate, up = pair.chunk(2, dim=-1)
        gate32 = gate.float()
        sigmoid = gate32.sigmoid()
        silu = functional.silu(gate32)
        derivative = (sigmoid + silu) - silu * sigmoid
        gate_gradient = ((gradient.float() * up.float()) * derivative).to(gate.dtype)
        up_gradient = (gradient.float() * silu).to(up.dtype)
        return gate_gradient, up_gradient


def install():
    """Apply this one factor after the complete prior arithmetic stack is installed."""
    original_finalize = muon_dense_adjacent_factor.install

    def finalize():
        original_finalize()
        complete_begin = Observer.on_train_begin

        def begin(self, state, **kwargs):
            complete_begin(self, state, **kwargs)
            name = 'model.layers.1.mlp.shared_experts'
            module = dict(self.trainer.model.named_modules())[name]

            def forward(self, inputs):
                weights = torch.stack((self.gate_proj.weight, self.up_proj.weight), dim=1).flatten(0, 1)
                projected = functional.linear(inputs, weights)
                activated = Fp32SiluUpGradient.apply(projected[..., 0::2], projected[..., 1::2])
                return self.down_proj(activated)

            module.forward = MethodType(forward, module)
            report = {'module': name, 'formula': 'BF16(dy.float()*silu(gate.float()))',
                      'teacher_values_used': False, 'changed_steps': 'all resumed steps, no step or coordinate special case',
                      'forward_and_gate_formula_unchanged': True, 'mtp_unchanged': True,
                      'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      'classification': 'Independent single-factor archive-compatibility ablation; no universal correctness claim'}
            (self.output / 'SHARED_UP_FP32_FACTOR.json').write_text(json.dumps(report, indent=2))
            print('[P12-SHARED-UP-FP32]', json.dumps(report), flush=True)

        Observer.on_train_begin = begin

    muon_dense_adjacent_factor.install = finalize
