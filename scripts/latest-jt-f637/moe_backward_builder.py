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

"""Single-factor backward layout and branch-accumulation probes on the aligned forward baseline."""
from typing import Any

import torch
from torch.nn import functional
import torch_npu

from hyper_parallel.components.functional.npu_grouped_swiglu import _get_npu_grouped_matmul
from hyper_parallel.models.jt_deepseek_v3 import modeling_jt_deepseek_v3 as jt
from forward_factor_probe import configure_factor as configure_qkv
from mlp_precision_probe import FusedActivationMLP
import moe_precision_builder
from moe_precision_probe import FP32BothMoE


class InterleavedProjectionMLP(FusedActivationMLP):
    """Use one interleaved gate/up projection while preserving separate parameter ownership."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.gate_proj.bias is not None or self.up_proj.bias is not None:
            raise ValueError('This diagnostic requires bias-free projections')
        weights = torch.stack((self.gate_proj.weight, self.up_proj.weight), dim=1).flatten(0, 1)
        interleaved = functional.linear(inputs, weights)
        blocked = interleaved.unflatten(-1, (-1, 2)).transpose(-1, -2).flatten(-2)
        return self.down_proj(torch_npu.npu_swiglu(blocked, dim=-1))


class InterleavedExperts(jt.JTDeepseekV3Experts):
    """Test interleaved grouped-GEMM channels while preserving stored blocked gate/up weights."""

    def forward_expert_major(self, inputs: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        grouped = _get_npu_grouped_matmul()
        group_list = torch.cumsum(counts.to(torch.int64), dim=0)
        weights = self.gate_up_proj.unflatten(1, (2, -1)).transpose(1, 2).flatten(1, 2)
        projected = grouped.apply(inputs, weights.transpose(1, 2).contiguous(), group_list)
        blocked = projected.unflatten(-1, (-1, 2)).transpose(-1, -2).flatten(-2)
        activated = torch_npu.npu_swiglu(blocked, dim=-1)
        return grouped.apply(activated, self.down_proj.transpose(1, 2), group_list)


class RoundedUpGradient(torch.autograd.Function):
    """Isolate the observed MTP up-gradient intermediate rounding without changing forward values."""

    @staticmethod
    def forward(ctx, gate, up):
        pair = torch.cat((gate, up), dim=-1)
        ctx.save_for_backward(pair)
        return torch_npu.npu_swiglu(pair, dim=-1)

    @staticmethod
    def backward(ctx, gradient):
        pair, = ctx.saved_tensors
        fused = torch.ops.npu.npu_swiglu_backward(gradient, pair, -1)
        gate, _ = pair.chunk(2, dim=-1)
        gate_gradient, _ = fused.chunk(2, dim=-1)
        return gate_gradient, gradient * functional.silu(gate)


class RoundedUpMLP(FusedActivationMLP):
    """Use the single measured archive compatibility boundary in the MTP shared MLP."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_proj(inputs), self.up_proj(inputs)
        return self.down_proj(RoundedUpGradient.apply(gate, up))


class InterleavedRoundedUpMLP(FusedActivationMLP):
    """Compose the separately measured combined FC1 and MTP activation-gradient boundaries."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        weights = torch.stack((self.gate_proj.weight, self.up_proj.weight), dim=1).flatten(0, 1)
        projected = functional.linear(inputs, weights)
        return self.down_proj(RoundedUpGradient.apply(projected[..., 0::2], projected[..., 1::2]))


class WeightedProductGradient(torch.autograd.Function):
    """Keep the weighted forward precise while isolating either measured backward cast."""

    @staticmethod
    def forward(ctx, outputs, weights, round_outputs, round_weights):
        ctx.save_for_backward(outputs, weights)
        ctx.round_outputs, ctx.round_weights = round_outputs, round_weights
        return outputs.float() * weights.float()

    @staticmethod
    def backward(ctx, gradient):
        outputs, weights = ctx.saved_tensors
        if ctx.round_outputs:
            output_gradient = gradient.to(outputs.dtype) * weights.to(outputs.dtype)
        else:
            output_gradient = gradient * weights.float()
        if ctx.round_weights:
            weight_gradient = (gradient.to(outputs.dtype) * outputs).sum(dim=-1, keepdim=True).to(weights.dtype)
        else:
            weight_gradient = (gradient * outputs.float()).sum(dim=-1, keepdim=True).to(weights.dtype)
        return output_gradient, weight_gradient, None, None


class RouterRoutedInputPair(torch.autograd.Function):
    """Join FP32 router and BF16 expert derivatives before the shared-branch accumulation."""

    @staticmethod
    def forward(ctx, hidden):
        ctx.input_dtype = hidden.dtype
        return hidden.view_as(hidden), hidden.float()

    @staticmethod
    def backward(ctx, expert_gradient, router_gradient):
        return (expert_gradient.float() + router_gradient).to(ctx.input_dtype)


class RoundedWeightedMoE(FP32BothMoE):
    """Preserve routing and forward accumulation while changing one backward rounding factor."""

    def local_routed_forward(self, hidden: torch.Tensor) -> torch.Tensor:
        router_hidden = hidden
        if getattr(self, 'join_router_input', False):
            hidden, router_hidden = RouterRoutedInputPair.apply(hidden)
        indices, probabilities = self.route(router_hidden)
        flat = hidden.reshape(-1, hidden.shape[-1])
        expert_ids = indices.reshape(-1)
        order = expert_ids.argsort()
        sources = torch.arange(flat.shape[0], device=flat.device).repeat_interleave(indices.shape[1])[order]
        counts = expert_ids.new_zeros(self.experts.num_experts).scatter_add_(
            0, expert_ids, torch.ones_like(expert_ids))
        outputs = self.experts.forward_expert_major(flat[sources], counts)
        weights = probabilities.reshape(-1)[order].unsqueeze(-1)
        weighted = WeightedProductGradient.apply(outputs, weights,
                                                self.backward_factor in {'expert-grad-round', 'mtp-expert-grad-round',
                                                                         'main-expert-grad-round', 'archive-mtp-backward'},
                                                self.backward_factor in {'prob-grad-round', 'archive-mtp-backward',
                                                                         'archive-main-backward'})
        combined = torch.zeros_like(flat, dtype=torch.float32).index_add_(0, sources, weighted)
        return combined.to(hidden.dtype).reshape(hidden.shape)


class FP32BranchInputMoE(FP32BothMoE):
    """Join routed, shared and router input gradients in FP32 before returning to BF16."""

    def local_routed_forward(self, hidden: torch.Tensor) -> torch.Tensor:
        indices, probabilities = self.route(hidden)
        flat = hidden.reshape(-1, hidden.shape[-1])
        expert_ids = indices.reshape(-1)
        order = expert_ids.argsort()
        sources = torch.arange(flat.shape[0], device=flat.device).repeat_interleave(indices.shape[1])[order]
        counts = expert_ids.new_zeros(self.experts.num_experts).scatter_add_(
            0, expert_ids, torch.ones_like(expert_ids))
        output_dtype = self.experts.gate_up_proj.dtype
        outputs = self.experts.forward_expert_major(flat[sources].to(output_dtype), counts)
        weights = probabilities.reshape(-1)[order].unsqueeze(-1)
        combined = torch.zeros_like(flat, dtype=torch.float32).index_add_(
            0, sources, outputs.float() * weights.float())
        return combined.to(output_dtype).reshape(hidden.shape)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.float()
        hidden = hidden_states
        if self.padding:
            hidden = torch.cat((hidden.new_zeros(1, self.padding, hidden.shape[-1]), hidden), dim=1)
        routed = self.local_routed_forward(hidden)[:, self.padding:]
        shared = self.shared_experts(hidden_states.to(self.shared_experts.gate_proj.weight.dtype))
        return routed.float() + shared.float()


def build_jt_model(*, backward_factor: str, **kwargs: Any) -> torch.nn.Module:
    """Apply exactly one new factor after the verified norm/MoE/QKV forward settings."""
    choices = {'shared-interleave': 3, 'expert-interleave': 2, 'branch-fp32': 2, 'mtp-up-round': 1,
               'prob-grad-round': 2, 'expert-grad-round': 2, 'weighted-backward-control': 2,
               'mtp-expert-grad-round': 1, 'main-expert-grad-round': 1, 'archive-measured-backward': 3,
               'archive-measured-projection-backward': 5, 'router-input-join': 2,
               'archive-measured-join-backward': 5}
    if backward_factor not in choices:
        raise ValueError('Unknown backward factor: ' + backward_factor)
    original = moe_precision_builder.configure_factor

    def configure(model, factor):
        original(model, factor)
        configure_qkv(model, 'qkv-forward')
        changed = []
        for name, module in model.named_modules():
            if (backward_factor in {'archive-measured-projection-backward', 'archive-measured-join-backward'}
                    and type(module) is FusedActivationMLP):
                module.__class__ = (InterleavedRoundedUpMLP if name.startswith('mtp.')
                                    else InterleavedProjectionMLP)
                changed.append(name)
            elif (backward_factor in {'mtp-up-round', 'archive-measured-backward'}
                    and name == 'mtp.layers.0.transformer_layer.mlp.shared_experts'):
                if type(module) is not FusedActivationMLP:
                    raise ValueError('Expected the already verified fused activation')
                module.__class__ = RoundedUpMLP
                changed.append(name)
            elif backward_factor == 'shared-interleave' and type(module) is FusedActivationMLP:
                module.__class__ = InterleavedProjectionMLP
                changed.append(name)
            elif backward_factor == 'expert-interleave' and type(module) is jt.JTDeepseekV3Experts:
                module.__class__ = InterleavedExperts
                changed.append(name)
            elif backward_factor == 'branch-fp32' and type(module) is FP32BothMoE:
                module.__class__ = FP32BranchInputMoE
                module.ep_compute = module.local_routed_forward
                changed.append(name)
            elif (backward_factor in {'prob-grad-round', 'expert-grad-round', 'weighted-backward-control',
                                      'mtp-expert-grad-round', 'main-expert-grad-round', 'archive-measured-backward',
                                      'archive-measured-projection-backward', 'router-input-join',
                                      'archive-measured-join-backward'}
                  and type(module) is FP32BothMoE):
                if backward_factor == 'mtp-expert-grad-round' and not name.startswith('mtp.'):
                    continue
                if backward_factor == 'main-expert-grad-round' and name.startswith('mtp.'):
                    continue
                module.__class__ = RoundedWeightedMoE
                module.backward_factor = backward_factor
                module.join_router_input = backward_factor in {'router-input-join', 'archive-measured-join-backward'}
                if backward_factor in {'archive-measured-backward', 'archive-measured-projection-backward',
                                       'archive-measured-join-backward'}:
                    module.backward_factor = 'archive-mtp-backward' if name.startswith('mtp.') else 'archive-main-backward'
                module.ep_compute = module.local_routed_forward
                changed.append(name)
        expected = choices[backward_factor]
        if len(changed) != expected:
            raise ValueError(f'Unexpected affected modules: {changed}')
        print('[P12-MOE-BACKWARD-FACTOR]', backward_factor, changed, flush=True)

    moe_precision_builder.configure_factor = configure
    try:
        return moe_precision_builder.build_jt_model(**kwargs)
    finally:
        moe_precision_builder.configure_factor = original
