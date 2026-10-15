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

"""Isolate one shared sigmoid graph for routed weights and sequence auxiliary loss."""
from types import MethodType

import torch

from hyper_parallel.core.activation_memory.recompute_state import is_recomputing



class ArithmeticSigmoid(torch.autograd.Function):
    """Isolate the derivative composition while preserving native sigmoid values."""

    @staticmethod
    def forward(ctx, values, mode):
        result = values.sigmoid()
        ctx.save_for_backward(result)
        ctx.mode = mode
        return result

    @staticmethod
    def backward(ctx, gradient):
        result, = ctx.saved_tensors
        if ctx.mode == 'sub-square':
            value = (result - result * result) * gradient
        elif ctx.mode == 'factor-first':
            value = (result * (1 - result)) * gradient
        elif ctx.mode == 'gradient-first':
            value = (gradient * result) * (1 - result)
        else:
            raise ValueError('Unknown sigmoid derivative mode')
        return value, None

def shared_sigmoid_route(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep the measured JT routing contract while sharing the sigmoid derivative node."""
    padding, config = self.padding, self.config
    if config.n_group != 1 or config.topk_group != 1:
        raise ValueError('This isolated factor requires one routing group')
    logits = self.gate(hidden[:, padding:])
    self.gate.pop_router_logits()
    values = logits.view(-1, logits.shape[-1]).float()
    scores = values.sigmoid() if self.sigmoid_backward_mode == 'native' else ArithmeticSigmoid.apply(values, self.sigmoid_backward_mode)
    choice = scores + self.gate.e_score_correction_bias
    indices = choice.topk(config.num_experts_per_tok, dim=-1, sorted=False)[1]
    selected = scores.gather(1, indices)
    if config.norm_topk_prob:
        selected = selected / (selected.sum(dim=-1, keepdim=True) + 1e-20)
    selected = selected * float(config.routed_scaling_factor)
    auxiliary_loss = self.routing_statistics(indices, scores)
    if not is_recomputing():
        self.auxiliary_loss = auxiliary_loss
    flat_indices = indices.reshape(-1)
    counts = flat_indices.new_zeros(config.n_routed_experts, dtype=torch.float32).scatter_add_(
        0, flat_indices, flat_indices.new_ones(flat_indices.shape, dtype=torch.float32))
    self.tokens_per_expert = counts if self.tokens_per_expert is None else self.tokens_per_expert + counts
    if padding:
        pad_ids = torch.arange(padding * config.num_experts_per_tok, device=indices.device)
        pad_ids = pad_ids.reshape(padding, config.num_experts_per_tok) % padding
        indices = torch.cat((pad_ids, indices))
        selected = torch.cat((selected.new_zeros(padding, selected.shape[-1]), selected))
    return indices, selected


def configure(module: torch.nn.Module, mode: str) -> None:
    """Change only this module's route implementation for independent fixed-operand replay."""
    module.sigmoid_backward_mode = mode
    module.route = MethodType(shared_sigmoid_route, module)
