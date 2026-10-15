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

"""Match the archive's expert-major token ordering using only live routing indices."""
import json
from types import MethodType

import torch

import capture_hp_native
from moe_backward_builder import RoundedWeightedMoE, RouterRoutedInputPair, WeightedProductGradient


def expert_order(indices: torch.Tensor) -> torch.Tensor:
    """Map the archive's two float sorts of K-major indices back to N-major ordinals."""
    if indices.ndim != 2 or indices.numel() >= 2 ** 24:
        raise ValueError('Diagnostic archive ordering requires exact FP32 integer expert indices')
    token_count, top_k = indices.shape
    ids = indices.transpose(0, 1).reshape(1, -1).float()
    sorted_ids, first = torch.sort(ids, dim=1)
    second = torch.sort(sorted_ids, dim=1)[1]
    combined = first.gather(1, second).reshape(-1)
    return (combined % token_count) * top_k + combined // token_count


def local_routed_forward(self, hidden: torch.Tensor) -> torch.Tensor:
    """Keep all forward/backward operators and replace only the token permutation."""
    router_hidden = hidden
    if getattr(self, 'join_router_input', False):
        hidden, router_hidden = RouterRoutedInputPair.apply(hidden)
    indices, probabilities = self.route(router_hidden)
    flat = hidden.reshape(-1, hidden.shape[-1])
    expert_ids = indices.reshape(-1)
    order = expert_order(indices)
    sources = torch.arange(flat.shape[0], device=flat.device).repeat_interleave(indices.shape[1])[order]
    counts = expert_ids.new_zeros(self.experts.num_experts).scatter_add_(
        0, expert_ids, torch.ones_like(expert_ids))
    outputs = self.experts.forward_expert_major(flat[sources], counts)
    weights = probabilities.reshape(-1)[order].unsqueeze(-1)
    weighted = WeightedProductGradient.apply(
        outputs, weights,
        self.backward_factor in {'expert-grad-round', 'mtp-expert-grad-round',
                                 'main-expert-grad-round', 'archive-mtp-backward'},
        self.backward_factor in {'prob-grad-round', 'archive-mtp-backward', 'archive-main-backward'})
    combined = torch.zeros_like(flat, dtype=torch.float32).index_add_(0, sources, weighted)
    return combined.to(hidden.dtype).reshape(hidden.shape)


def configure(module: RoundedWeightedMoE) -> None:
    """Bind the replacement to both call paths used by the single-card module."""
    module.local_routed_forward = MethodType(local_routed_forward, module)
    module.ep_compute = module.local_routed_forward


def install() -> None:
    """Install after all previous composition factors and record the affected modules."""
    original = capture_hp_native.Observer.on_train_begin

    def begin(observer, state, **kwargs):
        original(observer, state, **kwargs)
        changed = []
        for name, module in observer.trainer.model.named_modules():
            if isinstance(module, RoundedWeightedMoE):
                configure(module)
                changed.append(name)
        if len(changed) != 2:
            raise ValueError('Expected exactly two single-card routed modules')
        report = {'modules': changed, 'reference_values_used': False,
                  'factor': 'K-major FP32 sorting twice, mapped back to N-major ordinal',
                  'classification': 'Archive token order compatibility; not a mathematical correctness claim'}
        (observer.output / 'EXPERT_TOKEN_ORDER.json').write_text(json.dumps(report, indent=2))
        print('[P12-EXPERT-ORDER]', json.dumps(report), flush=True)

    capture_hp_native.Observer.on_train_begin = begin
