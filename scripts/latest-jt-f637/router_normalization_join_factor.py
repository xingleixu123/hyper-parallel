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

"""Independent single-card routing arithmetic from audited MF O1 branch ordering."""
import json
import os
from types import MethodType

import torch

import capture_hp_native
from hyper_parallel.core.activation_memory.recompute_state import is_recomputing
from moe_backward_builder import RoundedWeightedMoE
from router_sigmoid_arithmetic_factor import ArithmeticSigmoid
import router_sigmoid_arithmetic_factor


class RoutingNormalization(torch.autograd.Function):
    """Keep original normalized weights while composing the measured derivative paths."""

    @staticmethod
    def forward(ctx, scores, indices, scale):
        selected=scores.gather(1,indices)
        selected_sum=selected.sum(-1,keepdim=True)+1e-20
        score_sum=scores.sum(-1,keepdim=True)+1e-20
        ctx.save_for_backward(scores,indices,selected,selected_sum,score_sum)
        ctx.scale=scale
        return selected/selected_sum*scale, scores/score_sum

    @staticmethod
    def backward(ctx, coefficient_gradient, auxiliary_gradient):
        scores,indices,selected,selected_sum,score_sum=ctx.saved_tensors
        scaled=coefficient_gradient*ctx.scale
        top_numerator=scaled/selected_sum
        top_denominator=(-(selected/selected_sum/selected_sum)*scaled).sum(-1,keepdim=True)
        top_gradient=top_numerator+top_denominator
        selected_gradient=torch.zeros_like(scores).scatter_add_(1,indices,top_gradient)
        auxiliary_denominator=(-(scores/score_sum/score_sum)*auxiliary_gradient).sum(-1,keepdim=True)
        result=(selected_gradient+auxiliary_denominator)+auxiliary_gradient/score_sum
        return result,None,None


def route(self, hidden):
    """Keep the single-sequence routing contract and independent tensor arithmetic."""
    config,padding=self.config,self.padding
    if config.n_group!=1 or config.topk_group!=1 or not config.norm_topk_prob or self.auxiliary_loss_group is not None:
        raise ValueError('Diagnostic routing factor requires single-card normalized routing')
    logits=self.gate(hidden[:,padding:])
    self.gate.pop_router_logits()
    values=logits.view(-1,logits.shape[-1]).float()
    scores=ArithmeticSigmoid.apply(values,'gradient-first')
    indices=(scores+self.gate.e_score_correction_bias).topk(config.num_experts_per_tok,dim=-1,sorted=False)[1]
    selected,normalized=RoutingNormalization.apply(scores,indices,float(config.routed_scaling_factor))
    flat_indices=indices.reshape(-1)
    counts=flat_indices.new_zeros(config.n_routed_experts,dtype=torch.float32).scatter_add_(
        0,flat_indices,flat_indices.new_ones(flat_indices.shape,dtype=torch.float32))
    load=counts/flat_indices.numel()
    auxiliary_loss=(normalized.mean(0)*load).sum()*config.n_routed_experts*config.moe_aux_loss_coeff
    if not is_recomputing():
        self.auxiliary_loss=auxiliary_loss
    self.tokens_per_expert=counts if self.tokens_per_expert is None else self.tokens_per_expert+counts
    if padding:
        pad_ids=torch.arange(padding*config.num_experts_per_tok,device=indices.device)
        pad_ids=pad_ids.reshape(padding,config.num_experts_per_tok)%padding
        indices=torch.cat((pad_ids,indices))
        selected=torch.cat((selected.new_zeros(padding,selected.shape[-1]),selected))
    return indices,selected


def configure(module):
    """Use this independent arithmetic only on the explicit replay module."""
    module.route=MethodType(route,module)


def install():
    """Install after model restore and record the compatibility scope."""
    mode=os.environ.get("P12_ROUTER_MODE","joint-normalization")
    if mode not in ["shared-native","shared-gradientfirst","joint-normalization"]:
        raise ValueError("Unsupported routing factor")
    original=capture_hp_native.Observer.on_train_begin
    def begin(observer,state,**kwargs):
        original(observer,state,**kwargs)
        changed=[]
        for name,module in observer.trainer.model.named_modules():
            if isinstance(module,RoundedWeightedMoE):
                if mode=="joint-normalization":
                    configure(module)
                else:
                    router_sigmoid_arithmetic_factor.configure(module,"native" if mode=="shared-native" else "gradient-first")
                changed.append(name)
        if len(changed)!=2:
            raise ValueError('Expected two routed modules')
        (observer.output/'ROUTER_ARITHMETIC.json').write_text(json.dumps({
            'modules':changed,'mode':mode,'reference_values_used':False,'scope':'single-card archive O1 arithmetic compatibility',
            'sigmoid':'(dy*s)*(1-s)','join':'(selected_gradient+aux_denominator)+aux_numerator'},indent=2))
    capture_hp_native.Observer.on_train_begin=begin
