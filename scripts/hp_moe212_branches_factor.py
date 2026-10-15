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

"""Observe native HP MoE212 branch gradients and FP32 router GEMM operands."""
import hashlib
import json
from pathlib import Path
from types import MethodType

import numpy as np
import torch
from capture_hp_native import Observer
import muon_dense_adjacent_factor
from moe_backward_builder import RouterRoutedInputPair
from router_weight_contiguous_factor import RouterLinear

CONTEXT={'step':201,'directory':None,'records':[],'active':None}


def save(name, kind, tensor):
    """Store exact212 observations; never use a saved tensor in training."""
    if CONTEXT['step'] != 212:
        return
    array = tensor.detach().float().cpu().numpy()
    filename = name.replace('.', '_') + '_' + kind + '.npy'
    path = CONTEXT['directory'] / filename
    digest = hashlib.sha256(array.tobytes()).hexdigest()
    if path.exists():
        if not np.array_equal(np.load(path, mmap_mode='r'), array):
            raise ValueError('Recomputed final-norm operand changed: ' + filename)
        return
    np.save(path, array)
    CONTEXT['records'].append({'name': name, 'kind': kind, 'file': filename,
                               'shape': list(array.shape), 'sha256': digest, 'dtype': str(tensor.dtype)})


class InputBoundary(torch.autograd.Function):
    """Separate the norm branch from other uses of its shared hidden-state input."""

    @staticmethod
    def forward(ctx, value, name):
        ctx.name = name
        return value

    @staticmethod
    def backward(ctx, gradient):
        save(ctx.name, 'dx', gradient)
        return gradient, None


def install():
    original_end=Observer.on_step_end
    pair_forward=RouterRoutedInputPair.forward
    pair_backward=RouterRoutedInputPair.backward
    router_forward=RouterLinear.forward
    router_backward=RouterLinear.backward

    def pair_fwd(ctx,hidden):
        ctx.probe_name=CONTEXT['active']
        return pair_forward(ctx,hidden)

    def pair_bwd(ctx,expert_gradient,router_gradient):
        result=pair_backward(ctx,expert_gradient,router_gradient)
        if ctx.probe_name=='main':
            save('pair','routed_dx',expert_gradient)
            save('pair','router_dx',router_gradient)
            save('pair','joined_dx',result)
        return result

    def router_fwd(ctx,values,weight):
        ctx.probe_name=CONTEXT['active']
        return router_forward(ctx,values,weight)

    def router_bwd(ctx,gradient):
        result=router_backward(ctx,gradient)
        if ctx.probe_name=='main':
            values,weight=ctx.saved_tensors
            save('router','weight',weight)
            save('router','dy',gradient)
            save('router','dx',result[0])
        return result

    def begin(self,state,**kwargs):
        directory=Path('/state/jt-recipe-20261014')/self.output.name/'branches'
        directory.mkdir(parents=True,exist_ok=False)
        CONTEXT.update(step=int(state.global_step)+1,directory=directory)
        modules=dict(self.trainer.model.named_modules())
        module=modules['model.layers.1.mlp']
        if not module.join_router_input:
            raise ValueError('Expected measured router/expert gradient pair')
        original=module.forward
        def forward(self,values):
            del self
            values=InputBoundary.apply(values,'main')
            save('main','x',values)
            CONTEXT['active']='main'
            try:result=original(values)
            finally:CONTEXT['active']=None
            def gradient_hook(gradient):
                save('main','dy',gradient)
                return gradient
            save('main','y',result)
            result.register_hook(gradient_hook)
            return result
        module.forward=MethodType(forward,module)
        shared=module.shared_experts
        shared_original=shared.forward
        def shared_forward(self,values):
            del self
            return shared_original(InputBoundary.apply(values,'shared'))
        shared.forward=MethodType(shared_forward,shared)
        (self.output/'MOE_BRANCH_OBSERVER.json').write_text(json.dumps({'padding':module.padding,
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'teacher_values_used':False,'retained_steps':[212]},indent=2))

    def end(self,state,**kwargs):
        original_end(self,state,**kwargs)
        if int(state.global_step)==212:
            if len(CONTEXT['records'])!=11:
                raise ValueError('Expected11branchobservations:'+str(len(CONTEXT['records'])))
            (self.output/'MOE_BRANCHES.json').write_text(json.dumps({'step':212,'records':CONTEXT['records']},indent=2))
        CONTEXT['step']=int(state.global_step)+1

    RouterRoutedInputPair.forward=staticmethod(pair_fwd)
    RouterRoutedInputPair.backward=staticmethod(pair_bwd)
    RouterLinear.forward=staticmethod(router_fwd)
    RouterLinear.backward=staticmethod(router_bwd)
    original_finalize=muon_dense_adjacent_factor.install
    def finalize():
        original_finalize()
        complete_begin=Observer.on_train_begin
        def begin_after_factors(self,state,**kwargs):
            complete_begin(self,state,**kwargs)
            begin(self,state,**kwargs)
        Observer.on_train_begin=begin_after_factors
    muon_dense_adjacent_factor.install=finalize
    Observer.on_step_end=end
