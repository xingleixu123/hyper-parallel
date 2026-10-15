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

"""Observe the actual native shared-expert FC1 GEMM without replacing backward arithmetic."""
import hashlib
import json
from pathlib import Path
from types import MethodType

import numpy as np
import torch
from torch.nn import functional
from capture_hp_native import Observer
import muon_dense_adjacent_factor

CONTEXT={'step':201,'directory':None,'records':[],'active':False,'call':0}


def save(name, kind, tensor):
    """Store exact205 observations; never use a saved tensor in training."""
    if CONTEXT['step'] != 205:
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
    original_linear=functional.linear
    original_finalize=muon_dense_adjacent_factor.install

    def linear(values,weight,bias=None):
        if not CONTEXT['active']:
            return original_linear(values,weight,bias)
        CONTEXT['call']+=1
        if CONTEXT['call']!=1:
            return original_linear(values,weight,bias)
        if bias is not None:raise ValueError('Expected bias-free FC1')
        values=InputBoundary.apply(values,'shared_fc1')
        result=original_linear(values,weight,bias)
        save('shared_fc1','x',values)
        save('shared_fc1','weight',weight)
        save('shared_fc1','y',result)
        def gradient_hook(gradient):
            save('shared_fc1','dy',gradient)
            return gradient
        result.register_hook(gradient_hook)
        return result

    def attach(self,state,**kwargs):
        directory=Path('/state/jt-recipe-20261014')/self.output.name/'shared_fc1'
        directory.mkdir(parents=True,exist_ok=False)
        CONTEXT.update(step=int(state.global_step)+1,directory=directory)
        module=dict(self.trainer.model.named_modules())['model.layers.1.mlp.shared_experts']
        original=module.forward
        def forward(self,values):
            del self
            CONTEXT.update(active=True,call=0)
            try:result=original(values)
            finally:CONTEXT['active']=False
            if CONTEXT['call']!=2:raise ValueError('Expected FC1 and down-projection native linear calls')
            return result
        module.forward=MethodType(forward,module)
        (self.output/'SHARED_FC1_OBSERVER.json').write_text(json.dumps({'retained_steps':[205],
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'native_linear_and_autograd_unchanged':True},indent=2))

    def end(self,state,**kwargs):
        original_end(self,state,**kwargs)
        if int(state.global_step)==205:
            if len(CONTEXT['records'])!=5:raise ValueError('Expected five FC1 observations')
            (self.output/'SHARED_FC1_BOUNDARIES.json').write_text(json.dumps({'step':205,'records':CONTEXT['records']},indent=2))
        CONTEXT['step']=int(state.global_step)+1

    def finalize():
        original_finalize()
        complete_begin=Observer.on_train_begin
        def begin(self,state,**kwargs):
            complete_begin(self,state,**kwargs)
            attach(self,state,**kwargs)
        Observer.on_train_begin=begin

    functional.linear=linear
    muon_dense_adjacent_factor.install=finalize
    Observer.on_step_end=end
