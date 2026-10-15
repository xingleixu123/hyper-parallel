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

"""Observe one actual activation-output row and its derivative without changing arithmetic."""
import hashlib
import json
from pathlib import Path
from types import MethodType

import numpy as np
import torch
from capture_hp_native import Observer
import muon_dense_adjacent_factor

CONTEXT={'step':201,'directory':None,'records':[]}


def save(name, kind, tensor):
    """Store exact205 observations; never use a saved tensor in training."""
    if CONTEXT['step'] != 205:
        return
    array = tensor.detach().reshape(-1,tensor.shape[-1])[1605].float().cpu().numpy()
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
    original_finalize=muon_dense_adjacent_factor.install
    def attach(self,state,**kwargs):
        directory=Path('/state/jt-recipe-20261014')/self.output.name/'activation_row'
        directory.mkdir(parents=True,exist_ok=False)
        CONTEXT.update(step=int(state.global_step)+1,directory=directory)
        module=dict(self.trainer.model.named_modules())['model.layers.1.mlp.shared_experts.down_proj']
        original=module.forward
        def forward(self,values):
            del self
            values=InputBoundary.apply(values,'shared_fc2');save('shared_fc2','x',values)
            return original(values)
        module.forward=MethodType(forward,module)
        (self.output/'SHARED_ACTIVATION_OBSERVER.json').write_text(json.dumps({'token_row':1605,'retained_steps':[205],
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2))
    def end(self,state,**kwargs):
        original_end(self,state,**kwargs)
        if int(state.global_step)==205:
            if len(CONTEXT['records'])!=2:raise ValueError('Expected two row observations')
            (self.output/'SHARED_ACTIVATION_ROW.json').write_text(json.dumps({'step':205,'records':CONTEXT['records']},indent=2))
        CONTEXT['step']=int(state.global_step)+1
    def finalize():
        original_finalize();complete_begin=Observer.on_train_begin
        def begin(self,state,**kwargs):
            complete_begin(self,state,**kwargs);attach(self,state,**kwargs)
        Observer.on_train_begin=begin
    muon_dense_adjacent_factor.install=finalize;Observer.on_step_end=end
