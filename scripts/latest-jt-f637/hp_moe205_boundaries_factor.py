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

"""Observe main-layer MoE native forward/backward205 without injecting values."""
import hashlib
import json
from pathlib import Path
from types import MethodType

import numpy as np
import torch
from capture_hp_native import Observer

CONTEXT = {'step':201, 'directory':None, 'records':[]}


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
    original_begin = Observer.on_train_begin
    original_end = Observer.on_step_end

    def begin(self, state, **kwargs):
        original_begin(self, state, **kwargs)
        directory = Path('/state/jt-recipe-20261014')/self.output.name/'moe'
        directory.mkdir(parents=True,exist_ok=False)
        CONTEXT.update(step=int(state.global_step)+1, directory=directory)
        name = 'model.layers.1.mlp'
        module = dict(self.trainer.model.named_modules())[name]
        original = module.forward
        def forward(self, values):
            del self
            values = InputBoundary.apply(values, name)
            save(name,'x',values)
            result = original(values)
            if not isinstance(result,torch.Tensor):
                raise ValueError('Expected native HP MoE tensor output')
            save(name,'y',result)
            def gradient_hook(gradient):
                save(name,'dy',gradient)
                return gradient
            result.register_hook(gradient_hook)
            return result
        module.forward = MethodType(forward,module)
        (self.output/'MOE_OBSERVER.json').write_text(json.dumps({'name':name,'class':str(type(module)),
            'retained_steps':[205], 'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2))

    def end(self, state, **kwargs):
        original_end(self,state,**kwargs)
        if int(state.global_step)==205:
            if len(CONTEXT['records'])!=4:
                raise ValueError('Expected four MoE observations')
            (self.output/'MOE_BOUNDARIES.json').write_text(json.dumps({'step':205,'records':CONTEXT['records']},indent=2))
        CONTEXT['step'] = int(state.global_step)+1

    Observer.on_train_begin = begin
    Observer.on_step_end = end
