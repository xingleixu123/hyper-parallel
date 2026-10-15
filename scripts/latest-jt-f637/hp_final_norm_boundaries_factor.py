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

"""Observe native HP final-norm operator results and branch gradients only at205."""
import hashlib
import json
from pathlib import Path
from types import MethodType

import numpy as np
import torch
import torch_npu

from capture_hp_native import Observer
from one_factor_probe import MFNorm

CONTEXT = {'step': 201, 'active': None, 'output': None, 'directory': None, 'records': []}
TARGETS = ['model.norm', 'mtp.layers.0.final_layernorm']


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


def attach(module, name):
    """Retain the native forward and autograd implementation with identity observations."""
    original = module.forward

    def forward(self, values):
        del self
        values = InputBoundary.apply(values, name)
        if CONTEXT['active'] is not None:
            raise ValueError('Unexpected nested final norm')
        CONTEXT['active'] = name
        try:
            result = original(values)
        finally:
            CONTEXT['active'] = None
        def gradient_hook(gradient):
            save(name, 'dy', gradient)
            return gradient
        result.register_hook(gradient_hook)
        return result

    module.forward = MethodType(forward, module)


def install():
    """Bind targets from the actual FSDP model and capture original NPU norm outputs."""
    original_begin = Observer.on_train_begin
    original_end = Observer.on_step_end
    original_norm = torch_npu.npu_rms_norm

    def norm(*args, **kwargs):
        result = original_norm(*args, **kwargs)
        name = CONTEXT['active']
        if name is not None:
            save(name, 'x', args[0])
            save(name, 'weight', args[1])
            save(name, 'y_fp32', result[0])
            save(name, 'inverse_rms', result[1])
        return result

    def begin(self, state, **kwargs):
        original_begin(self, state, **kwargs)
        directory = Path('/state/jt-recipe-20261014') / self.output.name / 'norms'
        directory.mkdir(parents=True, exist_ok=False)
        CONTEXT.update(step=int(state.global_step)+1, output=self.output, directory=directory)
        modules = dict(self.trainer.model.named_modules())
        metadata = []
        for name in TARGETS:
            module = modules[name]
            if not isinstance(module, MFNorm):
                raise ValueError('Unexpected final norm:' + repr(type(module)))
            metadata.append({'name': name, 'class': str(type(module)), 'epsilon': module.variance_epsilon})
            attach(module, name)
        (self.output / 'FINAL_NORM_OBSERVER.json').write_text(json.dumps({
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'modules': metadata, 'teacher_values_used': False, 'retained_steps': [205]}, indent=2))

    def end(self, state, **kwargs):
        original_end(self, state, **kwargs)
        if int(state.global_step) == 205:
            if len(CONTEXT['records']) != 12:
                raise ValueError('Expected12finalnormboundaries:' + str(len(CONTEXT['records'])))
            (self.output / 'FINAL_NORM_BOUNDARIES.json').write_text(json.dumps({
                'step': 205, 'records': CONTEXT['records']}, indent=2))
        CONTEXT['step'] = int(state.global_step) + 1

    torch_npu.npu_rms_norm = norm
    Observer.on_train_begin = begin
    Observer.on_step_end = end
