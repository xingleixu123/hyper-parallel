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

"""Observe native HP final-norm operator results and branch gradients only at236."""
import hashlib
import json
from pathlib import Path
from types import MethodType

import numpy as np
import torch
import torch_npu

from capture_hp_native import Observer
from one_factor_probe import MFNorm
from mindspore272_stat_hash import native_dump_sha1

CONTEXT = {'step': 201, 'active': None, 'output': None, 'directory': None, 'records': []}
TARGETS = ['model.norm', 'mtp.layers.0.final_layernorm', 'model.layers.1.mlp']


def save(name, kind, tensor):
    """Store exact236 observations; never use a saved tensor in training."""
    if CONTEXT['step'] != 236:
        return
    array = np.ascontiguousarray(tensor.detach().float().cpu().numpy())
    digest = hashlib.sha256(array.tobytes()).hexdigest()
    prior = [row for row in CONTEXT['records'] if (row['name'], row['kind']) == (name, kind)]
    if prior:
        if len(prior) != 1 or prior[0]['sha256'] != digest:
            raise ValueError('Recomputed boundary changed: ' + name + ':' + kind)
        return
    native = tensor.detach().contiguous().cpu()
    native_array = native.view(torch.uint16).numpy() if native.dtype == torch.bfloat16 else native.numpy()
    CONTEXT['records'].append({'name': name, 'kind': kind,
                               'shape': list(array.shape), 'sha256': digest, 'dtype': str(tensor.dtype),
                               'fp32_native_dump_sha1': native_dump_sha1(array),
                               'native_dump_sha1': native_dump_sha1(native_array)})



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
        if name.endswith('.mlp'):
            save(name, 'x', values)
        CONTEXT['active'] = name
        try:
            result = original(values)
        finally:
            CONTEXT['active'] = None
        if name.endswith('.mlp'):
            save(name, 'y_fp32', result)
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
        CONTEXT.update(step=int(state.global_step)+1, output=self.output)
        modules = dict(self.trainer.model.named_modules())
        metadata = []
        for name in TARGETS:
            module = modules[name]
            if not name.endswith('.mlp') and not isinstance(module, MFNorm):
                raise ValueError('Unexpected final norm:' + repr(type(module)))
            metadata.append({'name': name, 'class': str(type(module)), 'epsilon': getattr(module, 'variance_epsilon', None)})
            attach(module, name)
        (self.output / 'BACKWARD236_OBSERVER.json').write_text(json.dumps({
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'modules': metadata, 'teacher_values_used': False, 'retained_steps': [236],
            'raw_arrays_written': False,
            'hash_helper_sha256': hashlib.sha256(Path(__file__).with_name('mindspore272_stat_hash.py').read_bytes()).hexdigest()}, indent=2))

    def end(self, state, **kwargs):
        original_end(self, state, **kwargs)
        if int(state.global_step) == 236:
            if len(CONTEXT['records']) != 16:
                raise ValueError('Expected16backwardboundaries:' + str(len(CONTEXT['records'])))
            (self.output / 'BACKWARD236_BOUNDARIES.json').write_text(json.dumps({
                'step': 236, 'records': CONTEXT['records']}, indent=2))
        CONTEXT['step'] = int(state.global_step) + 1

    torch_npu.npu_rms_norm = norm
    Observer.on_train_begin = begin
    Observer.on_step_end = end
