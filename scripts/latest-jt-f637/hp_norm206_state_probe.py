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

"""Observe one actual norm gradient and AdamW state under the shared-up candidate."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import capture_hp_native
import hyper_parallel.trainer.base as trainer_base
import native_clip_reduction_factor
from observe_hp_preclip_norm_adjacent import mf_gradient

TARGET = 'mtp.layers.0.final_layernorm.weight'
CONTEXT = {}


def save(directory, kind, tensor):
    """Record a small local tensor without injecting reference state."""
    tensor = tensor.to_local() if hasattr(tensor, 'to_local') else tensor
    array = tensor.detach().float().cpu().numpy()
    if array.size != 5120:
        raise ValueError('Unexpected norm vector size')
    filename = kind + '.npy'
    np.save(directory / filename, array, allow_pickle=False)
    return {'kind': kind, 'file': filename, 'shape': list(array.shape), 'dtype': str(tensor.dtype),
            'sha256': hashlib.sha256(array.tobytes()).hexdigest()}


def install():
    original_begin = capture_hp_native.Observer.on_train_begin
    original_end = capture_hp_native.Observer.on_step_end
    original_install = native_clip_reduction_factor.install

    def begin(self, state, **kwargs):
        original_begin(self, state, **kwargs)
        root = Path('/state/jt-recipe-20261014') / self.output.name
        root.mkdir(exist_ok=False)
        CONTEXT.update(step=int(state.global_step), root=root, output=self.output)

    def end(self, state, **kwargs):
        original_end(self, state, **kwargs)
        step = int(state.global_step)
        if step not in (204, 205, 206):
            return
        parameter = dict(self.trainer.model.named_parameters())[TARGET]
        matches = [leaf.state[parameter] for leaf in capture_hp_native.leaves(self.trainer.optimizer)
                   if parameter in leaf.state]
        if len(matches) != 1:
            raise ValueError('Target optimizer state not unique')
        current = matches[0]
        directory = CONTEXT['root'] / ('state-' + str(step))
        directory.mkdir(exist_ok=False)
        records = [save(directory, key, value) for key, value in [
            ('weight', parameter), ('exp_avg', current['exp_avg']), ('exp_avg_sq', current['exp_avg_sq'])]]
        report = {'step': step, 'name': TARGET, 'records': records,
                  'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        (self.output / ('NORM_STATE-' + str(step) + '.json')).write_text(json.dumps(report, indent=2))

    def clip_install():
        original_install()
        original_clip = trainer_base.clip_grad_norm_

        @torch.no_grad()
        def clip(model, *args, **kwargs):
            CONTEXT['step'] += 1
            step = CONTEXT['step']
            if step not in (205, 206):
                return original_clip(model, *args, **kwargs)
            directory = CONTEXT['root'] / ('gradient-' + str(step))
            directory.mkdir(exist_ok=False)
            parameters = dict(model.named_parameters())
            records = [save(directory, 'preclip', mf_gradient(parameters, TARGET))]
            norm = original_clip(model, *args, **kwargs)
            records.append(save(directory, 'postclip', mf_gradient(parameters, TARGET)))
            report = {'step': step, 'name': TARGET, 'norm': float(norm.cpu()), 'records': records}
            (CONTEXT['output'] / ('NORM_GRADIENT-' + str(step) + '.json')).write_text(json.dumps(report, indent=2))
            return norm

        trainer_base.clip_grad_norm_ = clip

    capture_hp_native.Observer.on_train_begin = begin
    capture_hp_native.Observer.on_step_end = end
    native_clip_reduction_factor.install = clip_install
