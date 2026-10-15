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

"""Observe two newly divergent model weights, Muon buffers and live gradients around222."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import capture_hp_native
import hyper_parallel.trainer.base as trainer_base
import native_clip_reduction_factor
from observe_hp_preclip_norm_adjacent import mf_gradient

TARGETS = {'main': 'model.layers.0.self_attn.linear_qkv.weight', 'mtp': 'mtp.layers.0.transformer_layer.self_attn.linear_qkv.weight'}
MF_TARGETS = {'q_main': 'decoder.layers.0.self_attention.linear_q_down_proj.weight', 'kv_main': 'decoder.layers.0.self_attention.linear_kv_down_proj.weight', 'q_mtp': 'mtp.layers.0.transformer_layer.self_attention.linear_q_down_proj.weight', 'kv_mtp': 'mtp.layers.0.transformer_layer.self_attention.linear_kv_down_proj.weight'}
SHAPES = {'main': (2112, 5120), 'mtp': (2112, 5120)}
GRAD_SHAPES = {'q_main': (1536, 5120), 'kv_main': (576, 5120), 'q_mtp': (1536, 5120), 'kv_mtp': (576, 5120)}
CONTEXT = {}


def save(directory, label, kind, tensor):
    tensor = tensor.to_local() if hasattr(tensor, 'to_local') else tensor
    array = tensor.detach().float().cpu().numpy()
    if tuple(array.shape) != (GRAD_SHAPES if kind in ['preclip', 'postclip'] else SHAPES)[label]:
        raise ValueError('Unexpected actual tensor shape')
    filename = label + '-' + kind + '.npy'
    np.save(directory / filename, array, allow_pickle=False)
    return {'label': label, 'kind': kind, 'file': filename, 'shape': list(array.shape),
            'dtype': str(tensor.dtype), 'sha256': hashlib.sha256(array.tobytes()).hexdigest()}


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
        if step not in (221, 222):
            return
        parameters = dict(self.trainer.model.named_parameters())
        directory = CONTEXT['root'] / ('state-' + str(step))
        directory.mkdir(exist_ok=False)
        records = []
        for label, name in TARGETS.items():
            parameter = parameters[name]
            matches = [leaf.state[parameter] for leaf in capture_hp_native.leaves(self.trainer.optimizer)
                       if parameter in leaf.state]
            if len(matches) != 1:
                raise ValueError('Target optimizer state not unique')
            for kind, value in [('weight', parameter), ('muon_m', matches[0]['momentum_buffer'])]:
                records.append(save(directory, label, kind, value))
        report = {'step': step, 'targets': TARGETS, 'records': records,
                  'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'read_only': True, 'teacher_values_used': False}
        (self.output / ('FRONTIER_STATE-' + str(step) + '.json')).write_text(json.dumps(report, indent=2))

    def clip_install():
        original_install()
        original_clip = trainer_base.clip_grad_norm_

        @torch.no_grad()
        def clip(model, *args, **kwargs):
            CONTEXT['step'] += 1
            step = CONTEXT['step']
            if step != 222:
                return original_clip(model, *args, **kwargs)
            directory = CONTEXT['root'] / ('gradient-' + str(step))
            directory.mkdir(exist_ok=False)
            parameters = dict(model.named_parameters())
            records = [save(directory, label, 'preclip', mf_gradient(parameters, name))
                       for label, name in MF_TARGETS.items()]
            norm = original_clip(model, *args, **kwargs)
            records.extend(save(directory, label, 'postclip', mf_gradient(parameters, name))
                           for label, name in MF_TARGETS.items())
            report = {'step': step, 'targets': MF_TARGETS, 'norm': float(norm.cpu()), 'records': records,
                      'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
            (CONTEXT['output'] / ('FRONTIER_GRADIENT-' + str(step) + '.json')).write_text(json.dumps(report, indent=2))
            return norm

        trainer_base.clip_grad_norm_ = clip

    capture_hp_native.Observer.on_train_begin = begin
    capture_hp_native.Observer.on_step_end = end
    native_clip_reduction_factor.install = clip_install
