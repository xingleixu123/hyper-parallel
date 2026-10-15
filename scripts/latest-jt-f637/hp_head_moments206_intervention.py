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

"""Isolate accumulated head AdamW moments with a single explicit teacher intervention."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

import capture_hp_native
import hp_head206_intervention as model_intervention

ROOT = Path('/alignment/phase12')
CONTEXT = {}


def owner(observer):
    """Resolve the actual shared-head optimizer slots and its unmodified step counter."""
    parameter = dict(observer.trainer.model.named_parameters())['lm_head.weight']
    matches = [leaf for leaf in capture_hp_native.leaves(observer.trainer.optimizer) if parameter in leaf.state]
    if len(matches) != 1:
        raise ValueError('Head optimizer ownership is not unique')
    optimizer = matches[0]
    groups = [group for group in optimizer.param_groups if any(value is parameter for value in group['params'])]
    if len(groups) != 1:
        raise ValueError('Head parameter group is not unique')
    return parameter, optimizer.state[parameter], groups[0]


def install() -> None:
    """Intervene once at206, then observe independently updated moment hashes at207."""
    original_begin = capture_hp_native.Observer.on_train_begin
    original_end = capture_hp_native.Observer.on_step_end
    mode = os.environ['P12_HEAD_MOMENT_DIAG_MODE']
    if mode not in ('moments', 'head_moments', 'head_router_moments'):
        raise ValueError('Unknown declared diagnostic mode')

    def begin(self, state, **kwargs):
        original_begin(self, state, **kwargs)
        report = json.loads((ROOT / 'evidence/head-moments206-comparison.json').read_text())
        if not report['complete'] or not report['all_patches_ready']:
            raise ValueError('Full moment observations have not been verified')
        patches = {}
        for kind in ['exp_avg', 'exp_avg_sq']:
            row = report['moments'][kind]
            path = Path(row['patch_file'])
            if hashlib.sha256(path.read_bytes()).hexdigest() != row['patch_sha256']:
                raise ValueError('Moment patch changed')
            with np.load(path) as archive:
                indices, replacement = archive['indices'].copy(), archive['mf'].copy()
            if indices.dtype != np.int64 or indices.size != row['different']:
                raise ValueError('Unexpected moment patch shape')
            expected = {family: next(item['sha256'] for item in report['metadata'][family]['records']
                                     if item['kind'] == kind) for family in ['mf', 'hp']}
            patches[kind] = (indices, replacement, expected['hp'], expected['mf'])
        CONTEXT.update(report=report, patches=patches)
        model_marker = None
        if mode != 'moments':
            model_marker = json.loads((self.output / 'DIAGNOSTIC_ONLY.json').read_text())
            if not model_marker['diagnostic_only'] or model_marker['optimizer_state_modified']:
                raise ValueError('Unexpected preceding model intervention')
        marker = {'diagnostic_only': True, 'eligible_for_acceptance': False, 'step': 206, 'mode': mode,
                  'teacher_moment_values_used': True, 'optimizer_counter_modified': False,
                  'model_intervention': model_marker,
                  'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'patch_helper_sha256': hashlib.sha256(Path(model_intervention.__file__).read_bytes()).hexdigest(),
                  'patches': {kind: report['moments'][kind]['patch_sha256'] for kind in patches}}
        (self.output / 'DIAGNOSTIC_ONLY.json').write_text(json.dumps(marker, indent=2))

    @torch.no_grad()
    def end(self, state, **kwargs):
        original_end(self, state, **kwargs)
        step = int(state.global_step)
        if step not in (206, 207):
            return
        parameter, current, group = owner(self)
        if int(group['step']) != step:
            raise ValueError('Optimizer and trainer counters differ')
        if step == 206:
            changed = {kind: model_intervention.apply_patch(current[kind], *patch)
                       for kind, patch in CONTEXT['patches'].items()}
            report = {'step': step, 'mode': mode, 'changed': changed,
                      'optimizer_step_before': step, 'optimizer_step_after': int(group['step']),
                      'other_optimizer_slots_modified': False,
                      'weight_fp32_sha256': model_intervention.digest(capture_hp_native.numpy_value(parameter))}
            (self.output / 'HEAD_MOMENTS206_INTERVENTION.json').write_text(json.dumps(report, indent=2))
        else:
            records = [{'kind': kind, 'sha256': model_intervention.digest(capture_hp_native.numpy_value(current[kind]))}
                       for kind in ['exp_avg', 'exp_avg_sq']]
            report = {'step': step, 'records': records, 'optimizer_step': int(group['step']),
                      'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
            (self.output / 'HEAD_MOMENTS207_HASH.json').write_text(json.dumps(report, indent=2))

    capture_hp_native.Observer.on_train_begin = begin
    capture_hp_native.Observer.on_step_end = end
