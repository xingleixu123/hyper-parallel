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

"""Diagnostic-only2x2 model-state interventions at the206 frontier."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from capture_hp_native import Observer, numpy_value
import hp_router206_intervention

ROOT = Path('/alignment/phase12')
CONTEXT = {}


def digest(array):
    """Hash actual contiguous FP32 storage without making a second full-sized byte copy."""
    if array.dtype != np.float32 or not array.flags.c_contiguous:
        raise ValueError('Expected contiguous FP32 model storage')
    return hashlib.sha256(memoryview(array).cast('B')).hexdigest()


def apply_patch(parameter, indices, replacement, expected_before, expected_after):
    """Check the entire tensor before and after changing only measured coordinates."""
    parameter = parameter.to_local() if hasattr(parameter, 'to_local') else parameter
    before = numpy_value(parameter)
    if digest(before) != expected_before:
        raise ValueError('Model control state did not reproduce')
    original_values = before.reshape(-1)[indices].copy()
    del before
    if not np.all(original_values != replacement):
        raise ValueError('Patch contains unchanged coordinates')
    index = torch.from_numpy(indices).to(parameter.device)
    values = torch.from_numpy(replacement).to(parameter.device)
    parameter.view(-1).index_copy_(0, index, values)
    after = numpy_value(parameter)
    if digest(after) != expected_after:
        raise ValueError('Complete native model hash not restored by the patch')
    return {'changed_elements': int(indices.size), 'before_sha256': expected_before,
            'after_sha256': expected_after, 'other_elements_unchanged': True}


def install():
    """Replace model values once; preserve all optimizer slots and later independent updates."""
    original_begin = Observer.on_train_begin
    original_end = Observer.on_step_end
    mode = os.environ['P12_MODEL206_INTERVENTION']
    if mode not in ('head', 'head_router'):
        raise ValueError('Expected a declared head/router diagnostic scope')

    def begin(self, state, **kwargs):
        original_begin(self, state, **kwargs)
        report = json.loads((ROOT / 'evidence/head206-model-comparison.json').read_text())
        patch = Path(report['patch_file'])
        if hashlib.sha256(patch.read_bytes()).hexdigest() != report['patch_sha256']:
            raise ValueError('Compact head patch changed')
        with np.load(patch) as archive:
            indices = archive['indices'].copy()
            expected = archive['mf'].copy()
        if indices.dtype != np.int64 or indices.size != report['different_fp32']:
            raise ValueError('Unexpected compact patch shape')
        CONTEXT.update(report=report, indices=indices, expected=expected)
        if mode == 'head_router':
            mf, mf_hash = hp_router206_intervention.load_weight('mf-router206-state')
            hp, hp_hash = hp_router206_intervention.load_weight('hp-jtf637-sharedup-router206-state')
            indices = np.flatnonzero(mf.reshape(-1) != hp.reshape(-1))
            if indices.size != 12:
                raise ValueError('Router patch no longer has12 differences')
            CONTEXT['router'] = (indices, mf.reshape(-1)[indices].copy(), hp_hash, mf_hash)
        marker = {'diagnostic_only': True, 'eligible_for_acceptance': False, 'step': 206, 'mode': mode,
                  'optimizer_state_modified': False, 'patch_sha256': report['patch_sha256'],
                  'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'router_helper_sha256': hashlib.sha256(Path(hp_router206_intervention.__file__).read_bytes()).hexdigest()}
        (self.output / 'DIAGNOSTIC_ONLY.json').write_text(json.dumps(marker, indent=2))

    @torch.no_grad()
    def end(self, state, **kwargs):
        original_end(self, state, **kwargs)
        if int(state.global_step) != 206:
            return
        parameters = dict(self.trainer.model.named_parameters())
        records = CONTEXT['report']['model_records']
        changed = {'lm_head.weight': apply_patch(parameters['lm_head.weight'], CONTEXT['indices'], CONTEXT['expected'],
                   records['hp']['fp32_sha256'], records['mf']['fp32_sha256'])}
        if mode == 'head_router':
            changed['model.layers.1.mlp.gate.weight'] = apply_patch(parameters['model.layers.1.mlp.gate.weight'], *CONTEXT['router'])
        report = {'step': 206, 'mode': mode, 'changed': changed, 'optimizer_state_modified': False}
        (self.output / 'MODEL206_INTERVENTION.json').write_text(json.dumps(report, indent=2))

    Observer.on_train_begin = begin
    Observer.on_step_end = end
