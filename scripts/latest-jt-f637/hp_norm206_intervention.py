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

"""Diagnostic-only removal of the measured single norm206 gradient difference."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from capture_hp_native import Observer
import hyper_parallel.trainer.base as trainer_base
import native_clip_reduction_factor
from observe_hp_preclip_norm_adjacent import mf_gradient

ROOT = Path('/alignment/phase12')
STATE = Path('/state/jt-recipe-20261014')
TARGET = 'mtp.layers.0.final_layernorm.weight'
CONTEXT = {}


def install():
    """Mark the trajectory ineligible before training, then change only one measured element."""
    original_begin = Observer.on_train_begin
    original_install = native_clip_reduction_factor.install

    def begin(self, state, **kwargs):
        original_begin(self, state, **kwargs)
        rows = json.loads((ROOT / 'runs/mf-norm206-state/NORM_GRADIENTS-206.json').read_text())
        row = next(row for row in rows if row['name'] == TARGET)
        expected = np.load(STATE / 'mf-norm206-state/gradients' / row['file'])
        if hashlib.sha256(expected.tobytes()).hexdigest() != row['sha256']:
            raise ValueError('Native norm reference changed')
        observed = json.loads((ROOT / 'runs/hp-jtf637-sharedup-norm206-state/NORM_GRADIENT-206.json').read_text())
        actual = next(row for row in observed['records'] if row['kind'] == 'postclip')
        CONTEXT.update(step=int(state.global_step), expected=expected, expected_hash=row['sha256'],
                       original_hash=actual['sha256'], output=self.output)
        marker = {'diagnostic_only': True, 'eligible_for_acceptance': False, 'step': 206, 'parameter': TARGET,
                  'scope': 'Replace only the measured one differing postclip norm gradient element; no later teacher state.',
                  'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        (self.output / 'DIAGNOSTIC_ONLY.json').write_text(json.dumps(marker, indent=2))

    def clip_install():
        original_install()
        original_clip = trainer_base.clip_grad_norm_

        @torch.no_grad()
        def clip(model, *args, **kwargs):
            CONTEXT['step'] += 1
            norm = original_clip(model, *args, **kwargs)
            if CONTEXT['step'] != 206:
                return norm
            gradient = mf_gradient(dict(model.named_parameters()), TARGET)
            before = gradient.cpu().numpy().copy()
            if hashlib.sha256(before.tobytes()).hexdigest() != CONTEXT['original_hash']:
                raise ValueError('Independent control gradient did not reproduce')
            coordinates = np.flatnonzero(before != CONTEXT['expected'])
            if coordinates.tolist() != [4903]:
                raise ValueError('Intervention must change exactly the measured one coordinate')
            index = torch.tensor(coordinates, dtype=torch.long, device=gradient.device)
            value = torch.from_numpy(CONTEXT['expected'][coordinates].copy()).to(gradient.device)
            gradient.view(-1).index_copy_(0, index, value)
            after = gradient.cpu().numpy()
            digest = hashlib.sha256(after.tobytes()).hexdigest()
            if digest != CONTEXT['expected_hash'] or np.count_nonzero(before != after) != 1:
                raise ValueError('Intervention hash/count failed')
            report = {'step': 206, 'changed_elements': 1, 'all_other_elements_unchanged': True,
                      'before_sha256': CONTEXT['original_hash'], 'after_sha256': digest,
                      'native_mf_sha256': CONTEXT['expected_hash']}
            (CONTEXT['output'] / 'NORM206_INTERVENTION.json').write_text(json.dumps(report, indent=2))
            print('[P12-NORM206-DIAGNOSTIC]', json.dumps(report), flush=True)
            return norm

        trainer_base.clip_grad_norm_ = clip

    Observer.on_train_begin = begin
    native_clip_reduction_factor.install = clip_install
