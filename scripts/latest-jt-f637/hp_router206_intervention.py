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

"""Diagnostic-only replacement of the measured router206 model update difference."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from capture_hp_native import Observer

ROOT = Path('/alignment/phase12')
STATE = Path('/state/jt-recipe-20261014')
TARGET = 'model.layers.1.mlp.gate.weight'
CONTEXT = {}


def load_weight(run):
    """Load the small actual postupdate router state and verify its complete hash."""
    report = json.loads((ROOT / 'runs' / run / 'ROUTER_STATE-206.json').read_text())
    row = next(item for item in report['records'] if item['kind'] == 'weight')
    value = np.load(STATE / run / 'state-206' / row['file'])
    if value.shape != (8, 5120) or hashlib.sha256(value.tobytes()).hexdigest() != row['sha256']:
        raise ValueError('Router state shape or content changed')
    return value, row['sha256']


def install():
    """Change one postupdate model tensor only once; keep every optimizer moment native."""
    original_begin = Observer.on_train_begin
    original_end = Observer.on_step_end

    def begin(self, state, **kwargs):
        original_begin(self, state, **kwargs)
        expected, expected_hash = load_weight('mf-router206-state')
        control, control_hash = load_weight('hp-jtf637-sharedup-router206-state')
        coordinates = np.flatnonzero(expected.reshape(-1) != control.reshape(-1))
        if not coordinates.size:
            raise ValueError('No measured router weight difference to intervene on')
        CONTEXT.update(expected=expected, expected_hash=expected_hash, control_hash=control_hash,
                       coordinates=coordinates)
        marker = {'diagnostic_only': True, 'eligible_for_acceptance': False, 'step': 206, 'parameter': TARGET,
                  'scope': 'Replace only differing router model coordinates once after206; optimizer states and later updates unchanged.',
                  'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        (self.output / 'DIAGNOSTIC_ONLY.json').write_text(json.dumps(marker, indent=2))

    @torch.no_grad()
    def end(self, state, **kwargs):
        original_end(self, state, **kwargs)
        if int(state.global_step) != 206:
            return
        parameter = dict(self.trainer.model.named_parameters())[TARGET]
        parameter = parameter.to_local() if hasattr(parameter, 'to_local') else parameter
        before = parameter.detach().float().cpu().numpy().copy()
        if hashlib.sha256(before.tobytes()).hexdigest() != CONTEXT['control_hash']:
            raise ValueError('Independent router control weight did not reproduce')
        coordinates = CONTEXT['coordinates']
        index = torch.tensor(coordinates, dtype=torch.long, device=parameter.device)
        value = torch.from_numpy(CONTEXT['expected'].reshape(-1)[coordinates].copy()).to(parameter.device)
        parameter.view(-1).index_copy_(0, index, value)
        after = parameter.detach().float().cpu().numpy()
        digest = hashlib.sha256(after.tobytes()).hexdigest()
        if digest != CONTEXT['expected_hash']:
            raise ValueError('Router intervention did not restore exact native weight')
        report = {'step': 206, 'changed_elements': int(np.count_nonzero(before != after)),
                  'all_other_elements_unchanged': bool(np.array_equal(np.flatnonzero(before.reshape(-1) != after.reshape(-1)), coordinates)),
                  'before_sha256': CONTEXT['control_hash'], 'after_sha256': digest,
                  'native_mf_sha256': CONTEXT['expected_hash'], 'optimizer_state_modified': False}
        (self.output / 'ROUTER206_INTERVENTION.json').write_text(json.dumps(report, indent=2))

    Observer.on_train_begin = begin
    Observer.on_step_end = end
