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

"""Select the existing Muon learning-rate folding option after checkpoint restoration."""
import hashlib
import json
from pathlib import Path

from capture_hp_native import Observer, leaves
from hyper_parallel.core.optimizer.muon import Muon


def install():
    original = Observer.on_train_begin

    def begin(self, state, **kwargs):
        original(self, state, **kwargs)
        records = []
        for leaf in leaves(self.trainer.optimizer):
            if not isinstance(leaf, Muon):
                continue
            for index, group in enumerate(leaf.param_groups):
                records.append({'group': index, 'restored_apply_lr_in_update': group['apply_lr_in_update'],
                                'effective_apply_lr_in_update': True, 'lr': group['lr']})
                group['apply_lr_in_update'] = True
        if not records:
            raise ValueError('No native Muon group found')
        report = {'single_factor': 'Existing apply_lr_in_update=True; preserve original scalar precision',
                  'teacher_values_used': False, 'groups': records,
                  'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        (self.output / 'MUON_LR_FOLDING.json').write_text(json.dumps(report, indent=2))
        print('[P12-LR-FOLDING]', json.dumps(report), flush=True)

    Observer.on_train_begin = begin
