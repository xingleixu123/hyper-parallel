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

"""Compare model204 against native MF without saving or replacing model weights."""
import json
from pathlib import Path

from capture_hp_native import Observer, numpy_value
from observe_hp_poststep202 import converter, model
from observe_hp_poststep_model import compare

ROOT = Path('/state/jt-recipe-20261014')


def install():
    """Observe the actual step204 state without changing the independent trajectory."""
    original = Observer.on_step_end

    def end(self, state, **kwargs):
        original(self, state, **kwargs)
        if state.global_step != 204:
            return
        conversion = converter()
        before = model(ROOT / 'mf-grad203-state/step-000203', conversion)
        expected = model(ROOT / 'mf-grad204-state-r2/step-000204', conversion)
        actual = self.trainer.model.state_dict()
        records = []
        output = Path('/alignment/phase12/evidence') / (self.output.name + '-poststep204.json')
        for name in sorted(expected):
            row = compare(name, numpy_value(actual[name]), expected[name], before[name])
            records.append(row)
            output.write_text(json.dumps({'complete': False, 'records': records}, indent=2))
        if len(records) != 48:
            raise ValueError('Expected all48 model tensors')
        output.write_text(json.dumps({'complete': True, 'step': 204, 'records': records,
            'scope': 'Actual independent model; MF reference is compared only, never injected.'}, indent=2))

    Observer.on_step_end = end
