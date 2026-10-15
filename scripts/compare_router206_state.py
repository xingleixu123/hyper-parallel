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

"""Quantify router206 gradient/state differences only after observer fidelity checks."""
import hashlib
import json
from pathlib import Path

import numpy as np
from compare_vocab_optimizer_trace import fidelity
from compare_head_selected_operands import stats

ROOT = Path('/alignment/phase12')
STATE = Path('/state/jt-recipe-20261014')
MF = 'mf-router206-state'
HP = 'hp-jtf637-sharedup-router206-state'
TARGET = 'decoder.layers.1.mlp.router.weight'


def load(folder, row):
    """Load an immutable small observation."""
    value = np.load(folder / row['file'])
    if hashlib.sha256(value.tobytes()).hexdigest() != row['sha256']:
        raise ValueError('Observed array changed')
    return value


def main():
    checks = {}
    for label, run, base in [('mf', MF, 'mf-resume'), ('hp', HP, 'hp-jtf637-adamw-all-sharedupfp32206')]:
        checks[label + '_exit_zero'] = (ROOT / 'runs' / (run + '.exit')).read_text().strip() == '0'
        checks.update({label + '_' + key: value for key, value in fidelity(run, base, 6).items()})
    checks['hp_optimizer_trace_exact'] = (ROOT / 'runs' / HP / 'VOCAB_OPTIMIZER_TRACE.jsonl').read_bytes() == (
        ROOT / 'runs/hp-jtf637-adamw-all-sharedupfp32206/VOCAB_OPTIMIZER_TRACE.jsonl').read_bytes()
    states, gradients = {}, {}
    for step in (204, 205, 206):
        observations = {}
        for label, run, source in [('mf', MF, 'tools/mf_router206_state_tap.py'),
                                    ('hp', HP, 'tools-f637/hp_router206_state_probe.py')]:
            report = json.loads((ROOT / 'runs' / run / ('ROUTER_STATE-' + str(step) + '.json')).read_text())
            checks[label + '_source_exact'] = report['source_sha256'] == hashlib.sha256((ROOT / source).read_bytes()).hexdigest()
            observations[label] = {row['kind']: load(STATE / run / ('state-' + str(step)), row)
                                   for row in report['records']}
        states[str(step)] = {key: stats(observations['hp'][key], observations['mf'][key])
                             for key in ['weight', 'muon_m']}
    for step in (205, 206):
        rows = json.loads((ROOT / 'runs' / MF / ('ROUTER_GRADIENTS-' + str(step) + '.json')).read_text())
        row = next(row for row in rows if row['name'] == TARGET)
        mf = load(STATE / MF / 'gradients', row)
        old = next(item for item in json.loads((ROOT / 'runs/mf-grad205-state' / ('GRADIENTS-' + str(step) + '.json')).read_text()) if item['name'] == TARGET)
        checks['mf_gradient' + str(step) + '_old_hash_exact'] = row['sha256'] == old['sha256']
        hp_report = json.loads((ROOT / 'runs' / HP / ('ROUTER_GRADIENT-' + str(step) + '.json')).read_text())
        hp_row = next(row for row in hp_report['records'] if row['kind'] == 'postclip')
        hp = load(STATE / HP / ('gradient-' + str(step)), hp_row)
        gradients[str(step)] = stats(hp, mf)
        if step == 206:
            old_hp = json.loads((ROOT / 'evidence/hp-jtf637-adamw-all-sharedup-grad206hash-grad206.json').read_text())
            old_hp = next(row for row in old_hp['records'] if row['name'] == TARGET)
            checks['hp_gradient206_old_hash_exact'] = hp_row['sha256'] == old_hp['actual_sha256']
    if not all(checks.values()):
        raise ValueError(checks)
    report = {'complete': True, 'fidelity': checks, 'states': states, 'gradients': gradients,
              'scope': 'Small actual router state after each update; causal impact requires a separate intervention.'}
    (ROOT / 'evidence/router206-gradient-state-comparison.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == '__main__':
    main()
