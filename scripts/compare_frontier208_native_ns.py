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

"""Audit full-training NS observations before comparing optimizer boundaries."""
import hashlib
import json
from pathlib import Path

import numpy as np
from compare_vocab_optimizer_trace import fidelity
from compare_frontier208_state import stats

ROOT = Path('/alignment/phase12')
STATE = Path('/state/jt-recipe-20261014')


def main():
    run = 'mf-frontier208-ns-train'
    directory = ROOT / 'runs' / run
    control = ROOT / 'runs/mf-frontier208-state'
    check = fidelity(run, 'mf-resume', 8)
    check['exit_zero'] = (ROOT / 'runs' / (run + '.exit')).read_text().strip() == '0'
    for step in [207, 208]:
        name = 'FRONTIER_STATE-' + str(step) + '.json'
        check['state_exact_' + str(step)] = json.loads((directory / name).read_text()) == json.loads((control / name).read_text())
    actual_grads = json.loads((directory / 'FRONTIER_GRADIENTS-208.json').read_text())
    control_grads = json.loads((control / 'FRONTIER_GRADIENTS-208.json').read_text())
    canonical = lambda rows: [{key: value for key, value in row.items() if key != 'file'} for row in rows]
    check['actual_gradients_exact'] = canonical(actual_grads) == canonical(control_grads)
    rows = json.loads((directory / 'NS_BOUNDARIES-208.json').read_text())
    check['48_boundaries'] = len(rows) == 48
    report = {'complete': True, 'fidelity': check, 'usable_as_native_training': all(check.values()),
              'scope': 'Read-only observations of actual208 Muon updates; no teacher replacement.'}
    destination = ROOT / 'evidence/frontier208-native-ns-audit.json'
    if not report['usable_as_native_training']:
        destination.write_text(json.dumps(report, indent=2))
        print(json.dumps(report))
        raise ValueError('Full-training observation fidelity failed')
    records = {row['prefix']: row for row in rows}
    comparisons = []
    scalar_values = []
    for label in ['q', 'router']:
        for name in ['pre_ns', 'norm', 'denominator', 'normalized'] + [
                kind + '-' + str(index) for index in range(1, 6) for kind in ['gram', 'poly', 'product', 'stage']]:
            row = records[label + '-' + name]
            mf = np.load(STATE / 'mf-frontier208-ns-operators' / row['file'])
            if hashlib.sha256(mf.tobytes()).hexdigest() != row['sha256']:
                raise ValueError('Actual full-training NS array changed')
            if name in ('norm', 'denominator'):
                scalar_values.append({'label': label, 'boundary': name, 'shape': list(mf.shape),
                                      'values': mf.tolist(), 'sha256': row['sha256']})
                continue
            hp = np.load(STATE / 'hp-frontier208-muon-replay' / label / (name + '.npy'))
            isolated_files = list((STATE / 'mf-frontier208-muon-replay' / label).glob(name + '_*.npy'))
            if len(isolated_files) != 1:
                raise ValueError('Ambiguous isolated MF boundary')
            isolated = np.load(isolated_files[0])
            comparisons.append({'label': label, 'boundary': name, 'hp_vs_native': stats(hp, mf),
                                'mf_isolated_vs_native': stats(isolated, mf)})
    report.update(comparisons=comparisons, native_scalars=scalar_values,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    destination.write_text(json.dumps(report, indent=2))
    print(json.dumps({'fidelity': check, 'scalars': scalar_values, 'boundaries': [
        {'label': row['label'], 'boundary': row['boundary'],
         'hp_different': row['hp_vs_native']['different'], 'hp_max_abs': row['hp_vs_native']['max_abs'],
         'mf_isolated_different': row['mf_isolated_vs_native']['different']} for row in comparisons]}))


if __name__ == '__main__':
    main()
