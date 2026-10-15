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

"""Locate full head input and returned-gradient divergence under audited observers."""
import hashlib
import json
from pathlib import Path

import numpy as np

from compare_head_selected_operands import stats
from compare_vocab_optimizer_trace import fidelity, read_rows

ROOT = Path('/alignment/phase12')
STATE = Path('/state/jt-recipe-20261014')
MF = 'mf-head-full-boundaries206'
HP = 'hp-jtf637-adamw-all-headfull206'


def main():
    checks = {}
    for label, run, baseline in [('mf', MF, 'mf-resume'), ('hp', HP, 'hp-jtf637-adamw-all-archive206')]:
        checks[label + '_exit_zero'] = (ROOT/'runs'/(run+'.exit')).read_text().strip() == '0'
        checks.update({label + '_' + key: value for key, value in fidelity(run, baseline, 6).items()})
    checks['hp_optimizer_coordinates_exact'] = (ROOT/'runs'/HP/'VOCAB_OPTIMIZER_TRACE.jsonl').read_bytes() == (
        ROOT/'runs/hp-jtf637-adamw-all-archive206/VOCAB_OPTIMIZER_TRACE.jsonl').read_bytes()
    def values_only(run):
        return [(r['step'], [(item['name'], item['values']) for item in r['records']])
                for r in read_rows(ROOT/'runs'/run/'HEAD_WEIGHT_BOUNDARIES.jsonl')]
    checks['mf_branch_raw_clipped_norm_exact'] = values_only(MF) == values_only('mf-head-weight-boundaries206-r2')
    target = ROOT/'evidence/head-full-input-gradient-boundaries.json'
    if not all(checks.values()):
        target.write_text(json.dumps({'complete': False, 'fidelity': checks}, indent=2))
        raise ValueError(checks)
    rows = read_rows(ROOT/'runs'/MF/'HEAD_FULL_BOUNDARIES.jsonl')
    directory = STATE/MF/'deep'
    results = []
    for step in [201, 205]:
        records = {(r['name'], r['kind']): r for r in next(row for row in rows if row['step'] == step)['records']}
        for branch, call in [('model.mtp.output_layer', (step-201)*2+1), ('model.output_layer', (step-201)*2+2)]:
            with np.load(ROOT/'runs'/HP/f'head-full-call{call}.npz') as hp:
                for kind in ['x', 'dx']:
                    record = records[branch, kind]
                    mf = np.load(directory/record['file'], mmap_mode='r')
                    if hashlib.sha256(mf.tobytes()).hexdigest() != record['sha256']:
                        raise ValueError('MF full operand hash changed')
                    mf = mf.reshape(-1, 5120)
                    actual = hp[kind]
                    comparison = stats(actual, mf)
                    different = np.argwhere(actual != mf)
                    examples = [{'token': int(i), 'hidden': int(j), 'hp': float(actual[i,j]), 'mf': float(mf[i,j])}
                                for i,j in different[:30]]
                    results.append({'step': step, 'branch': branch, 'kind': kind, 'comparison': comparison,
                                    'first_different_examples': examples})
    report = {'complete': True, 'fidelity': checks, 'records': results,
              'scope': 'Complete4096x5120headinputanddinputfor201and205, branchweightderivativesremainselected541coordinates.'}
    target.write_text(json.dumps(report, indent=2))
    print(json.dumps({'fidelity': checks, 'records': [{k:v for k,v in r.items() if k != 'first_different_examples'} for r in results]}))


if __name__ == '__main__':
    main()
