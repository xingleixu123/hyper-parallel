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

"""Compare native MF selected head operands with executed HP head GEMMs."""
import hashlib
import json
from pathlib import Path

import numpy as np

from compare_vocab_optimizer_trace import fidelity, read_rows

ROOT = Path('/alignment/phase12')
STATE = Path('/state/jt-recipe-20261014')
MF = 'mf-head-selected-operands206'
HP = 'hp-jtf637-adamw-all-headcontig206-r3'


def stats(left, right):
    """Compare finite selected operand arrays without tolerance rounding."""
    if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError('Invalid operands')
    delta = left.astype(np.float64) - right.astype(np.float64)
    worst = np.unravel_index(np.argmax(np.abs(delta)), left.shape)
    return {'exact': bool(np.array_equal(left, right)), 'size': int(left.size),
            'different': int(np.count_nonzero(left != right)), 'max_abs': float(np.max(np.abs(delta))),
            'rel_l2': float(np.linalg.norm(delta) / max(np.linalg.norm(right.astype(np.float64)), 1e-300)),
            'worst_index': [int(v) for v in worst], 'hp_at_worst': float(left[worst]),
            'mf_at_worst': float(right[worst])}


def main():
    """Audit the observer and identify branch mapping from actual activation values."""
    if (ROOT/'runs'/(MF+'.exit')).read_text().strip() != '0':
        raise ValueError('MF probe did not finish successfully')
    checks = fidelity(MF, 'mf-resume', 6)
    if not all(checks.values()):
        raise ValueError(checks)
    directory = STATE/MF/'deep'
    manifest = json.loads((directory/'manifest.json').read_text())
    checks['observer_source_hash'] = manifest['source_sha256'] == hashlib.sha256(
        (ROOT/'tools/mf_head_selected_operands_tap.py').read_bytes()).hexdigest()
    rows = read_rows(ROOT/'runs'/MF/'HEAD_OPERANDS.jsonl')
    if [r['step'] for r in rows] != list(range(201, 207)):
        raise ValueError('Incomplete operand records')
    mf_arrays = {}
    for record in rows[0]['records']:
        key = record['name'], record['kind']
        array = np.load(directory/record['file'])
        if hashlib.sha256(array.tobytes()).hexdigest() != record['sha256']:
            raise ValueError('Operand dump hash changed')
        if key in mf_arrays and not np.array_equal(mf_arrays[key], array):
            raise ValueError('Different recomputation operand for ' + repr(key))
        mf_arrays[key] = array
    comparisons = []
    for call in [1, 2]:
        with np.load(ROOT/'runs'/HP/f'head-operands-call{call}.npz') as hp:
            if hp['vocabulary_rows'].tolist() != manifest['vocabulary_rows'] or hp['hidden_columns'].tolist() != manifest['hidden_columns']:
                raise ValueError('Coordinate mismatch')
            for module in manifest['modules']:
                name = module['name']
                comparisons.append({'hp_call': call, 'mf_module': name,
                                    'x': stats(hp['x'], mf_arrays[name, 'x']),
                                    'dy_matmul': stats(hp['dy'], mf_arrays[name, 'dy_matmul']),
                                    'dy_output': stats(hp['dy'], mf_arrays[name, 'dy_output'])})
    report = {'complete': all(checks.values()), 'mf_run': MF, 'hp_run': HP, 'step': 201,
              'mf_observer_fidelity': checks, 'mf_manifest': manifest, 'comparisons': comparisons,
              'scope': 'Selected columns only; mapping is based on actual activation equality, not assumed autograd order.'}
    (ROOT/'evidence/head-selected-operands-step201-comparison.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'complete': report['complete'], 'checks': checks, 'comparisons': comparisons}))


if __name__ == '__main__':
    main()
