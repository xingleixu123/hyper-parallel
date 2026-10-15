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

"""Compare exact activation operands and high-precision SwiGLU up derivatives."""
from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path

import numpy as np
from compare_vocab_optimizer_trace import fidelity, read_rows
from compare_head_selected_operands import stats

ROOT = Path('/alignment/phase12')
STATE = Path('/state/jt-recipe-20261014')
MF = 'mf-sharedactrow205206'
HP = 'hp-jtf637-adamw-all-sharedactrow206'


def load_array(rows, folder, kind):
    """Load an immutable recorded operand."""
    row = next(row for row in rows if row['kind'] == kind)
    value = np.load(folder / row['file'], mmap_mode='r')
    if hashlib.sha256(value.tobytes()).hexdigest() != row['sha256']:
        raise ValueError('Operand changed')
    return value


def main():
    """Validate fidelity before comparing the actual affected token."""
    checks = {}
    for label, run, baseline in [('mf', MF, 'mf-resume'), ('hp', HP, 'hp-jtf637-adamw-all-archive206')]:
        checks[label + '_exit_zero'] = (ROOT / 'runs' / (run + '.exit')).read_text().strip() == '0'
        checks.update({label + '_' + key: value for key, value in fidelity(run, baseline, 6).items()})
    checks['hp_optimizer_trace_exact'] = (ROOT / 'runs' / HP / 'VOCAB_OPTIMIZER_TRACE.jsonl').read_bytes() == (
        ROOT / 'runs/hp-jtf637-adamw-all-archive206/VOCAB_OPTIMIZER_TRACE.jsonl').read_bytes()
    mm = json.loads((STATE / MF / 'deep/manifest.json').read_text())
    hm = json.loads((ROOT / 'runs' / HP / 'SHARED_ACTIVATION_OBSERVER.json').read_text())
    for label, metadata, source in [('mf', mm, 'tools/mf_shared_activation_row_tap.py'),
                                    ('hp', hm, 'tools-f637/hp_shared_activation_row_factor.py')]:
        checks[label + '_source_exact'] = metadata['source_sha256'] == hashlib.sha256((ROOT / source).read_bytes()).hexdigest()
    if not all(checks.values()):
        raise ValueError(checks)
    mr = next(row for row in read_rows(ROOT / 'runs' / MF / 'SHARED_ACTIVATION_ROW.jsonl') if row['step'] == 205)['records']
    hr = json.loads((ROOT / 'runs' / HP / 'SHARED_ACTIVATION_ROW.json').read_text())['records']
    arrays = {}
    for label, rows, folder in [('mf', mr, STATE / MF / 'deep'), ('hp', hr, STATE / HP / 'activation_row')]:
        for kind in ['x', 'dx']:
            arrays[label, kind] = load_array(rows, folder, kind)
    comparisons = {kind: stats(arrays['hp', kind], arrays['mf', kind]) for kind in ['x', 'dx']}
    fc1_mf = 'mf-sharedfc1-205206'
    fc1_hp = 'hp-jtf637-adamw-all-sharedfc1206'
    mr = next(row for row in read_rows(ROOT / 'runs' / fc1_mf / 'SHARED_FC1_BOUNDARIES.jsonl') if row['step'] == 205)['records']
    hr = json.loads((ROOT / 'runs' / fc1_hp / 'SHARED_FC1_BOUNDARIES.json').read_text())['records']
    projected = load_array(hr, STATE / fc1_hp / 'shared_fc1', 'y').reshape(4096, 6144)[1605]
    actual_hp = load_array(hr, STATE / fc1_hp / 'shared_fc1', 'dy').reshape(4096, 6144)[1605]
    actual_mf = load_array(mr, STATE / fc1_mf / 'deep', 'dy').reshape(4096, 6144)[1605]
    counts = {'hp_closer': 0, 'mf_closer': 0, 'tie': 0}
    examples = []
    if not all(row['exact'] for row in comparisons.values()):
        raise ValueError('Upstream activation operands differ; cannot compare derivative accuracy')
    with localcontext() as context:
        context.prec = 90
        for index in np.flatnonzero(actual_hp != actual_mf):
            if index % 2 != 1:
                raise ValueError('Expected up-lane derivative difference')
            channel = int(index // 2)
            gate = Decimal.from_float(float(projected[index - 1]))
            incoming = Decimal.from_float(float(arrays['hp', 'dx'][channel]))
            ideal = incoming * gate / (Decimal(1) + (-gate).exp())
            hp_error = abs(Decimal.from_float(float(actual_hp[index])) - ideal)
            mf_error = abs(Decimal.from_float(float(actual_mf[index])) - ideal)
            winner = 'hp_closer' if hp_error < mf_error else 'mf_closer' if mf_error < hp_error else 'tie'
            counts[winner] += 1
            examples.append({'channel': channel, 'gate': str(gate), 'incoming': str(incoming),
                             'ideal': str(ideal), 'hp': float(actual_hp[index]), 'mf': float(actual_mf[index]),
                             'hp_error': str(hp_error), 'mf_error': str(mf_error), 'winner': winner})
    report = {'complete': True, 'fidelity': checks, 'comparisons': comparisons, 'accuracy': {'counts': counts,
              'examples': examples}, 'scope': 'Actual token1605 has identical activation output and incoming gradient. Decimal90 compares ideal incoming*SiLU(gate) at every differing up derivative in this row; not a global operator ranking.'}
    (ROOT / 'evidence/shared-activation-row205-comparison.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == '__main__':
    main()
