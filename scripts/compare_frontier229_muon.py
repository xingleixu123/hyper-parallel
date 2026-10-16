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

"""Compare NS boundaries only after native and observed replays match training states."""
import hashlib
import json
from pathlib import Path

import numpy as np
from compare_preclip217 import compare_arrays

ROOT = Path('/alignment/phase12')
STATE = Path('/state/jt-recipe-20261014')


def native_array(directory: Path, prefix: str) -> tuple:
    """Require one complete native dump for the requested boundary."""
    matches = list(directory.glob(prefix + '_*.npy'))
    if len(matches) != 1:
        raise ValueError('Expected one native array: ' + prefix)
    array = np.load(matches[0], mmap_mode='r')
    if array.dtype != np.float32 or not np.isfinite(array).all():
        raise ValueError('Unexpected native array')
    return array, {'file': matches[0].name, 'shape': list(array.shape),
                   'sha256': hashlib.sha256(array.tobytes()).hexdigest()}


def main() -> None:
    """Keep diagnostic replays separate from independent training acceptance."""
    hp = json.loads((ROOT / 'evidence/frontier229-hp-muon-replay.json').read_text())
    mf = json.loads((ROOT / 'evidence/frontier229-mf-muon-replay.json').read_text())
    checks = {'hp_complete': hp['complete'], 'mf_complete': mf['complete'],
              'hp_training_fidelity': all(c['weight_exact_training'] and c['momentum_exact_training'] for c in hp['checks']),
              'hp_explicit_normalization_exact': all(c['explicit_normalization_exact'] for c in hp['records']),
              'mf_training_fidelity': all(c[k + '_weight_exact_training'] and c[k + '_momentum_exact_training']
                  for c in mf['records'] for k in ['native', 'observed'])}
    for family, report in [('hp', hp), ('mf', mf)]:
        folder = 'tools-f637' if family == 'hp' else 'tools'
        source = ROOT / folder / ('replay_' + family + '_frontier229_muon.py')
        checks[family + '_source'] = report['source_sha256'] == hashlib.sha256(source.read_bytes()).hexdigest()
        checks[family + '_exit_zero'] = (ROOT / 'runs' / ('frontier229-' + family + '-muon-replay.exit')).read_text().strip() == '0'
    if not all(checks.values()):
        raise ValueError(checks)
    rows, native_records = [], []
    mf_root = STATE / 'mf-frontier229-muon-replay'
    for piece in [0, 1]:
        record = next(row for row in hp['records'] if row['piece'] == piece)
        hp_root = STATE / 'hp-frontier229-muon-replay' / ('mtp_q_up-' + str(piece))
        bounds = {}
        for key in ['pre_norm', 'normalized', 'ns_output']:
            left = np.load(hp_root / (key + '.npy'), mmap_mode='r')
            source = next(row for row in record['arrays'] if row['name'] == key)
            if hashlib.sha256(left.tobytes()).hexdigest() != source['sha256']:
                raise ValueError('HP boundary changed')
            right, native = native_array(mf_root, 'mtp_q_up-' + str(piece) + '-' + key)
            native_records.append(native)
            bounds[key] = compare_arrays(left, right)
        scalars = {}
        for key in ['norm', 'denominator']:
            right, native = native_array(mf_root, 'mtp_q_up-' + str(piece) + '-' + key)
            native_records.append(native)
            scalars[key] = {'hp': record['scalar'][key][0], 'mf': float(right.reshape(-1)[0])}
        norm64 = record['norm_fp64']
        rows.append({'piece': piece, 'label': 'nope' if piece == 0 else 'rope', 'boundaries': bounds,
                     'scalars': scalars, 'norm_fp64': norm64,
                     'norm_abs_error_vs_fp64': {family: abs(scalars['norm'][family] - norm64) for family in ['hp', 'mf']}})
    report = {'complete': True, 'diagnostic_only': True, 'eligible_for_acceptance': False,
              'fidelity': checks, 'records': rows, 'native_arrays': native_records,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'interpretation': 'Exact final weight/momentum replay permits attribution to the actual update; '
                                'local FP64 error does not establish a generic product defect.'}
    (ROOT / 'evidence/frontier229-muon-boundaries.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'fidelity': checks, 'records': rows}))


if __name__ == '__main__':
    main()
