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

"""Compare both normalization results with an identical-formula FP64 reference."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path('/alignment/phase12')
STATE = Path('/state/jt-recipe-20261014')
EPSILON = 1.0011717677116394e-7


def main() -> None:
    """Assess numerical accuracy without changing archive or training semantics."""
    records = []
    for piece in [0, 1]:
        directory = STATE / 'hp-frontier258-muon-replay' / ('main1_q_up-' + str(piece))
        values = np.load(directory / 'pre_norm.npy', mmap_mode='r')
        hp = np.load(directory / 'normalized.npy', mmap_mode='r')
        paths = list((STATE / 'mf-frontier258-muon-replay').glob(
            'main1_q_up-' + str(piece) + '-normalized_*.npy'))
        if len(paths) != 1:
            raise ValueError('Expected one MF normalized array')
        mf = np.load(paths[0], mmap_mode='r')
        total64 = np.square(values, dtype=np.float64).sum(dtype=np.float64)
        norm64 = float(np.sqrt(total64))
        denominator = norm64 + EPSILON
        counts = {'different': 0, 'hp_closer_on_differences': 0,
                  'mf_closer_on_differences': 0, 'ties_on_differences': 0}
        error = {'hp': {'max': 0., 'sum': 0.}, 'mf': {'max': 0., 'sum': 0.}}
        for start in range(0, values.size, 1048576):
            raw = values.reshape(-1)[start:start + 1048576].astype(np.float64)
            first = hp.reshape(-1)[start:start + 1048576].astype(np.float64)
            second = mf.reshape(-1)[start:start + 1048576].astype(np.float64)
            reference = raw / denominator
            hp_error, mf_error = np.abs(first - reference), np.abs(second - reference)
            different = first != second
            counts['different'] += int(np.count_nonzero(different))
            counts['hp_closer_on_differences'] += int(np.count_nonzero(different & (hp_error < mf_error)))
            counts['mf_closer_on_differences'] += int(np.count_nonzero(different & (mf_error < hp_error)))
            counts['ties_on_differences'] += int(np.count_nonzero(different & (hp_error == mf_error)))
            for label, data in [('hp', hp_error), ('mf', mf_error)]:
                error[label]['max'] = max(error[label]['max'], float(data.max()))
                error[label]['sum'] += float(data.sum())
        for data in error.values():
            data['mean'] = data.pop('sum') / values.size
        records.append({'piece': piece, 'elements': int(values.size), 'fp64_square_sum': float(total64),
                        'fp64_norm': norm64, 'epsilon': EPSILON, 'counts': counts, 'absolute_error': error})
    report = {'complete': True, 'records': records,
              'reference': 'Same BF16-representable inputs, FP64 sum/sqrt/division and the actual archived epsilon.',
              'claim_limit': 'Numerical proximity for these inputs only; does not prove end-to-end model quality.',
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (ROOT / 'evidence/frontier258-normalization-precision.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == '__main__':
    main()
