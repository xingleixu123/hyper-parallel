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

"""Verify complete retained and reconstructed head moment hashes before redundant-array cleanup."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path('/alignment/phase12')
STATE = Path('/state/jt-recipe-20261014')


def main() -> None:
    """Stream both exact moment states from native arrays and their bounded sparse patches."""
    report = json.loads((ROOT / 'evidence/head-moments206-comparison.json').read_text())
    checks, actual = {}, {}
    for kind in ['exp_avg', 'exp_avg_sq']:
        row = report['moments'][kind]
        patch = Path(row['patch_file'])
        current = {'patch_sha256': hashlib.sha256(patch.read_bytes()).hexdigest() == row['patch_sha256']}
        with np.load(patch) as archive:
            indices, hp, mf = (archive[key].copy() for key in ['indices', 'hp', 'mf'])
        original = np.load(STATE / report['runs']['mf'] / 'moments206' / (kind + '.npy'), mmap_mode='r')
        flat = original.reshape(-1)
        current['shape_dtype'] = list(original.shape) == [152064, 5120] and original.dtype == np.float32
        current['count'] = indices.size == row['different'] and flat.size == row['size']
        current['indices'] = bool(np.all(indices[1:] > indices[:-1]) and indices.min() >= 0
                                  and indices.max() < flat.size)
        current['native_values_exact'] = bool(np.array_equal(flat[indices], mf) and np.all(mf != hp))
        if not all(current.values()):
            raise ValueError(current)
        hashes = {side: hashlib.sha256() for side in ['mf', 'hp']}
        for start in range(0, flat.size, 1_000_000):
            stop = min(start + 1_000_000, flat.size)
            value = np.asarray(flat[start:stop]).copy()
            lower, upper = np.searchsorted(indices, [start, stop])
            hashes['mf'].update(memoryview(value).cast('B'))
            value[indices[lower:upper] - start] = hp[lower:upper]
            hashes['hp'].update(memoryview(value).cast('B'))
        actual[kind] = {side: sha.hexdigest() for side, sha in hashes.items()}
        for side, sha in actual[kind].items():
            expected = next(item['sha256'] for item in report['metadata'][side]['records'] if item['kind'] == kind)
            current[side + '_full_hash'] = sha == expected
        checks[kind] = current
    if not all(all(row.values()) for row in checks.values()):
        raise ValueError(checks)
    result = {'complete': True, 'checks': checks, 'hashes': actual, 'full_hp_arrays_recreated': False,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (ROOT / 'evidence/head-moments206-compact-reconstruction.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
