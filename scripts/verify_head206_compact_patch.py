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

"""Verify both complete head states from one retained array and a small difference patch."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path('/alignment/phase12')
STATE = Path('/state/jt-recipe-20261014')


def main():
    """Reconstruct HP hashes in bounded memory without recreating the deleted full array."""
    report = json.loads((ROOT / 'evidence/head206-model-comparison.json').read_text())
    patch = Path(report['patch_file'])
    checks = {'patch_sha256': hashlib.sha256(patch.read_bytes()).hexdigest() == report['patch_sha256']}
    with np.load(patch) as archive:
        indices, hp, mf = (archive[key].copy() for key in ['indices', 'hp', 'mf'])
    original = np.load(STATE / 'mf-head206-state-r2/head206.npy', mmap_mode='r')
    flat = original.reshape(-1)
    checks['shape_dtype'] = list(original.shape) == [152064, 5120] and original.dtype == np.float32
    checks['count'] = indices.size == report['different_fp32'] and flat.size == report['size']
    checks['indices'] = bool(np.all(indices[1:] > indices[:-1]) and indices.min() >= 0 and indices.max() < flat.size)
    checks['native_values_exact'] = bool(np.array_equal(flat[indices], mf) and np.all(mf != hp))
    if not all(checks.values()):
        raise ValueError(checks)
    hashes = {side: {kind: hashlib.sha256() for kind in ['fp32_sha256', 'bf16_as_fp32_sha256']}
              for side in ['mf', 'hp']}
    for start in range(0, flat.size, 1_000_000):
        stop = min(start + 1_000_000, flat.size)
        value = np.asarray(flat[start:stop]).copy()
        lower, upper = np.searchsorted(indices, [start, stop])
        for side in ['mf', 'hp']:
            if side == 'hp':
                value[indices[lower:upper] - start] = hp[lower:upper]
            hashes[side]['fp32_sha256'].update(memoryview(value).cast('B'))
            bits = value.view(np.uint32)
            rounded = (bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) & np.uint32(0xffff0000)
            hashes[side]['bf16_as_fp32_sha256'].update(memoryview(rounded).cast('B'))
    actual = {side: {key: digest.hexdigest() for key, digest in kinds.items()} for side, kinds in hashes.items()}
    for side, kinds in actual.items():
        for key, digest in kinds.items():
            checks[side + '_' + key] = digest == report['model_records'][side][key]
    if not all(checks.values()):
        raise ValueError(checks)
    result = {'complete': True, 'checks': checks, 'hashes': actual,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'full_hp_array_recreated': False,
              'scope': 'Both full FP32/BF16 head hashes verified from retained MF head plus compact patch after redundant HP array cleanup.'}
    (ROOT / 'evidence/head206-compact-reconstruction.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
