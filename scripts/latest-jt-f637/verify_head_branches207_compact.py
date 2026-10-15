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

"""Retain exact compact branch-dweight patches and verify all reconstructed BF16 bits."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path('/alignment/phase12')
STATE = Path('/state/jt-recipe-20261014')


def main() -> None:
    """Keep the native dumps and prove redundant HP dweight arrays can be rebuilt exactly."""
    report = json.loads((ROOT / 'evidence/head-moments207-branches-comparison.json').read_text())
    if not report['complete'] or not all(all(row.values()) for row in report['fidelity'].values()):
        raise ValueError('Branch observations are not audited')
    rows = []
    for pair in report['comparisons']:
        number, call = str(pair['native_kernel']), str(pair['hp_forward_call'])
        native, hp = report['native'][number]['dw'], report['hp'][call]['dw']
        mf_array = np.load(native['path'], mmap_mode='r').reshape(-1)
        hp_array = np.load(hp['path'], mmap_mode='r').reshape(-1)
        path = Path(pair['comparison']['dw']['coordinate_file'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != pair['comparison']['dw']['coordinate_sha256']:
            raise ValueError('Difference coordinates changed')
        indices = np.load(path)
        replacement = hp_array[indices].copy()
        sha = hashlib.sha256()
        for start in range(0, mf_array.size, 1_000_000):
            stop = min(start + 1_000_000, mf_array.size)
            bits = mf_array[start:stop].view(np.uint32)
            if np.any(bits & np.uint32(0xffff)):
                raise ValueError('Native dump is not exact BF16')
            values = (bits >> np.uint32(16)).astype(np.uint16)
            low, high = np.searchsorted(indices, [start, stop])
            values[indices[low:high] - start] = replacement[low:high]
            sha.update(memoryview(values).cast('B'))
        if sha.hexdigest() != hp['bf16_sha256']:
            raise ValueError('Reconstructed HP branch differs')
        target = STATE / 'head-branches207-differences-r2' / ('op' + number + '-hp-patch.npz')
        if target.exists():
            raise FileExistsError(target)
        np.savez(target, indices=indices, hp=replacement)
        rows.append({'native_kernel': int(number), 'hp_call': int(call), 'count': int(indices.size),
                     'native_array': native['path'], 'redundant_hp_array': hp['path'],
                     'native_bf16_sha256': native['bf16_sha256'], 'hp_bf16_sha256': sha.hexdigest(),
                     'patch': str(target), 'patch_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
                     'patch_bytes': target.stat().st_size, 'reconstruction_exact': True})
    output = {'complete': True, 'branches': rows,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (ROOT / 'evidence/head-moments207-branch-compact-reconstruction.json').write_text(json.dumps(output, indent=2))
    print(json.dumps(output))


if __name__ == '__main__':
    main()
