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

"""Assess all differing actual207 head GEMM coordinates against FP64 dot-product references."""
import hashlib
import json
from pathlib import Path

import numpy as np

from compare_head_branches207 import canonical_sha, fp32

ROOT = Path('/alignment/phase12')
STATE = Path('/state/jt-recipe-20261014')


def main() -> None:
    """Compare both numerical results without presuming either framework is the ground truth."""
    report = json.loads((ROOT / 'evidence/head-moments207-branches-comparison.json').read_text())
    if not report['complete'] or not all(all(row.values()) for row in report['fidelity'].values()):
        raise ValueError('Branch observations are not audited')
    results = []
    for pair in report['comparisons']:
        if pair['comparison']['x']['different'] or pair['comparison']['dy']['different']:
            raise ValueError('Same-input correctness comparison requires identical operands')
        number, call = str(pair['native_kernel']), str(pair['hp_forward_call'])
        native, hp = report['native'][number], report['hp'][call]
        arrays = {kind: np.load(native[kind]['path'], mmap_mode='r') for kind in ['x', 'dy', 'dw']}
        actual = np.load(hp['dw']['path'], mmap_mode='r')
        for kind, array in arrays.items():
            if canonical_sha(array) != native[kind]['bf16_sha256']:
                raise ValueError('Native operand or output changed')
        if canonical_sha(actual) != hp['dw']['bf16_sha256']:
            raise ValueError('HP output changed')
        coordinates = Path(pair['comparison']['dw']['coordinate_file'])
        if hashlib.sha256(coordinates.read_bytes()).hexdigest() != pair['comparison']['dw']['coordinate_sha256']:
            raise ValueError('Coordinate set changed')
        indices = np.load(coordinates)
        ideals = np.empty(indices.size, dtype=np.float64)
        mf_values = fp32(arrays['dw'].reshape(-1)[indices]).astype(np.float64)
        hp_values = fp32(actual.reshape(-1)[indices]).astype(np.float64)
        for start in range(0, indices.size, 128):
            selected = indices[start:start + 128]
            vocab, hidden = selected // 5120, selected % 5120
            dy = fp32(arrays['dy'][:, vocab]).astype(np.float64)
            inputs = fp32(arrays['x'][:, hidden]).astype(np.float64)
            ideals[start:start + selected.size] = np.sum(dy * inputs, axis=0, dtype=np.float64)
        mf_error, hp_error = np.abs(mf_values - ideals), np.abs(hp_values - ideals)
        target = STATE / 'head-branches207-differences' / ('op' + number + '-fp64.npz')
        if target.exists():
            raise FileExistsError(target)
        np.savez(target, indices=indices, reference=ideals, mf=mf_values, hp=hp_values)
        row = {'native_kernel': int(number), 'hp_call': int(call), 'different_coordinates': int(indices.size),
               'mf_closer': int(np.count_nonzero(mf_error < hp_error)),
               'hp_closer': int(np.count_nonzero(hp_error < mf_error)),
               'equal_distance': int(np.count_nonzero(mf_error == hp_error)),
               'max_mf_error': float(mf_error.max(initial=0)), 'max_hp_error': float(hp_error.max(initial=0)),
               'mean_mf_error': float(mf_error.mean()) if indices.size else 0,
               'mean_hp_error': float(hp_error.mean()) if indices.size else 0,
               'local_evidence': str(target), 'local_evidence_sha256': hashlib.sha256(target.read_bytes()).hexdigest()}
        results.append(row)
        print('[P12-HEAD207-FP64]', json.dumps(row), flush=True)
    output = {'complete': True, 'comparisons': results,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'reference': 'FP64 multiplication and summation over4096 exact BF16 operand values at every differing coordinate',
              'scope': 'All differing207 branch dweight coordinates, not every unchanged coordinate, not a real-arithmetic proof '
                       'or a universal framework accuracy ranking. No training values injected.'}
    (ROOT / 'evidence/head-moments207-fp64-correctness.json').write_text(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
