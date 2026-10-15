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

"""Audit native full-model Q norm reduction schedules from emitted DVM tiling."""
import json
from pathlib import Path
import numpy as np


def half_tree(work):
    while work.shape[-1] > 1:
        size = work.shape[-1]
        half = size // 2
        result = work[..., :half] + work[..., half:2 * half]
        if size % 2:
            result[..., 0] += work[..., -1]
        work = result
    return work[..., 0]


def seq(values):
    result = np.float32(0)
    for value in values:
        result = np.float32(result + value)
    return result


def main():
    root = Path('/state/jt-recipe-20261014/mf-native-q-normalization201-kernels')
    rows = []
    for number in [1, 12, 25]:
        prefix = '*split-op' + str(number) + '.*'
        value = np.load(next(root.rglob(prefix + '.input.1.*.npy')))
        gold = np.load(next(root.rglob(prefix + '.input.0.*.npy')))
        padded = np.zeros(1127 * 6984, dtype=np.float32)
        padded[:value.size] = (value * value).reshape(-1)
        tiles = padded.reshape(1127, 291, 24).transpose(0, 2, 1)
        partials = half_tree(half_tree(tiles))
        candidates = {'seq': seq(partials)}
        for cores in [20, 40]:
            for kind in ['strided', 'contiguous']:
                splits = [partials[i::cores] for i in range(cores)] if kind == 'strided' else np.array_split(partials, cores)
                sums = np.array([seq(v) for v in splits], dtype=np.float32)
                candidates[str(cores) + '_' + kind + '_seq'] = seq(sums)
                candidates[str(cores) + '_' + kind + '_tree'] = half_tree(sums)
        rows.append({'kernel': number, 'gold': float(gold), 'candidates': [
            {'schedule': key, 'sum': float(v), 'ulp_error': int(v.view(np.uint32)) - int(gold.view(np.uint32)),
             'denominator': float(np.float32(np.sqrt(v) + np.float32(1.0011717677116394e-7)))}
            for key, v in candidates.items()]})
    report = {'complete': True, 'tile_elements': 6984, 'simd_width': 24, 'tile_count': 1127,
              'source': 'native-q-reduction-dvm-op1/op4/op8.txt', 'records': rows}
    Path('/alignment/phase12/evidence/native-q-reduction-schedules.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
