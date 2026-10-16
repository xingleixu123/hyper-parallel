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

"""Validate the five-core expert schedule inferred from official DVM VisitRed_1."""
import hashlib
import json
from pathlib import Path

import numpy as np

from compare_frontier219_state import load
from compare_preclip217 import bf16_bits
from mindspore272_stat_hash import native_dump_sha1

ROOT = Path('/alignment/phase12')
STATE = Path('/state/jt-recipe-20261014')


def half_tree(value: np.ndarray) -> np.ndarray:
    """Use the explicit folded half-tree candidate without changing FP32 arithmetic."""
    while value.shape[-1] > 1:
        size = value.shape[-1]
        half = size // 2
        reduced = value[..., :half] + value[..., half:2 * half]
        if size % 2:
            reduced[..., 0] = reduced[..., 0] + value[..., -1]
        value = reduced
    return value[..., 0]


def adjacent(value: np.ndarray) -> np.ndarray:
    """Reduce power-of-two lane counts by adjacent pairs."""
    while value.shape[-1] > 1:
        value = value[..., ::2] + value[..., 1::2]
    return value[..., 0]


def examine(step: int) -> dict:
    """Require an exact native input hash before examining alternative reduction schedules."""
    run = 'hp-jte213-fc1norm-qupboth-state' + str(step)
    prefix = 'frontier' + str(step)
    replay = json.loads((ROOT / 'evidence' / (prefix + '-hp-control-replay.json')).read_text())
    if not replay['complete'] or not all(row['weight_exact_training'] and row['momentum_exact_training']
                                         for row in replay['checks']):
        raise ValueError('Control replay fidelity is not established')
    report = json.loads((ROOT / 'runs' / run / ('FRONTIER_STATE-' + str(step) + '.json')).read_text())
    momentum = load(STATE / run / ('state-' + str(step)), next(row for row in report['records'] if row['kind'] == 'muon_m'))
    gradient_report = json.loads((ROOT / 'runs' / run / ('FRONTIER_GRADIENT-' + str(step) + '.json')).read_text())
    gradient = load(STATE / run / ('gradient-' + str(step)), next(row for row in gradient_report['records']
                                                       if row['kind'] == 'postclip'))
    gradient = gradient.reshape(8, 1536, 5120).transpose(0, 2, 1)
    represented = bf16_bits(momentum * np.float32(.95) + gradient)
    represented = np.ascontiguousarray(represented.transpose(0, 2, 1))
    input_hash = native_dump_sha1(represented)
    if input_hash != replay['records'][0]['pre_norm']['native_dump_sha1']:
        raise ValueError('CPU reconstruction does not match the actual native NS input')
    raw = (represented.astype(np.uint32) << 16).view(np.float32)
    squares = np.square(raw).reshape(8, 768, 10240)
    native = json.loads((ROOT / 'evidence' / (prefix + '-native-normalization-scalars-values.json')).read_text())
    record = next(row for row in native['records'] if '.input.0.' in row['file'])
    expected = np.asarray(record['values'], dtype=np.float32)
    if hashlib.sha1(expected.tobytes()).hexdigest() != record['sha1']:
        raise ValueError('Native scalar serialization changed')
    results = []
    for lanes in [64]:
        shape = squares.reshape(8, 768, 10240 // lanes, lanes).transpose(0, 1, 3, 2)
        partial = half_tree(shape)
        for lane_order, reducer in [('adjacent', adjacent), ('half', half_tree)]:
            tiles = reducer(partial)
            for join in ['cores2', 'cores5']:
                if join == 'sequential':
                    total = np.zeros(8, dtype=np.float32)
                    for index in range(768):
                        total = total + tiles[:, index]
                elif join == 'half':
                    total = half_tree(tiles)
                elif join == 'numpy':
                    total = np.sum(tiles, axis=1, dtype=np.float32)
                else:
                    cores = int(join[5:])
                    groups = np.zeros((8, cores), dtype=np.float32)
                    for index in range(768):
                        groups[:, index % cores] = groups[:, index % cores] + tiles[:, index]
                    total = np.zeros(8, dtype=np.float32)
                    for index in range(cores):
                        total = total + groups[:, index]
                exact = total.view(np.uint32) == expected.view(np.uint32)
                results.append({'lanes': lanes, 'lane_order': lane_order, 'join': join,
                                'exact_experts': np.flatnonzero(exact).tolist(), 'sums': total.tolist(),
                                'max_abs_error': float(np.max(np.abs(total - expected))),
                                'all_exact': bool(np.all(exact))})
    return {'step': step, 'input_native_hash_exact': True,
            'input_sha1': input_hash, 'native_sums_sha1': record['sha1'],
            'native_sums': expected.tolist(), 'results': results}


def main() -> None:
    """Check both the first-divergence update and an independent neighboring update."""
    records = [examine(step) for step in [328, 329]]
    report = {
        'complete': True, 'diagnostic_only': True, 'records': records,
        'dvm_tile_elements': 10240, 'simd_width': 64, 'tiles_per_expert': 768,
        'expected_virtual_vector_cores': 40, 'experts': 8, 'expected_cores_per_expert': 5,
        'official_reference_commit': 'a36395c4f66cbdbb8a9104075af611d9e5f230de',
        'official_reference_url': 'https://gitcode.com/mindspore/dvm/blob/a36395c4f66cbdbb8a9104075af611d9e5f230de/src/vm_aiv.cce',
        'source_evidence': 'ReduceSumX rows698-735; ReduceJoin rows780-874; VisitRed_1 rows1690-1734.',
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'limitation': 'Official r2.7 source is a pinned reference, not a verified installed-binary revision. '
                      'Measured native input/output hashes are required to validate the inferred schedule.',
    }
    (ROOT / 'evidence/frontier328-329-dvm-schedule-screen.json').write_text(json.dumps(report, indent=2))
    print(json.dumps([{'step': row['step'], 'results': row['results']} for row in records]))


if __name__ == '__main__':
    main()
