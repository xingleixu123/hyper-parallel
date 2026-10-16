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

"""Compare actual represented expert normalization with a fixed-input FP64 reference."""
import hashlib
import json
from pathlib import Path

import numpy as np

from compare_frontier219_state import load
from compare_preclip217 import bf16_bits
from mindspore272_stat_hash import native_dump_sha1

ROOT = Path('/alignment/phase12')
STATE = Path('/state/jt-recipe-20261014')
RUN = 'hp-jte213-fc1norm-qupboth-state328'


def main() -> None:
    """Require all native hashes before scoring normalized values; never persist large arrays."""
    audit = json.loads((ROOT / 'evidence/frontier328-native-cause-audit.json').read_text())
    if not audit['complete'] or not all(all(row['fidelity'].values()) for row in audit['records']):
        raise ValueError('Native first-update causal audit failed')
    replay = json.loads((ROOT / 'evidence/frontier328-hp-expert-native-replay.json').read_text())
    records = {row['mode']: row for row in replay['records']}
    report = json.loads((ROOT / 'runs' / RUN / 'FRONTIER_STATE-328.json').read_text())
    momentum = load(STATE / RUN / 'state-328', next(row for row in report['records'] if row['kind'] == 'muon_m'))
    report = json.loads((ROOT / 'runs' / RUN / 'FRONTIER_GRADIENT-328.json').read_text())
    gradient = load(STATE / RUN / 'gradient-328', next(row for row in report['records'] if row['kind'] == 'postclip'))
    gradient = gradient.reshape(8, 1536, 5120).transpose(0, 2, 1)
    bits = bf16_bits(momentum * np.float32(.95) + gradient)
    bits = np.ascontiguousarray(bits.transpose(0, 2, 1))
    if native_dump_sha1(bits) != records['control']['pre_norm']['native_dump_sha1']:
        raise ValueError('Reconstructed input does not match the native NS boundary')
    del momentum, gradient
    raw = (bits.astype(np.uint32) << 16).view(np.float32)
    outputs, hashes = {}, {}
    for key in ['control', 'dvm_tree']:
        denominator = np.asarray(records[key]['denominator_bf16'], dtype=np.float32).reshape(8, 1, 1)
        normalized = bf16_bits(raw / denominator)
        hashes[key] = native_dump_sha1(normalized) == records[key]['normalized']['native_dump_sha1']
        outputs[key] = normalized
    if not all(hashes.values()):
        raise ValueError({'normalized_native_hashes': hashes})
    results = []
    for index in range(8):
        matrix = raw[index].astype(np.float64)
        norm = float(np.sqrt(np.square(matrix).sum(dtype=np.float64)))
        reference = matrix / (norm + 1e-7)
        hp = (outputs['control'][index].astype(np.uint32) << 16).view(np.float32).astype(np.float64)
        mf = (outputs['dvm_tree'][index].astype(np.uint32) << 16).view(np.float32).astype(np.float64)
        hp_error = np.abs(hp - reference)
        mf_error = np.abs(mf - reference)
        different = outputs['control'][index] != outputs['dvm_tree'][index]
        row = {
            'expert': index, 'elements': int(matrix.size), 'different_bf16': int(different.sum()),
            'hp_closer_on_different': int(np.sum((hp_error < mf_error) & different)),
            'mf_closer_on_different': int(np.sum((mf_error < hp_error) & different)),
            'equal_error_on_different': int(np.sum((hp_error == mf_error) & different)),
            'hp_mean_abs_error': float(hp_error.mean()), 'mf_mean_abs_error': float(mf_error.mean()),
            'norm_fp64': norm, 'ideal_denominator_fp64': norm + 1e-7,
            'hp_norm_abs_error': abs(records['control']['norm_fp32'][index] - norm),
            'mf_compatible_norm_abs_error': abs(records['dvm_tree']['candidate_stats']['norm'][index] - norm),
            'hp_denominator_abs_error': abs(records['control']['denominator_bf16'][index] - (norm + 1e-7)),
            'mf_denominator_abs_error': abs(records['dvm_tree']['denominator_bf16'][index] - (norm + 1e-7)),
        }
        results.append(row)
    result = {
        'complete': True, 'native_input_exact': True, 'native_normalized_hashes_exact': hashes,
        'records': results, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'reference': 'Same actual represented BF16 input, FP64 Frobenius norm and division with mathematical epsilon1e-7.',
        'scope': 'Only normalization accuracy; this does not classify the entire optimizer or either framework as universally correct.',
    }
    (ROOT / 'evidence/frontier328-normalization-precision.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
