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

"""Compare native framework Dump with faithful HP and source-observer replays."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path('/alignment/phase12')
STATE = Path('/state/jt-recipe-20261014')
RUN = 'frontier229-mf-builtin-tensor-dump'


def digest(array: np.ndarray) -> str:
    """Hash contiguous tensor values independently of NPY headers."""
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def compare(left: np.ndarray, right: np.ndarray) -> dict:
    """Compare complete FP32 representations, including signed-zero bits."""
    if left.shape != right.shape or left.dtype != np.float32 or right.dtype != np.float32:
        raise ValueError('Unexpected comparison shape or dtype')
    different = int(np.count_nonzero(left.view(np.uint32) != right.view(np.uint32)))
    return {'size': int(left.size), 'bitwise_different': different,
            'max_abs': float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))}


def unique(pattern: str) -> Path:
    """Require exactly one matching original native dump tensor."""
    paths = list((STATE / RUN).rglob(pattern))
    if len(paths) != 1:
        raise ValueError('Unexpected dump matches: ' + pattern)
    return paths[0]


def main() -> None:
    """Require native-state fidelity before interpreting selected kernel boundaries."""
    native = json.loads((ROOT / 'evidence' / (RUN + '.json')).read_text())
    hp = json.loads((ROOT / 'evidence/frontier229-hp-epsilon-replay.json').read_text())
    checks = {
        'native_complete': native['complete'],
        'native_state_fidelity': all(row['exact_actual_training'] for row in native['native_state_fidelity'].values()),
        'native_no_source_taps': not native['source_tensor_dump_inserted'],
        'native_exit_zero': (ROOT / 'runs' / (RUN + '.exit')).read_text().strip() == '0',
        'epsilon_update_fidelity': all(row['weight_exact_mf_training'] and row['momentum_exact_training']
                                       for row in hp['checks']),
        'epsilon_explicit_normalization_fidelity': all(row['explicit_normalization_exact'] for row in hp['records']),
        'six_tensors_only': len(list((STATE / RUN).rglob('*.npy'))) == 6,
        'native_version_272': native['mindspore_version'] == '2.7.2',
    }
    if not all(checks.values()):
        raise ValueError(checks)
    rows, arrays = [], []
    for piece in [0, 1]:
        paths = {key: unique('*-op' + str(piece) + '.*.' + slot + '.DefaultFormat.*.npy')
                 for key, slot in [('square_sum', 'input.0'), ('pre_norm', 'input.1'), ('normalized', 'output.0')]}
        values = {key: np.load(path, mmap_mode='r') for key, path in paths.items()}
        for key, path in paths.items():
            arrays.append({'piece': piece, 'boundary': key, 'path': str(path.relative_to(STATE / RUN)),
                           'bytes': path.stat().st_size, 'shape': list(values[key].shape),
                           'dtype': str(values[key].dtype), 'value_sha256': digest(values[key]),
                           'file_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        record = {'piece': piece, 'square_sum_float32': float(values['square_sum']),
                  'square_sum_uint32': int(values['square_sum'].view(np.uint32)),
                  'host_sqrt_for_reference_only': float(np.sqrt(values['square_sum'])), 'comparisons': {}}
        for side, directory in [('hp_control', 'hp-frontier229-muon-replay'),
                                ('hp_epsilon', 'hp-frontier229-epsilon-replay')]:
            base = STATE / directory / ('mtp_q_up-' + str(piece))
            record['comparisons'][side] = {key: compare(values[key], np.load(base / (key + '.npy'), mmap_mode='r'))
                                           for key in ['pre_norm', 'normalized']}
        for key in ['pre_norm', 'normalized']:
            candidates = list((STATE / 'mf-frontier229-muon-replay').glob(
                'mtp_q_up-' + str(piece) + '-' + key + '_*.npy'))
            if len(candidates) != 1:
                raise ValueError('Missing faithful source observer array')
            record.setdefault('mf_source_observer', {})[key] = compare(values[key], np.load(candidates[0], mmap_mode='r'))
        rows.append(record)
    report = {'complete': True, 'diagnostic_only': True, 'fidelity': checks, 'records': rows, 'arrays': arrays,
              'total_tensor_bytes': sum(row['bytes'] for row in arrays),
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'scope': 'Original native fusion inputs and outputs only; sqrt and epsilon addition remain inside fusion. '
                       'The source observer scalar casts do not expose those hidden FP32 intermediates.'}
    (ROOT / 'evidence/frontier229-builtin-boundaries.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'fidelity': checks, 'records': rows, 'tensor_bytes': report['total_tensor_bytes']}))


if __name__ == '__main__':
    main()
