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

"""Audit native backward226 boundaries, comparing all elements rather than samples."""
import hashlib
import json
from pathlib import Path

import numpy as np

from compare_preclip212 import compare_arrays
from compare_head206_interventions import source_sha
from compare_vocab_optimizer_trace import fidelity, read_rows

ROOT = Path('/alignment/phase12')
STATE = Path('/state/jt-recipe-20261014')
HP = 'hp-jtf637-qsqrt222-backward226'
MF = 'mf-backward226'
BASE = 'hp-jtf637-qsqrt222-both226'


def pairs():
    """Return explicit final-norm and main-MoE branch pairs."""
    result = []
    for left, right in [('model.decoder.final_layernorm', 'model.norm'),
                        ('model.mtp.layers.0.final_layernorm', 'mtp.layers.0.final_layernorm'),
                        ('model.decoder.layers.1.mlp', 'model.layers.1.mlp')]:
        kinds = ['x', 'y_fp32', 'dy', 'dx'] if left.endswith('.mlp') else [
            'x', 'weight', 'y_fp32', 'inverse_rms', 'dy', 'dx']
        result.extend((left, right, kind) for kind in kinds)
    return result


def main() -> None:
    """Require native trajectory fidelity before comparing retained observations."""
    checks = {'hp': fidelity(HP, BASE, 26), 'mf': fidelity(MF, 'mf-resume', 26)}
    for side, run in [('hp', HP), ('mf', MF)]:
        checks[side]['exit_zero'] = (ROOT / 'runs' / (run + '.exit')).read_text().strip() == '0'
    checks['hp']['config_exact_except_length_path'] = (ROOT / 'configs' / (BASE + '.yaml')).read_text().replace(
        BASE, HP).replace('train_iters: 226', 'train_iters: 226') == (ROOT / 'configs' / (HP + '.yaml')).read_text()
    checks['mf']['config_exact_except_path'] = (ROOT / 'configs/mf-preclip217.yaml').read_text().replace(
        'mf-preclip217', MF).replace('  stop_step: 17', '  stop_step: 26') == (ROOT / 'configs' / (MF + '.yaml')).read_text()
    for name in ['SOURCE_REVISION.json', 'PROBE_SOURCE_MANIFEST.json', 'HEAD_GEMM_ROUTE.json',
                 'SHARED_UP_FP32_FACTOR.json', 'ADAMW_ARCHIVE_ARITHMETIC.json', 'Q_NS_ADJACENT_FACTOR.json',
                 'ROUTER_NS_NPU_SQRT_FACTOR.json', 'WEIGHTED_DVM_FACTOR.json', 'PACKED_SEMANTICS_FACTOR.json', 'NORM219_FACTOR.json', 'Q_NPU_SQRT222_FACTOR.json']:
        checks['hp']['unchanged_' + name] = (ROOT / 'runs' / HP / name).read_bytes() == (ROOT / 'runs' / BASE / name).read_bytes()
    for name in ['HEAD_GEMM_ROUTE_CALLS.jsonl', 'Q_NS_ADJACENT_CALLS.jsonl', 'ROUTER_NS_NPU_SQRT_CALLS.jsonl',
                 'CLIP_REDUCTION.jsonl', 'WEIGHTED_DVM_CALLS.jsonl', 'PACKED_SEMANTICS_CALLS.jsonl', 'NORM219_CALLS.jsonl']:
        checks['hp']['exact26_' + name] = read_rows(ROOT / 'runs' / HP / name) == read_rows(ROOT / 'runs' / BASE / name)[:26]
    frozen = json.loads((ROOT / 'runs' / HP / 'PROBE_SOURCE_MANIFEST.json').read_text())
    checks['hp']['49_frozen_sources'] = len(frozen['files']) == 49 and all(
        row['sha256'] == source_sha(row['name']) for row in frozen['files'])
    hp_report = json.loads((ROOT / 'runs' / HP / 'BACKWARD226_BOUNDARIES.json').read_text())
    mf_rows = read_rows(ROOT / 'runs' / MF / 'BACKWARD226_BOUNDARIES.jsonl')
    mf_manifest = json.loads((STATE / MF / 'deep/manifest.json').read_text())
    hp_metadata = json.loads((ROOT / 'runs' / HP / 'BACKWARD226_OBSERVER.json').read_text())
    checks['hp']['observer_source'] = hp_metadata['source_sha256'] == source_sha('hp_backward226_boundaries.py')
    checks['mf']['observer_source'] = mf_manifest['source_sha256'] == hashlib.sha256((ROOT / 'tools/mf_backward226_tap.py').read_bytes()).hexdigest()
    checks['mf']['all26_observations'] = [r['step'] for r in mf_rows] == list(range(201, 227))
    checks['hp']['all16_boundaries'] = len(hp_report['records']) == 16
    checks['mf']['all16_boundaries'] = len({(r['name'], r['kind']) for r in mf_rows[-1]['records']}) == 16
    if not all(all(row.values()) for row in checks.values()):
        raise ValueError(checks)
    hp = {(r['name'], r['kind']): r for r in hp_report['records']}
    mf = {}
    for row in mf_rows[-1]['records']:
        key = row['name'], row['kind']
        if key in mf and row['sha256'] != mf[key]['sha256']:
            raise ValueError('Recomputed native forward boundary differs')
        mf[key] = row
    observations = []
    for mf_name, hp_name, kind in pairs():
        left_row, right_row = hp[hp_name, kind], mf[mf_name, kind]
        left = np.load(STATE / HP / 'norms' / left_row['file'], mmap_mode='r')
        right = np.load(STATE / MF / 'deep' / right_row['file'], mmap_mode='r')
        for array, row in [(left, left_row), (right, right_row)]:
            if list(array.shape) != row['shape'] or hashlib.sha256(array.tobytes()).hexdigest() != row['sha256']:
                raise ValueError('Retained forward array differs from native observation')
        row = {'mf_name': mf_name, 'hp_name': hp_name, 'kind': kind,
               'hp_shape': list(left.shape), 'mf_shape': list(right.shape), 'hp_dtype': left_row['dtype'],
               **compare_arrays(left.reshape(-1, left.shape[-1]), right.reshape(-1, right.shape[-1]))}
        observations.append(row)
        del left, right
    if len(observations) != 16:
        raise ValueError('Incomplete full-boundary mapping')
    report = {'complete': True, 'fidelity': checks, 'observations': observations,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'scope': 'All values of16 final-norm and main-MoE branch boundaries; BF16 round equivalence reported separately.'}
    (ROOT / 'evidence/backward226-comparison.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'fidelity': True, 'observations': observations}))


if __name__ == '__main__':
    main()
