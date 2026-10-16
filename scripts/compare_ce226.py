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

"""Reject observer-induced drift, then locate the first actual CE internal difference."""
import hashlib
import json
from pathlib import Path

import numpy as np

from compare_preclip212 import compare_arrays
from compare_head206_interventions import source_sha
from compare_vocab_optimizer_trace import fidelity, read_rows

ROOT = Path('/alignment/phase12')
STATE = Path('/state/jt-recipe-20261014')
HP, MF = 'hp-jtf637-qsqrt222-ce226', 'mf-ce226'
BASE = 'hp-jtf637-qsqrt222-both226'


def trajectory_checks() -> dict:
    """Verify full26step input, state, metric and arithmetic-factor fidelity."""
    checks = {'hp': fidelity(HP, BASE, 26), 'mf': fidelity(MF, 'mf-resume', 26)}
    for side, run in [('hp', HP), ('mf', MF)]:
        checks[side]['exit_zero'] = (ROOT / 'runs' / (run + '.exit')).read_text().strip() == '0'
        launch = json.loads((ROOT / 'evidence' / (run + '-launch.json')).read_text())
        checks[side]['config_sha'] = hashlib.sha256((ROOT / 'configs' / (run + '.yaml')).read_bytes()).hexdigest() == launch['config_sha256']
        checks[side]['launcher_sha'] = hashlib.sha256((ROOT / ('run-' + run + '.sh')).read_bytes()).hexdigest() == launch['launcher_sha256']
    checks['hp']['config_exact_except_path'] = (ROOT / 'configs' / (BASE + '.yaml')).read_text().replace(BASE, HP) == (ROOT / 'configs' / (HP + '.yaml')).read_text()
    checks['mf']['config_exact_except_path_length'] = (ROOT / 'configs/mf-preclip217.yaml').read_text().replace('mf-preclip217', MF).replace('  stop_step: 17', '  stop_step: 26') == (ROOT / 'configs' / (MF + '.yaml')).read_text()
    for name in ['SOURCE_REVISION.json', 'PROBE_SOURCE_MANIFEST.json', 'HEAD_GEMM_ROUTE.json',
                 'SHARED_UP_FP32_FACTOR.json', 'ADAMW_ARCHIVE_ARITHMETIC.json', 'Q_NS_ADJACENT_FACTOR.json',
                 'ROUTER_NS_NPU_SQRT_FACTOR.json', 'WEIGHTED_DVM_FACTOR.json', 'PACKED_SEMANTICS_FACTOR.json',
                 'NORM219_FACTOR.json', 'Q_NPU_SQRT222_FACTOR.json', 'CE_SUM_TREE.json']:
        checks['hp']['unchanged_' + name] = (ROOT / 'runs' / HP / name).read_bytes() == (ROOT / 'runs' / BASE / name).read_bytes()
    for name in ['HEAD_GEMM_ROUTE_CALLS.jsonl', 'Q_NS_ADJACENT_CALLS.jsonl', 'ROUTER_NS_NPU_SQRT_CALLS.jsonl',
                 'CLIP_REDUCTION.jsonl', 'WEIGHTED_DVM_CALLS.jsonl', 'PACKED_SEMANTICS_CALLS.jsonl', 'NORM219_CALLS.jsonl']:
        checks['hp']['exact26_' + name] = read_rows(ROOT / 'runs' / HP / name) == read_rows(ROOT / 'runs' / BASE / name)[:26]
    frozen = json.loads((ROOT / 'runs' / HP / 'PROBE_SOURCE_MANIFEST.json').read_text())
    checks['hp']['49_frozen_sources'] = len(frozen['files']) == 49 and all(row['sha256'] == source_sha(row['name']) for row in frozen['files'])
    return checks


def load_arrays(records: list, directory: Path, key_name: str) -> dict:
    """Validate captured arrays and reject differing recomputations."""
    result = {}
    for record in records:
        array = np.load(directory / record['file'], mmap_mode='r')
        if list(array.shape) != record['shape'] or hashlib.sha256(array.tobytes()).hexdigest() != record['sha256']:
            raise ValueError('Observed CE operand changed')
        key = record[key_name], record['kind']
        if key in result and not np.array_equal(result[key], array):
            raise ValueError('CE recomputation differs')
        result[key] = array.astype(np.float32)
    return result


def errors64(actual: np.ndarray, reference: np.ndarray) -> dict:
    """Compare entire selected arrays with FP64 arithmetic from the same logits."""
    delta = np.abs(actual.astype(np.float64) - reference)
    return {'max_abs': float(delta.max()), 'mean_abs': float(delta.mean()),
            'l2': float(np.linalg.norm(delta))}


def main() -> None:
    """Locate the first forward or backward stage that differs on actual selected rows."""
    checks = trajectory_checks()
    hp_meta = json.loads((ROOT / 'runs' / HP / 'CE226_OBSERVER.json').read_text())
    mf_meta = json.loads((STATE / MF / 'deep/manifest.json').read_text())
    checks['hp']['observer_source'] = hp_meta['source_sha256'] == source_sha('hp_ce226_observer.py')
    checks['mf']['observer_source'] = mf_meta['source_sha256'] == hashlib.sha256((ROOT / 'tools/mf_ce226_tap.py').read_bytes()).hexdigest()
    checks['hp']['all26_calls'] = read_rows(ROOT / 'runs' / HP / 'CE226_CALLS.jsonl') == [{'step': step, 'calls': 2} for step in range(201, 227)]
    mf_rows = read_rows(ROOT / 'runs' / MF / 'CE226_OPERANDS.jsonl')
    checks['mf']['all26_observations'] = [r['step'] for r in mf_rows] == list(range(201, 227))
    report = {'complete': False, 'fidelity': checks, 'runs': {'hp': HP, 'mf': MF},
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'scope': 'Four full vocabulary rows selected from actual full4096x152064 operations; no reference values injected.'}
    destination = ROOT / 'evidence/ce226-comparison.json'
    if not all(all(row.values()) for row in checks.values()):
        destination.write_text(json.dumps(report, indent=2))
        raise ValueError(checks)
    hp_records = json.loads((ROOT / 'runs' / HP / 'CE226_OPERANDS.json').read_text())['records']
    hp = load_arrays(hp_records, STATE / HP / 'ce', 'call')
    mf = load_arrays(mf_rows[-1]['records'], STATE / MF / 'deep', 'name')
    hp_head_records = json.loads((ROOT / 'runs/hp-jtf637-qsqrt222-head226-r3/HEAD226_OPERANDS.json').read_text())['records']
    mf_head_records = read_rows(ROOT / 'runs/mf-head226/HEAD226_OPERANDS.jsonl')[-1]['records']
    hp_head = load_arrays(hp_head_records, STATE / 'hp-jtf637-qsqrt222-head226-r3/head', 'call')
    mf_head = load_arrays(mf_head_records, STATE / 'mf-head226/deep', 'name')
    selection = hp_meta['selection']['token_rows']
    observations, precision, mapping = [], [], {}
    kinds = ['logits', 'shifted', 'exponent', 'sum', 'log_sum', 'log_probability',
             'probability', 'dout', 'dx_fp32', 'dx_bf16']
    for module in mf_meta['modules']:
        name = module['name']
        candidates = [call for call in [1, 2] if np.array_equal(mf[name, 'logits'], hp[call, 'logits'])]
        if len(candidates) != 1:
            raise ValueError('CE branch mapping not unique')
        call = candidates[0]
        head_names = [n for n, k in mf_head if k == 'logits' and np.array_equal(mf_head[n, k], mf[name, 'logits'])]
        if len(head_names) != 1:
            raise ValueError('CE-to-head mapping not unique')
        head_name = head_names[0]
        head_calls = [c for c, k in hp_head if k == 'logits' and np.array_equal(hp_head[c, k], hp[call, 'logits'])]
        if len(head_calls) != 1:
            raise ValueError('HP CE-to-head mapping not unique')
        mapping[name] = {'ce_call': call, 'mf_head': head_name, 'hp_head_call': head_calls[0]}
        checks['hp']['head_dy_exact_' + name] = np.array_equal(hp[call, 'dx_bf16'], hp_head[head_calls[0], 'dy'])
        checks['mf']['head_dy_exact_' + name] = np.array_equal(mf[name, 'dx_bf16'], mf_head[head_name, 'dy_matmul'])
        mask = mf[name, 'mask'].reshape(-1)
        checks['hp']['mask_exact_' + name] = np.array_equal(mask, hp[call, 'mask'].reshape(-1))
        valid = mask != 0
        checks['hp']['valid_targets_exact_' + name] = np.array_equal(mf[name, 'targets'].reshape(-1)[valid], hp[call, 'targets'].reshape(-1)[valid])
        for kind in kinds:
            left, right = hp[call, kind], mf[name, kind]
            observations.append({'name': name, 'kind': kind, 'numeric_different': int(np.count_nonzero(left != right)), **compare_arrays(left, right)})
        logits = hp[call, 'logits'].astype(np.float64)
        shifted = logits - logits.max(axis=-1, keepdims=True)
        exponent = np.exp(shifted)
        total = exponent.sum(axis=-1, keepdims=True)
        log_total = np.log(total)
        log_probability = shifted - log_total
        probability = np.exp(log_probability)
        refs = {'shifted': shifted, 'exponent': exponent, 'sum': total, 'log_sum': log_total,
                'log_probability': log_probability, 'probability': probability}
        for kind, reference in refs.items():
            precision.append({'name': name, 'kind': kind, 'hp': errors64(hp[call, kind], reference), 'mf': errors64(mf[name, kind], reference)})
        for index, token in enumerate(selection):
            observations.append({'name': name, 'token': token, 'mask': float(mask[token]),
                                 'hp_dout': float(hp[call, 'dout'][index, 0]),
                                 'mf_dout': float(mf[name, 'dout'][index, 0]),
                                 'bf16_numeric_different': int(np.count_nonzero(hp[call, 'dx_bf16'][index] != mf[name, 'dx_bf16'][index]))})
    report.update(complete=all(all(row.values()) for row in checks.values()), mapping=mapping,
                  observations=observations, precision_fp64=precision)
    destination.write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
    if not report['complete']:
        raise ValueError('Cross-run boundary fidelity failed')


if __name__ == '__main__':
    main()
