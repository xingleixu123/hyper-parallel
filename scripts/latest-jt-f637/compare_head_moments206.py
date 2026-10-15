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

"""Audit full AdamW moment differences and make bounded diagnostic-only sparse patches."""
import hashlib
import json
from pathlib import Path

import numpy as np

from compare_vocab_optimizer_trace import fidelity
from head_moments206_observation import digest

ROOT = Path('/alignment/phase12')
STATE = Path('/state/jt-recipe-20261014')
RUNS = {'mf': 'mf-head206-moments', 'hp': 'hp-jtf637-sharedup-head206-moments'}


def compare(mf: np.ndarray, hp: np.ndarray, target: Path) -> dict:
    """Measure complete arrays before allocating a bounded sparse difference list."""
    mf, hp = mf.reshape(-1), hp.reshape(-1)
    count = 0
    maximum = error_squared = reference_squared = 0.0
    for start in range(0, mf.size, 1_000_000):
        left, right = mf[start:start + 1_000_000], hp[start:start + 1_000_000]
        difference = right.astype(np.float64) - left.astype(np.float64)
        count += int(np.count_nonzero(left != right))
        maximum = max(maximum, float(np.abs(difference).max()))
        error_squared += float(np.dot(difference, difference))
        reference = left.astype(np.float64)
        reference_squared += float(np.dot(reference, reference))
    if not np.isfinite(error_squared + reference_squared) or reference_squared <= 0:
        raise ValueError('Invalid moment values or reference norm')
    result = {'size': int(mf.size), 'different': count, 'max_abs': maximum,
              'relative_l2': float(np.sqrt(error_squared / reference_squared)), 'patch_ready': False}
    if count > 10_000_000:
        result['retention_reason'] = 'Difference is too dense for the planned small patch; retain full arrays.'
        return result
    indices = np.empty(count, dtype=np.int64)
    mf_values, hp_values = np.empty(count, dtype=np.float32), np.empty(count, dtype=np.float32)
    position = 0
    for start in range(0, mf.size, 1_000_000):
        left, right = mf[start:start + 1_000_000], hp[start:start + 1_000_000]
        selected = np.flatnonzero(left != right)
        end = position + selected.size
        indices[position:end] = selected + start
        mf_values[position:end], hp_values[position:end] = left[selected], right[selected]
        position = end
    if position != count:
        raise ValueError('Moment changed between comparison passes')
    if target.exists():
        raise FileExistsError(target)
    np.savez(target, indices=indices, mf=mf_values, hp=hp_values)
    result.update(patch_ready=True, patch_file=str(target), patch_bytes=target.stat().st_size,
                  patch_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    return result


def main() -> None:
    """Check actual input/restore/source and model identity before comparing moments."""
    checks, arrays, metadata = {}, {}, {}
    for family, run in RUNS.items():
        baseline = 'mf-resume' if family == 'mf' else 'hp-jtf637-adamw-all-sharedupfp32400'
        base = ROOT / ('tools' if family == 'mf' else 'tools-f637')
        directory = ROOT / 'runs' / run
        check = fidelity(run, baseline, 6)
        check['exit_zero'] = (ROOT / 'runs' / (run + '.exit')).read_text().strip() == '0'
        report = json.loads((directory / 'HEAD_MOMENTS206.json').read_text())
        check['step_and_kind'] = report['step'] == report['optimizer_step'] == 206 and report['family'] == family
        check['source'] = report['source_sha256'] == hashlib.sha256(
            (base / (family + '_head_moments206_probe.py')).read_bytes()).hexdigest()
        check['helper'] = report['helper_sha256'] == hashlib.sha256(
            (base / 'head_moments206_observation.py').read_bytes()).hexdigest()
        expected = json.loads((ROOT / 'evidence/head206-model-comparison.json').read_text())
        check['independent_head_state_exact'] = report['model_fp32_sha256'] == expected['model_records'][family]['fp32_sha256']
        check['read_only'] = report['read_only_observation']
        if family == 'hp':
            manifest = json.loads((directory / 'PROBE_SOURCE_MANIFEST.json').read_text())
            check['49_base_sources'] = len(manifest['files']) == 49 and all(
                row['sha256'] == hashlib.sha256((base / row['name']).read_bytes()).hexdigest()
                for row in manifest['files'])
        arrays[family] = {}
        for row in report['records']:
            array = np.load(STATE / run / row['file'], mmap_mode='r')
            check[row['kind'] + '_full_hash'] = digest(array) == row['sha256']
            arrays[family][row['kind']] = array
        check['two_moments'] = set(arrays[family]) == {'exp_avg', 'exp_avg_sq'}
        checks[family], metadata[family] = check, report
    if not all(all(row.values()) for row in checks.values()):
        raise ValueError(checks)
    target = STATE / 'head-moments206-difference-patch'
    target.mkdir(exist_ok=False)
    result = {kind: compare(arrays['mf'][kind], arrays['hp'][kind], target / (kind + '.npz'))
              for kind in ['exp_avg', 'exp_avg_sq']}
    report = {'complete': True, 'fidelity': checks, 'runs': RUNS, 'metadata': metadata,
              'moments': result, 'all_patches_ready': all(row['patch_ready'] for row in result.values()),
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'scope': 'Actual complete head206 moments; magnitude and sparse diagnostic patch only. '
                       'No claim of initial restore failure or numerical impact before controlled intervention.'}
    (ROOT / 'evidence/head-moments206-comparison.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'complete': True, 'fidelity': checks, 'moments': result}))


if __name__ == '__main__':
    main()
