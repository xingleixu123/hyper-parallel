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

"""Audit two actual head206 observations and preserve only a compact exact difference patch."""
import hashlib
import json
from pathlib import Path

import numpy as np
from compare_vocab_optimizer_trace import fidelity, read_rows
from model_hash_stream import model_record

ROOT = Path('/alignment/phase12')
STATE = Path('/state/jt-recipe-20261014')
MF = 'mf-head206-state-r2'
HP = 'hp-jtf637-sharedup-head206-state'


def main():
    checks, arrays, records = {}, {}, {}
    for family, run, baseline, observer, helper in [
        ('mf', MF, 'mf-resume', 'tools/mf_head206_state_probe_v2.py', 'tools/head206_observation_v2.py'),
        ('hp', HP, 'hp-jtf637-adamw-all-sharedupfp32206', 'tools-f637/hp_head206_state_probe.py', 'tools-f637/head206_observation.py')]:
        checks[family] = fidelity(run, baseline, 6)
        checks[family]['exit_zero'] = (ROOT / 'runs' / (run + '.exit')).read_text().strip() == '0'
        report = json.loads((ROOT / 'runs' / run / 'HEAD206_STATE.json').read_text())
        checks[family]['observer_source_exact'] = report['source_sha256'] == hashlib.sha256((ROOT / observer).read_bytes()).hexdigest()
        checks[family]['helper_source_exact'] = report['helper_sha256'] == hashlib.sha256((ROOT / helper).read_bytes()).hexdigest()
        checks[family]['hash_helper_source_exact'] = report['hash_helper_sha256'] == hashlib.sha256((ROOT / helper).with_name('model_hash_stream.py').read_bytes()).hexdigest()
        array = np.load(STATE / run / report['file'], mmap_mode='r')
        record = model_record('lm_head.weight', array)
        checks[family]['actual_array_hash_exact'] = record == report['record']
        hash_run = 'mf-modelhash226' if family == 'mf' else 'hp-jtf637-sharedup-modelhash226'
        old = next(row for row in read_rows(ROOT / 'runs' / hash_run / 'MODEL_HASHES.jsonl') if row['step'] == 206)
        old = next(row for row in old['records'] if row['name'] == 'lm_head.weight')
        checks[family]['independent_modelhash206_exact'] = record == old
        arrays[family], records[family] = array.reshape(-1), record
    if not all(all(values.values()) for values in checks.values()):
        raise ValueError(checks)
    indices, hp_values, mf_values = [], [], []
    maximum = error_squared = reference_squared = 0.0
    bf16_different = 0
    for start in range(0, arrays['mf'].size, 1_000_000):
        mf, hp = arrays['mf'][start:start + 1_000_000], arrays['hp'][start:start + 1_000_000]
        selected = np.flatnonzero(mf != hp)
        indices.append(selected.astype(np.int64) + start)
        mf_values.append(np.asarray(mf[selected]))
        hp_values.append(np.asarray(hp[selected]))
        difference = hp.astype(np.float64) - mf.astype(np.float64)
        maximum = max(maximum, float(abs(difference).max()))
        error_squared += float(np.dot(difference, difference))
        reference_squared += float(np.dot(mf.astype(np.float64), mf.astype(np.float64)))
        def bf16(values):
            bits = values.view(np.uint32)
            return (bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) & np.uint32(0xffff0000)
        bf16_different += int(np.count_nonzero(bf16(mf) != bf16(hp)))
    indices, mf_values, hp_values = np.concatenate(indices), np.concatenate(mf_values), np.concatenate(hp_values)
    patch_dir = STATE / 'head206-difference-patch'
    patch_dir.mkdir(exist_ok=False)
    patch = patch_dir / 'patch.npz'
    np.savez(patch, indices=indices, mf=mf_values, hp=hp_values)
    report = {'complete': True, 'fidelity': checks, 'model_records': records,
              'size': int(arrays['mf'].size), 'different_fp32': int(indices.size),
              'different_bf16': bf16_different, 'max_abs': maximum,
              'relative_l2': float((error_squared / reference_squared) ** .5),
              'patch_file': str(patch), 'patch_bytes': patch.stat().st_size,
              'patch_sha256': hashlib.sha256(patch.read_bytes()).hexdigest(),
              'scope': 'Actual entire head206 compared; compact patch contains only differing FP32 coordinates and is restricted to diagnostic interventions.'}
    (ROOT / 'evidence/head206-model-comparison.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
