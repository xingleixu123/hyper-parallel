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

"""Audit full MF training fidelity before attributing dense219 NS differences."""
import hashlib
import json
from pathlib import Path
import numpy as np
from compare_preclip217 import compare_arrays
from compare_modelhash226 import verify_sources
from compare_vocab_optimizer_trace import fidelity, read_rows
ROOT = Path('/alignment/phase12')
STATE = Path('/state/jt-recipe-20261014')
RUN = 'mf-frontier219-ns-train'


def main():
    checks = fidelity(RUN, 'mf-resume', 19)
    checks.update(verify_sources(RUN, 'mf'))
    checks['exit_zero'] = (ROOT / 'runs' / (RUN + '.exit')).read_text().strip() == '0'
    checks['same_recipe'] = (ROOT / 'configs/mf-frontier219-state.yaml').read_text().replace(
        'mf-frontier219-state', RUN) == (ROOT / 'configs' / (RUN + '.yaml')).read_text()
    models = read_rows(ROOT / 'runs' / RUN / 'MODEL_HASHES.jsonl')
    checks['all48_model_hashes_exact_201_219'] = models == read_rows(
        ROOT / 'runs/mf-modelhash226/MODEL_HASHES.jsonl')[:19]
    source = json.loads((ROOT / 'evidence/frontier219-mf-ns-source.json').read_text())
    checks['observer_source'] = source['tap_source_sha256'] == hashlib.sha256(
        (ROOT / 'tools/mf_frontier219_ns_tap.py').read_bytes()).hexdigest()
    directory = STATE / 'mf-frontier219-ns-operators'
    checks['generated_source'] = source['generated_source_sha256'] == hashlib.sha256(
        (directory / 'observed_native_update.py').read_bytes()).hexdigest()
    assert all(checks.values()), checks
    records = json.loads((ROOT / 'runs' / RUN / 'NS_BOUNDARIES-219.json').read_text())
    assert len(records) == 5
    arrays = {}
    for row in records:
        value = np.load(directory / row['file'], mmap_mode='r')
        assert hashlib.sha256(value.tobytes()).hexdigest() == row['sha256']
        assert value.dtype == np.float32 and list(value.shape) == row['shape']
        arrays[row['prefix'][5:]] = value
    comparisons = {}
    for label, report_name, folder in [
        ('control', 'frontier219-hp-muon-replay', 'hp-frontier219-muon-replay'),
        ('npu_sqrt', 'frontier219-dense-npusqrt-replay', 'hp-frontier219-dense-npusqrt-replay')]:
        report = json.loads((ROOT / 'evidence' / (report_name + '.json')).read_text())
        row = next(r for r in report['records'] if r['label'] == 'down')
        compared = {'hp_scalars': row['scalar'],
                    'mf_norm': arrays['norm'].reshape(-1).tolist(),
                    'mf_denominator': arrays['denominator'].reshape(-1).tolist()}
        for record in row['arrays']:
            value = np.load(STATE / folder / 'down-0' / (record['name'] + '.npy'), mmap_mode='r')
            assert hashlib.sha256(value.tobytes()).hexdigest() == record['sha256']
            compared[record['name']] = compare_arrays(value, arrays[record['name']])
        comparisons[label] = compared
    output = {'complete': True, 'full_training_fidelity': checks, 'comparisons': comparisons,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'scope': 'Native full-training dense219 NS boundaries; isolated MF dense replay deliberately excluded.'}
    (ROOT / 'evidence/frontier219-dense-ns-comparison.json').write_text(json.dumps(output, indent=2))
    print(json.dumps(output))


if __name__ == '__main__':
    main()
