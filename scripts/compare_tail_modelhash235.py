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

"""Audit unchanged trajectories and all model hashes at the next five boundaries."""
import hashlib
import json
from pathlib import Path

from compare_vocab_optimizer_trace import fidelity, read_rows

ROOT = Path('/alignment/phase12')
HP = 'hp-jtf637-epsilon235'
MF = 'mf-modelhash235'
STEPS = list(range(231, 236))


def digest(path: Path) -> str:
    """Compute a file hash without loading training arrays."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hashes(run: str) -> dict:
    """Require all 48 postupdate model tensors at each requested step."""
    rows = read_rows(ROOT / 'runs' / run / 'MODEL_HASHES.jsonl')
    if [row['step'] for row in rows] != STEPS:
        raise ValueError('Incomplete model observations: ' + run)
    result = {}
    for row in rows:
        records = {record['name']: record for record in row['records']}
        if len(records) != 48 or len(row['records']) != 48:
            raise ValueError('Expected48 model records')
        result[row['step']] = records
    return result


def main() -> None:
    """Check read-only instrumentation before reporting the next numerical boundary."""
    checks = {}
    for family, run, base in [('hp', HP, 'hp-jtf637-epsilon400'), ('mf', MF, 'mf-resume')]:
        directory = ROOT / 'runs' / run
        check = fidelity(run, base, 35)
        check['exit_zero'] = (ROOT / 'runs' / (run + '.exit')).read_text().strip() == '0'
        folder = ROOT / ('tools-f637' if family == 'hp' else 'tools')
        observer = json.loads((directory / 'MODEL_HASH_OBSERVER.json').read_text())
        check['observer_source'] = observer['source_sha256'] == digest(folder / (family + '_tail235_model_hash.py'))
        check['helper_source'] = observer['helper_sha256'] == digest(folder / 'model_hash_stream.py')
        old_run = 'hp-jtf637-epsilon230' if family == 'hp' else 'mf-modelhash230'
        config = (ROOT / 'configs' / (old_run + '.yaml')).read_text().replace(old_run, run)
        old, new = ('  train_iters: 230', '  train_iters: 235') if family == 'hp' else (
            '  stop_step: 30', '  stop_step: 35')
        check['config_only_window_and_paths'] = config.replace(old, new) == (
            ROOT / 'configs' / (run + '.yaml')).read_text()
        launch = json.loads((ROOT / 'evidence' / (run + '-launch.json')).read_text())
        check['config_hash'] = launch['config_sha256'] == digest(ROOT / 'configs' / (run + '.yaml'))
        check['launcher_hash'] = launch['launcher_sha256'] == digest(ROOT / ('run-' + run + '.sh'))
        entry = 'capture_hp_epsilon235.py' if family == 'hp' else 'capture_mf_model_hash235.py'
        check['entry_hash'] = launch['entry_sha256'] == digest(folder / entry)
        if family == 'mf':
            check['converter_source'] = observer['converter_sha256'] == digest(
                ROOT / 'hyper-parallel/scripts/checkpoint_conversion/convert_mf_ckpt_to_hp.py')
        else:
            manifest = json.loads((directory / 'PROBE_SOURCE_MANIFEST.json').read_text())
            check['49_frozen_sources'] = len(manifest['files']) == 49 and all(
                row['sha256'] == digest(folder / row['name']) for row in manifest['files'])
            for filename in ['CE_TAIL226_FACTOR.json', 'NORM219_FACTOR.json', 'Q_NPU_SQRT222_FACTOR.json',
                             'SOURCE_REVISION.json', 'PROBE_SOURCE_MANIFEST.json', 'RESTORE.json', 'NS_EPSILON229_FACTOR.json']:
                check['unchanged_' + filename] = (directory / filename).read_bytes() == (
                    ROOT / 'runs' / base / filename).read_bytes()
            check['all_tail_calls'] = read_rows(directory / 'CE_TAIL226_CALLS.jsonl') == [
                {'step': step, 'calls': 2} for step in range(201, 236)]
        checks[family] = check
    if not all(all(check.values()) for check in checks.values()):
        raise ValueError(checks)
    left, right = hashes(HP), hashes(MF)
    comparisons = []
    for step in STEPS:
        if set(left[step]) != set(right[step]):
            raise ValueError('Model names differ')
        for name in left[step]:
            if left[step][name]['shape'] != right[step][name]['shape']:
                raise ValueError('Model shapes differ')
        differences = {key: [name for name in left[step] if left[step][name][key] != right[step][name][key]]
                       for key in ['fp32_sha256', 'bf16_as_fp32_sha256']}
        comparisons.append({'step': step, 'different': differences})
    first = next((row for row in comparisons if any(row['different'].values())), None)
    report = {'complete': True, 'fidelity': checks, 'models': comparisons, 'first_model_difference': first,
              'source_sha256': digest(Path(__file__)),
              'scope': 'Postupdate model hashes only at231-235; 201-230 equality independently proven in epsilon230; '
                       'optimizer moment equality is not implied by model equality.'}
    (ROOT / 'evidence/frontier235-modelhash-audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'fidelity': checks, 'first_model_difference': first, 'models': comparisons}))


if __name__ == '__main__':
    main()
