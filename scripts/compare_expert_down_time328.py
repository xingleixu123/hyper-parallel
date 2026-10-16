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

"""Validate the narrowed expert-down trajectory and identify its first differing update."""
import hashlib
import json
from pathlib import Path

from compare_vocab_optimizer_trace import fidelity, read_rows

ROOT = Path('/alignment/phase12')
HP = 'hp-jte213-expertdown-time328'
MF = 'mf-expertdown-time328'
BASE = 'hp-jte213-fc1norm-qupboth400'
TARGET = 'model.layers.1.mlp.experts.down_proj'


def digest(path: Path) -> str:
    """Hash a small source or configuration file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Require faithful128-step trajectories and exact links to both observed endpoints."""
    checks = {'hp': fidelity(HP, BASE, 128), 'mf': fidelity(MF, 'mf-resume', 128)}
    observations = {}
    for family, run in [('hp', HP), ('mf', MF)]:
        launch = json.loads((ROOT / 'evidence' / (run + '-launch.json')).read_text())
        base = BASE if family == 'hp' else 'mf-native-stat330'
        before, after = ('  train_iters: 400', '  train_iters: 328') if family == 'hp' else (
            '  stop_step: 130', '  stop_step: 128')
        checks[family]['exit_zero'] = (ROOT / 'runs' / (run + '.exit')).read_text().strip() == '0'
        checks[family]['config_only_window_paths'] = (ROOT / 'configs' / (base + '.yaml')).read_text().replace(
            base, run).replace(before, after) == (ROOT / 'configs' / (run + '.yaml')).read_text()
        checks[family]['config_hash'] = launch['config_sha256'] == digest(ROOT / 'configs' / (run + '.yaml'))
        checks[family]['launcher_hash'] = launch['launcher_sha256'] == digest(ROOT / ('run-' + run + '.sh'))
        folder = ROOT / ('tools-f637' if family == 'hp' else 'tools')
        entry_folder = ROOT / 'tools-e213' if family == 'hp' else folder
        checks[family]['entry_hash'] = launch['entry_sha256'] == digest(
            entry_folder / ('capture_' + family + '_expert_down_time.py'))
        metadata = json.loads((ROOT / 'runs' / run / 'EXPERT_DOWN_TIME_OBSERVER.json').read_text())
        checks[family]['observer_hash'] = metadata['source_sha256'] == digest(
            folder / (family + '_expert_down_time_probe.py'))
        checks[family]['helper_hash'] = metadata['helper_sha256'] == digest(folder / 'model_hash_stream.py')
        checks[family]['readonly'] = not metadata['raw_arrays_written'] and not metadata['teacher_values_used']
        rows = read_rows(ROOT / 'runs' / run / 'EXPERT_DOWN_TIME.jsonl')
        checks[family]['complete_steps'] = [row['step'] for row in rows] == list(range(307, 329))
        observations[family] = {row['step']: {item['kind']: item for item in row['records']} for row in rows}
        checks[family]['complete_tensors'] = all(set(value) == {'weight', 'muon_m'} and all(
            len(row['experts']) == 8 for row in value.values()) for value in observations[family].values())
        if family == 'mf':
            checks[family]['optimizer_steps'] = all(row['optimizer_step'] == row['step'] for row in rows)
        endpoints = [('hp-jte213-fc1norm-qupboth-preclip307' if family == 'hp' else 'mf-native-stat307', 307),
                     ('hp-jte213-fc1norm-qupboth-state329' if family == 'hp' else 'mf-frontier329-state', 328)]
        for prior, step in endpoints:
            model = next(row for row in read_rows(ROOT / 'runs' / prior / 'MODEL_HASHES.jsonl') if row['step'] == step)
            expected = next(row for row in model['records'] if row['name'] == TARGET)
            actual = observations[family][step]['weight']['all']
            checks[family]['independent_endpoint_' + str(step)] = all(
                expected[key] == actual[key] for key in ['fp32_sha256', 'bf16_as_fp32_sha256'])
    for filename in ['SOURCE_REVISION.json', 'PROBE_SOURCE_MANIFEST.json', 'WEIGHTED_DVM_FACTOR.json',
                     'NORM219_FACTOR.json', 'Q_NPU_SQRT222_FACTOR.json', 'CE_TAIL226_FACTOR.json',
                     'NS_EPSILON229_FACTOR.json', 'GLOBAL_NPU_SQRT235_FACTOR.json', 'QUP258_FACTOR.json',
                     'FC1293_FACTOR.json']:
        checks['hp']['unchanged_' + filename] = (ROOT / 'runs' / HP / filename).read_bytes() == (
            ROOT / 'runs' / BASE / filename).read_bytes()
    frozen = json.loads((ROOT / 'runs' / HP / 'PROBE_SOURCE_MANIFEST.json').read_text())
    checks['hp']['49_frozen_sources'] = len(frozen['files']) == 49 and all(
        row['sha256'] == digest(ROOT / 'tools-f637' / row['name']) for row in frozen['files'])
    comparisons = []
    for step in range(307, 329):
        changes = {}
        for kind in ['weight', 'muon_m']:
            first, second = observations['hp'][step][kind], observations['mf'][step][kind]
            changes[kind] = {key: [index for index in range(8)
                                  if first['experts'][index][key] != second['experts'][index][key]]
                             for key in ['fp32_sha256', 'bf16_as_fp32_sha256']}
        comparisons.append({'step': step, 'differences': changes})
    firsts = {kind: {key: next((row['step'] for row in comparisons if row['differences'][kind][key]), None)
                    for key in ['fp32_sha256', 'bf16_as_fp32_sha256']} for kind in ['weight', 'muon_m']}
    if not all(all(row.values()) for row in checks.values()):
        raise ValueError(checks)
    report = {'complete': True, 'fidelity': checks, 'first_differences': firsts, 'comparisons': comparisons,
              'source_sha256': digest(Path(__file__)),
              'scope': 'Selected entire expert-down weight and momentum, both independent endpoints linked; '
                       'other weights and intermediate activations are outside this temporal probe.'}
    (ROOT / 'evidence/frontier329-temporal-first-difference.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'fidelity': checks, 'first_differences': firsts,
                      'changed_steps': [row for row in comparisons if any(row['differences']['weight'].values())]}))


if __name__ == '__main__':
    main()
