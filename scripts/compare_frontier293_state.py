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

"""Locate the sole293 main-layer0 packed gate/up divergence in gradient, momentum or weight update."""
import hashlib
import json
from pathlib import Path

from compare_frontier219_state import load
from compare_preclip217 import compare_arrays
from compare_vocab_optimizer_trace import fidelity, read_rows
from model_hash_stream import model_record

ROOT = Path('/alignment/phase12')
STATE = Path('/state/jt-recipe-20261014')
HP = 'hp-jtf637-qup-state293-r3'
MF = 'mf-frontier293-state'
BASE = 'hp-jtf637-qup-nope400'
TARGETS = {'main0_gate': 'model.layers.0.mlp.gate_proj.weight',
           'main0_up': 'model.layers.0.mlp.up_proj.weight'}
MF_TARGET = 'decoder.layers.0.mlp.linear_fc1.weight'


def digest(path: Path) -> str:
    """Hash bounded source or metadata files."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Require full trajectory fidelity and link sampled states to independent model hashes."""
    checks = {'hp': fidelity(HP, BASE, 93), 'mf': fidelity(MF, 'mf-resume', 93)}
    for family, run in [('hp', HP), ('mf', MF)]:
        checks[family]['exit_zero'] = (ROOT / 'runs' / (run + '.exit')).read_text().strip() == '0'
        original = BASE if family == 'hp' else 'mf-frontier229-state'
        before, after = ('  train_iters: 400', '  train_iters: 293') if family == 'hp' else (
            '  stop_step: 29', '  stop_step: 93')
        checks[family]['only_window_and_paths'] = (ROOT / 'configs' / (original + '.yaml')).read_text().replace(
            original, run).replace(before, after) == (ROOT / 'configs' / (run + '.yaml')).read_text()
        launch = json.loads((ROOT / 'evidence' / (run + '-launch.json')).read_text())
        checks[family]['config_sha'] = launch['config_sha256'] == digest(ROOT / 'configs' / (run + '.yaml'))
        checks[family]['launcher_sha'] = launch['launcher_sha256'] == digest(ROOT / ('run-' + run + '.sh'))
        folder = ROOT / ('tools-f637' if family == 'hp' else 'tools')
        checks[family]['entry_sha'] = launch['entry_sha256'] == digest(folder / ('capture_hp_frontier293_state_r2.py' if family == 'hp' else 'capture_mf_frontier293_state.py'))
    for filename in ['SOURCE_REVISION.json', 'PROBE_SOURCE_MANIFEST.json', 'HEAD_GEMM_ROUTE.json',
                     'SHARED_UP_FP32_FACTOR.json', 'ADAMW_ARCHIVE_ARITHMETIC.json', 'Q_NS_ADJACENT_FACTOR.json',
                     'ROUTER_NS_NPU_SQRT_FACTOR.json', 'WEIGHTED_DVM_FACTOR.json', 'PACKED_SEMANTICS_FACTOR.json',
                     'NORM219_FACTOR.json', 'Q_NPU_SQRT222_FACTOR.json', 'CE_TAIL226_FACTOR.json', 'NS_EPSILON229_FACTOR.json', 'GLOBAL_NPU_SQRT235_FACTOR.json', 'QUP258_FACTOR.json']:
        checks['hp']['unchanged_' + filename] = (ROOT / 'runs' / HP / filename).read_bytes() == (
            ROOT / 'runs' / BASE / filename).read_bytes()
    for filename in ['HEAD_GEMM_ROUTE_CALLS.jsonl', 'Q_NS_ADJACENT_CALLS.jsonl', 'ROUTER_NS_NPU_SQRT_CALLS.jsonl',
                     'WEIGHTED_DVM_CALLS.jsonl', 'PACKED_SEMANTICS_CALLS.jsonl', 'NORM219_CALLS.jsonl',
                     'CLIP_REDUCTION.jsonl', 'CE_TAIL226_CALLS.jsonl', 'NS_EPSILON229_CALLS.jsonl', 'QUP258_CALLS.jsonl']:
        checks['hp']['same93_' + filename] = read_rows(ROOT / 'runs' / HP / filename) == read_rows(
            ROOT / 'runs' / BASE / filename)[:93]
    manifest = json.loads((ROOT / 'runs' / HP / 'PROBE_SOURCE_MANIFEST.json').read_text())
    checks['hp']['49_frozen_sources'] = len(manifest['files']) == 49 and all(
        row['sha256'] == digest(ROOT / 'tools-f637' / row['name']) for row in manifest['files'])
    models = {family: {row['step']: {item['name']: item for item in row['records']} for row in read_rows(
        ROOT / 'runs' / run / 'MODEL_HASHES.jsonl')}
        for family, run in [('hp', HP), ('mf', MF)]}
    full_model_comparisons = []
    for step in [292, 293]:
        a, b = models['hp'][step], models['mf'][step]
        if len(a) != 48 or set(a) != set(b):
            raise ValueError('Incomplete full model identities')
        full_model_comparisons.append({'step': step, 'different': {
            key: [name for name in a if a[name][key] != b[name][key]]
            for key in ['fp32_sha256', 'bf16_as_fp32_sha256']}})
    for family, prior_run in [('hp', 'hp-jtf637-qup-preclip294'), ('mf', 'mf-native-stat294')]:
        prior = next(row for row in read_rows(ROOT / 'runs' / prior_run / 'MODEL_HASHES.jsonl')
                     if row['step'] == 293)
        checks[family]['all48_models293_native_link'] = models[family][293] == {
            item['name']: item for item in prior['records']}
    states, links = {}, {}
    for step in [292, 293]:
        arrays = {}
        for family, run in [('hp', HP), ('mf', MF)]:
            report = json.loads((ROOT / 'runs' / run / ('FRONTIER_STATE-' + str(step) + '.json')).read_text())
            folder = ROOT / ('tools-f637' if family == 'hp' else 'tools')
            source = 'hp_frontier293_state_probe.py' if family == 'hp' else 'mf_frontier293_state_tap.py'
            checks[family]['source_' + str(step)] = report['source_sha256'] == digest(folder / source)
            checks[family]['readonly_' + str(step)] = report['read_only'] and not report['teacher_values_used']
            checks[family]['state_count_' + str(step)] = len(report['records']) == (4 if family == 'hp' else 2)
            if family == 'mf':
                checks[family]['optimizer_step_' + str(step)] = report['optimizer_step'] == step
            for row in report['records']:
                value = load(STATE / run / ('state-' + str(step)), row)
                if family == 'hp':
                    arrays[family, row['label'], row['kind']] = value
                else:
                    if tuple(value.shape) != (24576, 5120):
                        raise ValueError('Unexpected native packed fc1 layout')
                    arrays[family, 'main0_gate', row['kind']] = value[::2]
                    arrays[family, 'main0_up', row['kind']] = value[1::2]
            for label, name in TARGETS.items():
                observed = model_record(name, arrays[family, label, 'weight'])
                links[family + '_' + str(step) + '_' + label] = all(
                    observed[key] == models[family][step][name][key]
                    for key in ['fp32_sha256', 'bf16_as_fp32_sha256'])
        states[step] = {label: {kind: compare_arrays(arrays['hp', label, kind], arrays['mf', label, kind])
                               for kind in ['weight', 'muon_m']} for label in TARGETS}
        del arrays
    native = json.loads((ROOT / 'runs' / MF / 'FRONTIER_GRADIENTS-293.json').read_text())
    checks['mf']['one_native_gradient'] = len(native) == 1 and native[0]['name'] == MF_TARGET and native[0]['retained']
    hp_gradient = json.loads((ROOT / 'runs' / HP / 'FRONTIER_GRADIENT-293.json').read_text())
    checks['hp']['gradient_source'] = hp_gradient['source_sha256'] == digest(ROOT / 'tools-f637/hp_frontier293_state_probe.py')
    checks['hp']['only_postclip_gradient'] = len(hp_gradient['records']) == 1
    row = next(row for row in hp_gradient['records'] if row['kind'] == 'postclip')
    gradient = compare_arrays(load(STATE / HP / 'gradient-293', row), load(STATE / MF / 'gradients', native[0]))
    if not all(all(row.values()) for row in checks.values()) or not all(links.values()):
        raise ValueError({'fidelity': checks, 'model_hash_links': links})
    report = {'complete': True, 'fidelity': checks, 'model_hash_links': links, 'states': states,
              'clipped_gradient': gradient, 'full_model_comparisons': full_model_comparisons, 'source_sha256': digest(Path(__file__)),
              'scope': 'One entire main-layer0 packed gate/up tensor; exact independent model links at292 and293; '
                       'no conclusion on other tensors or entire optimizer state.'}
    (ROOT / 'evidence/frontier293-state-comparison.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == '__main__':
    main()
