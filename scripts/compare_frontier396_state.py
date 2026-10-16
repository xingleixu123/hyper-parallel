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

"""Locate the sole396 main-layer1 shared-expert down divergence in gradient, momentum or weight update."""
import hashlib
import json
from pathlib import Path

import numpy as np

from compare_frontier219_state import load
from compare_preclip217 import compare_arrays
from compare_vocab_optimizer_trace import fidelity, read_rows
from model_hash_stream import model_record
from mindspore272_stat_hash import native_dump_sha1

ROOT = Path('/alignment/phase12')
STATE = Path('/state/jt-recipe-20261014')
HP = 'hp-jte213-expertnorm328-state396-r2'
MF = 'mf-frontier396-state'
BASE = 'hp-jte213-expertnorm328400'
TARGET = 'model.layers.1.mlp.shared_experts.down_proj.weight'
MF_TARGET = 'decoder.layers.1.mlp.shared_experts.linear_fc2.weight'


def digest(path: Path) -> str:
    """Hash bounded source or metadata files."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Require full trajectory fidelity and link sampled states to independent model hashes."""
    checks = {'hp': fidelity(HP, BASE, 196), 'mf': fidelity(MF, 'mf-resume', 196)}
    for family, run in [('hp', HP), ('mf', MF)]:
        checks[family]['exit_zero'] = (ROOT / 'runs' / (run + '.exit')).read_text().strip() == '0'
        original = BASE if family == 'hp' else 'mf-frontier328-state'
        before, after = ('  train_iters: 400', '  train_iters: 396') if family == 'hp' else (
            '  stop_step: 128', '  stop_step: 196')
        checks[family]['only_window_and_paths'] = (ROOT / 'configs' / (original + '.yaml')).read_text().replace(
            original, run).replace(before, after) == (ROOT / 'configs' / (run + '.yaml')).read_text()
        launch = json.loads((ROOT / 'evidence' / (run + '-launch.json')).read_text())
        checks[family]['config_sha'] = launch['config_sha256'] == digest(ROOT / 'configs' / (run + '.yaml'))
        checks[family]['launcher_sha'] = launch['launcher_sha256'] == digest(ROOT / ('run-' + run + '.sh'))
        folder = ROOT / ('tools-f637' if family == 'hp' else 'tools')
        entry_folder = ROOT / 'tools-e213' if family == 'hp' else folder
        entry = 'capture_hp_frontier396_state_r2.py' if family == 'hp' else 'capture_mf_frontier396_state.py'
        checks[family]['entry_sha'] = launch['entry_sha256'] == digest(entry_folder / entry)
    for filename in ['SOURCE_REVISION.json', 'PROBE_SOURCE_MANIFEST.json', 'HEAD_GEMM_ROUTE.json',
                     'SHARED_UP_FP32_FACTOR.json', 'ADAMW_ARCHIVE_ARITHMETIC.json', 'Q_NS_ADJACENT_FACTOR.json',
                     'ROUTER_NS_NPU_SQRT_FACTOR.json', 'WEIGHTED_DVM_FACTOR.json', 'PACKED_SEMANTICS_FACTOR.json',
                     'NORM219_FACTOR.json', 'Q_NPU_SQRT222_FACTOR.json', 'CE_TAIL226_FACTOR.json', 'NS_EPSILON229_FACTOR.json', 'GLOBAL_NPU_SQRT235_FACTOR.json', 'QUP258_FACTOR.json', 'FC1293_FACTOR.json', 'EXPERT328_FACTOR.json']:
        checks['hp']['unchanged_' + filename] = (ROOT / 'runs' / HP / filename).read_bytes() == (
            ROOT / 'runs' / BASE / filename).read_bytes()
    for filename in ['HEAD_GEMM_ROUTE_CALLS.jsonl', 'Q_NS_ADJACENT_CALLS.jsonl', 'ROUTER_NS_NPU_SQRT_CALLS.jsonl',
                     'WEIGHTED_DVM_CALLS.jsonl', 'PACKED_SEMANTICS_CALLS.jsonl', 'NORM219_CALLS.jsonl',
                     'CLIP_REDUCTION.jsonl', 'CE_TAIL226_CALLS.jsonl', 'QUP258_CALLS.jsonl', 'FC1293_CALLS.jsonl', 'EXPERT328_CALLS.jsonl',
                     'NS_EPSILON229_CALLS.jsonl']:
        checks['hp']['same196_' + filename] = read_rows(ROOT / 'runs' / HP / filename) == read_rows(
            ROOT / 'runs' / BASE / filename)[:196]
    manifest = json.loads((ROOT / 'runs' / HP / 'PROBE_SOURCE_MANIFEST.json').read_text())
    checks['hp']['49_frozen_sources'] = len(manifest['files']) == 49 and all(
        row['sha256'] == digest(ROOT / 'tools-f637' / row['name']) for row in manifest['files'])
    models = {}
    for family, run in [('hp', HP), ('mf', MF)]:
        records = read_rows(ROOT / 'runs' / run / 'MODEL_HASHES.jsonl')
        models[family] = {row['step']: {item['name']: item for item in row['records']} for row in records}
        checks[family]['both_model_steps_complete'] = set(models[family]) == {395, 396} and all(
            len(row) == 48 for row in models[family].values())
        folder = ROOT / ('tools-f637' if family == 'hp' else 'tools')
        observer = json.loads((ROOT / 'runs' / run / 'MODEL_HASH_OBSERVER.json').read_text())
        checks[family]['model_observer_source'] = observer['source_sha256'] == digest(
            folder / (family + '_tail396_model_hash.py'))
        checks[family]['model_helper_source'] = observer['helper_sha256'] == digest(folder / 'model_hash_stream.py')
        if family == 'mf':
            checks[family]['converter_source'] = observer['converter_sha256'] == digest(
                ROOT / 'hyper-parallel/scripts/checkpoint_conversion/convert_mf_ckpt_to_hp.py')
        independent = 'hp-jte213-expertnorm328-preclip397' if family == 'hp' else 'mf-native-stat397'
        for step in [395, 396]:
            native = next(row for row in read_rows(ROOT / 'runs' / independent / 'MODEL_HASHES.jsonl')
                          if row['step'] == step)
            checks[family]['independent_model_' + str(step)] = {
                row['name']: row for row in native['records']} == models[family][step]
    boundary = json.loads((ROOT / 'evidence/frontier397-native-statistics-comparison.json').read_text())
    checks['hp']['independent_boundary_fidelity'] = boundary['complete'] and all(
        all(row.values()) for row in boundary['fidelity'].values())
    full_model_comparisons = []
    for step in [395, 396]:
        differences = {key: [name for name, row in models['hp'][step].items()
                            if row[key] != models['mf'][step][name][key]]
                       for key in ['fp32_sha256', 'bf16_as_fp32_sha256']}
        full_model_comparisons.append({'step': step, 'different': differences})
    states, links = {}, {}
    for step in [395, 396]:
        arrays = {}
        for family, run in [('hp', HP), ('mf', MF)]:
            report = json.loads((ROOT / 'runs' / run / ('FRONTIER_STATE-' + str(step) + '.json')).read_text())
            folder = ROOT / ('tools-f637' if family == 'hp' else 'tools')
            source = 'hp_frontier396_state_probe.py' if family == 'hp' else 'mf_frontier396_state_tap.py'
            checks[family]['source_' + str(step)] = report['source_sha256'] == digest(folder / source)
            checks[family]['readonly_' + str(step)] = report['read_only'] and not report['teacher_values_used']
            checks[family]['two_states_' + str(step)] = len(report['records']) == 2
            if family == 'mf':
                checks[family]['optimizer_step_' + str(step)] = report['optimizer_step'] == step
            for row in report['records']:
                array = load(STATE / run / ('state-' + str(step)), row)
                if array.shape != (5120, 3072):
                    raise ValueError('Unexpected shared-expert state layout')
                arrays[family, row['kind']] = array
            observed = model_record(TARGET, arrays[family, 'weight'])
            links[family + '_' + str(step)] = all(observed[key] == models[family][step][TARGET][key]
                for key in ['fp32_sha256', 'bf16_as_fp32_sha256'])
        states[step] = {kind: compare_arrays(arrays['hp', kind], arrays['mf', kind])
                        for kind in ['weight', 'muon_m']}
        del arrays
    native = json.loads((ROOT / 'runs' / MF / 'FRONTIER_GRADIENTS-396.json').read_text())
    checks['mf']['one_native_gradient'] = len(native) == 1 and native[0]['name'] == MF_TARGET and native[0]['retained']
    hp_gradient = json.loads((ROOT / 'runs' / HP / 'FRONTIER_GRADIENT-396.json').read_text())
    checks['hp']['gradient_source'] = hp_gradient['source_sha256'] == digest(ROOT / 'tools-f637/hp_frontier396_state_probe.py')
    checks['hp']['two_gradient_boundaries'] = len(hp_gradient['records']) == 2
    preclip_row = next(row for row in hp_gradient['records'] if row['kind'] == 'preclip')
    preclip = load(STATE / HP / 'gradient-396', preclip_row)
    independent_preclip = json.loads((ROOT / 'runs/hp-jte213-expertnorm328-preclip397/PRECLIP-396.json').read_text())
    expected_hash = next(row['native_dump_sha1'] for row in independent_preclip['records']
                         if row['name'] == MF_TARGET)
    checks['hp']['independent_preclip_native_hash'] = native_dump_sha1(preclip) == expected_hash
    retained = list((STATE / MF / 'gradients').glob('*.npy'))
    checks['mf']['earlier_gradient_values_cleaned'] = len(retained) == 1 and retained[0].name == native[0]['file']
    row = next(row for row in hp_gradient['records'] if row['kind'] == 'postclip')
    gradient = compare_arrays(load(STATE / HP / 'gradient-396', row), load(STATE / MF / 'gradients', native[0]))
    if not all(all(row.values()) for row in checks.values()) or not all(links.values()):
        raise ValueError({'fidelity': checks, 'model_hash_links': links})
    report = {'complete': True, 'fidelity': checks, 'model_hash_links': links, 'states': states,
              'clipped_gradient': gradient, 'full_model_comparisons': full_model_comparisons,
              'source_sha256': digest(Path(__file__)),
              'scope': 'One entire shared-expert down tensor (5120,3072), selected395/396 states and396 gradient; '
                       'both full-model boundaries linked to the independent native-statistic run. '
                       'MF and HP selected states use the same dense layout. '
                       'No conclusion on unobserved optimizer slots.'}
    (ROOT / 'evidence/frontier396-state-comparison.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == '__main__':
    main()
