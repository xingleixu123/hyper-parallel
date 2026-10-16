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

"""Locate the sole306 main-layer0 Q-up divergence in gradient, momentum or weight update."""
import hashlib
import json
from pathlib import Path

from compare_frontier219_state import load
from compare_preclip217 import compare_arrays
from compare_vocab_optimizer_trace import fidelity, read_rows
from model_hash_stream import model_record

ROOT = Path('/alignment/phase12')
STATE = Path('/state/jt-recipe-20261014')
HP = 'hp-jte213-fc1norm-state306'
MF = 'mf-frontier306-state'
BASE = 'hp-jte213-fc1norm400'
TARGET = 'model.layers.0.self_attn.q_b_proj.weight'
MF_TARGET = 'decoder.layers.0.self_attention.linear_q_up_proj.weight'


def digest(path: Path) -> str:
    """Hash bounded source or metadata files."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Require full trajectory fidelity and link sampled states to independent model hashes."""
    checks = {'hp': fidelity(HP, BASE, 106), 'mf': fidelity(MF, 'mf-resume', 106)}
    for family, run in [('hp', HP), ('mf', MF)]:
        checks[family]['exit_zero'] = (ROOT / 'runs' / (run + '.exit')).read_text().strip() == '0'
        original = BASE if family == 'hp' else 'mf-frontier258-state'
        before, after = ('  train_iters: 400', '  train_iters: 306') if family == 'hp' else (
            '  stop_step: 58', '  stop_step: 106')
        checks[family]['only_window_and_paths'] = (ROOT / 'configs' / (original + '.yaml')).read_text().replace(
            original, run).replace(before, after) == (ROOT / 'configs' / (run + '.yaml')).read_text()
        launch = json.loads((ROOT / 'evidence' / (run + '-launch.json')).read_text())
        checks[family]['config_sha'] = launch['config_sha256'] == digest(ROOT / 'configs' / (run + '.yaml'))
        checks[family]['launcher_sha'] = launch['launcher_sha256'] == digest(ROOT / ('run-' + run + '.sh'))
        folder = ROOT / ('tools-f637' if family == 'hp' else 'tools')
        entry_folder = ROOT / 'tools-e213' if family == 'hp' else folder
        checks[family]['entry_sha'] = launch['entry_sha256'] == digest(entry_folder / ('capture_' + family + '_frontier306_state.py'))
    for filename in ['SOURCE_REVISION.json', 'PROBE_SOURCE_MANIFEST.json', 'HEAD_GEMM_ROUTE.json',
                     'SHARED_UP_FP32_FACTOR.json', 'ADAMW_ARCHIVE_ARITHMETIC.json', 'Q_NS_ADJACENT_FACTOR.json',
                     'ROUTER_NS_NPU_SQRT_FACTOR.json', 'WEIGHTED_DVM_FACTOR.json', 'PACKED_SEMANTICS_FACTOR.json',
                     'NORM219_FACTOR.json', 'Q_NPU_SQRT222_FACTOR.json', 'CE_TAIL226_FACTOR.json', 'NS_EPSILON229_FACTOR.json', 'GLOBAL_NPU_SQRT235_FACTOR.json', 'QUP258_FACTOR.json', 'FC1293_FACTOR.json']:
        checks['hp']['unchanged_' + filename] = (ROOT / 'runs' / HP / filename).read_bytes() == (
            ROOT / 'runs' / BASE / filename).read_bytes()
    for filename in ['HEAD_GEMM_ROUTE_CALLS.jsonl', 'Q_NS_ADJACENT_CALLS.jsonl', 'ROUTER_NS_NPU_SQRT_CALLS.jsonl',
                     'WEIGHTED_DVM_CALLS.jsonl', 'PACKED_SEMANTICS_CALLS.jsonl', 'NORM219_CALLS.jsonl',
                     'CLIP_REDUCTION.jsonl', 'CE_TAIL226_CALLS.jsonl', 'QUP258_CALLS.jsonl', 'FC1293_CALLS.jsonl']:
        checks['hp']['same106_' + filename] = read_rows(ROOT / 'runs' / HP / filename) == read_rows(
            ROOT / 'runs' / BASE / filename)[:106]
    manifest = json.loads((ROOT / 'runs' / HP / 'PROBE_SOURCE_MANIFEST.json').read_text())
    checks['hp']['49_frozen_sources'] = len(manifest['files']) == 49 and all(
        row['sha256'] == digest(ROOT / 'tools-f637' / row['name']) for row in manifest['files'])
    # The attempted305/306 full-model observer had an empty range; use independently
    # captured306 model hashes and do not claim a full-model305 observation.
    models = {}
    for family, prior_run in [('hp', 'hp-jte213-fc1norm-preclip307'), ('mf', 'mf-native-stat307')]:
        prior = next(row for row in read_rows(ROOT / 'runs' / prior_run / 'MODEL_HASHES.jsonl')
                     if row['step'] == 306)
        models[family] = {306: {item['name']: item for item in prior['records']}}
        checks[family]['independent306_model_complete'] = len(models[family][306]) == 48
        folder = ROOT / ('tools-f637' if family == 'hp' else 'tools')
        bad_probe = (folder / (family + '_tail306_model_hash.py')).read_text()
        checks[family]['recorded_empty_observer_window'] = 'not in range(305, 259)' in bad_probe
        checks[family]['missing_model_observer_recorded'] = not (
            ROOT / 'runs' / (HP if family == 'hp' else MF) / 'MODEL_HASHES.jsonl').exists()
    boundary = json.loads((ROOT / 'evidence/frontier307-native-statistics-comparison.json').read_text())
    checks['hp']['independent_boundary_fidelity'] = boundary['complete'] and all(
        all(row.values()) for row in boundary['fidelity'].values())
    full_model_comparisons = boundary['models'][:1]
    states, links = {}, {}
    for step in [305, 306]:
        arrays = {}
        for family, run in [('hp', HP), ('mf', MF)]:
            report = json.loads((ROOT / 'runs' / run / ('FRONTIER_STATE-' + str(step) + '.json')).read_text())
            folder = ROOT / ('tools-f637' if family == 'hp' else 'tools')
            source = 'hp_frontier306_state_probe.py' if family == 'hp' else 'mf_frontier306_state_tap.py'
            checks[family]['source_' + str(step)] = report['source_sha256'] == digest(folder / source)
            checks[family]['readonly_' + str(step)] = report['read_only'] and not report['teacher_values_used']
            checks[family]['two_states_' + str(step)] = len(report['records']) == 2
            if family == 'mf':
                checks[family]['optimizer_step_' + str(step)] = report['optimizer_step'] == step
            for row in report['records']:
                arrays[family, row['kind']] = load(STATE / run / ('state-' + str(step)), row)
            if step == 306:
                observed = model_record(TARGET, arrays[family, 'weight'])
                links[family + '_' + str(step)] = all(observed[key] == models[family][step][TARGET][key]
                    for key in ['fp32_sha256', 'bf16_as_fp32_sha256'])
        states[step] = {kind: compare_arrays(arrays['hp', kind], arrays['mf', kind])
                        for kind in ['weight', 'muon_m']}
        del arrays
    native = json.loads((ROOT / 'runs' / MF / 'FRONTIER_GRADIENTS-306.json').read_text())
    checks['mf']['one_native_gradient'] = len(native) == 1 and native[0]['name'] == MF_TARGET and native[0]['retained']
    hp_gradient = json.loads((ROOT / 'runs' / HP / 'FRONTIER_GRADIENT-306.json').read_text())
    checks['hp']['gradient_source'] = hp_gradient['source_sha256'] == digest(ROOT / 'tools-f637/hp_frontier306_state_probe.py')
    checks['hp']['two_gradient_boundaries'] = len(hp_gradient['records']) == 2
    row = next(row for row in hp_gradient['records'] if row['kind'] == 'postclip')
    gradient = compare_arrays(load(STATE / HP / 'gradient-306', row), load(STATE / MF / 'gradients', native[0]))
    if not all(all(row.values()) for row in checks.values()) or not all(links.values()):
        raise ValueError({'fidelity': checks, 'model_hash_links': links})
    report = {'complete': True, 'fidelity': checks, 'model_hash_links': links, 'states': states,
              'clipped_gradient': gradient, 'full_model_comparisons': full_model_comparisons,
              'observation_limitation': 'Full-model305 not captured; empty range(305,259) retained with original failure log.', 'source_sha256': digest(Path(__file__)),
              'scope': 'One entire main-layer0 Q-up tensor; exact independent model links at306; '
                       'full-model305 was not captured because its observer range was empty. '
                       'Selected305 states are compared directly;306 model links come from the independent native boundary run. '
                       'No conclusion on other optimizer states.'}
    (ROOT / 'evidence/frontier306-state-comparison.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == '__main__':
    main()
