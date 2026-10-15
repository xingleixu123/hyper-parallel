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

"""Audit the208 Q/router factorial diagnostics against the independent combination control."""
import hashlib
import json
from pathlib import Path

import yaml

from compare_vocab_optimizer_trace import fidelity, read_rows
from compare_loss_components import read_metrics
from compare_head206_interventions import model_hashes, source_sha, KEYS

ROOT = Path('/alignment/phase12')
BASE = 'hp-jtf637-sharedup-headlegacynz-routerdvm226'
RUNS = {'control': BASE, **{mode: 'hp-jtf637-frontier208-' + mode.replace('_', '-') + '-teacher226'
                         for mode in ['q', 'router', 'q_router']}}
TARGETS = {'q': ('qkv', 'model.layers.0.self_attn.linear_qkv.weight', 4477438),
           'router': ('router', 'model.layers.1.mlp.gate.weight', 15833)}


def main():
    """Verify exact pre-intervention reproduction, patch scope, states, source and every input."""
    comparison_path = ROOT / 'evidence/frontier208-gradient-state-comparison.json'
    comparison = json.loads(comparison_path.read_text())
    checks = {}
    hashes = {'control': model_hashes('hp-jtf637-headrouter-frontier210'), 'mf': model_hashes('mf-modelhash226')}
    baseline = ROOT / 'runs' / BASE
    base_config = yaml.safe_load((ROOT / 'configs' / (BASE + '.yaml')).read_text())
    for mode, run in RUNS.items():
        if mode == 'control':
            continue
        directory = ROOT / 'runs' / run
        check = fidelity(run, BASE, 26)
        check.pop('all_metrics_grad_norm_exact')
        check['exit_zero'] = (ROOT / 'runs' / (run + '.exit')).read_text().strip() == '0'
        strip = lambda rows: [{k: v for k, v in row.items() if k != 'wall_time'} for row in rows]
        actual = read_rows(directory / 'metrics.jsonl')
        control = read_rows(baseline / 'metrics.jsonl')
        check['all26_steps'] = [row['step'] for row in actual] == list(range(201, 227))
        check['first8_metrics_norm_exact'] = strip(actual[:8]) == strip(control[:8])
        marker = json.loads((directory / 'DIAGNOSTIC_ONLY.json').read_text())
        change = json.loads((directory / 'MODEL208_INTERVENTION.json').read_text())
        check['diagnostic_only'] = marker['diagnostic_only'] and not marker['eligible_for_acceptance']
        check['mode_and_step'] = marker['mode'] == change['mode'] == mode and marker['step'] == change['step'] == 208
        check['optimizer_not_modified'] = not marker['optimizer_state_modified'] and not change['optimizer_state_modified']
        check['intervention_source'] = marker['source_sha256'] == source_sha('hp_frontier208_intervention.py')
        check['patch_helper'] = marker['patch_helper_sha256'] == source_sha('hp_head206_intervention.py')
        check['comparison_source'] = marker['comparison_sha256'] == hashlib.sha256(comparison_path.read_bytes()).hexdigest()
        selected = ['q', 'router'] if mode == 'q_router' else [mode]
        names = [TARGETS[label][1] for label in selected]
        check['only_requested_parameters'] = sorted(change['changed']) == sorted(names)
        for label in selected:
            key, name, count = TARGETS[label]
            observed = comparison['states']['208'][key]
            patch = change['changed'][name]
            check[label + '_patch'] = patch['changed_elements'] == count and patch['other_elements_unchanged']
            check[label + '_before'] = patch['before_sha256'] == observed['weight']['hp_sha256']
            check[label + '_after'] = patch['after_sha256'] == observed['weight']['mf_sha256']
            check[label + '_momentum_unchanged_and_exact_mf'] = observed['muon_m']['exact'] and (
                patch['momentum_sha256_before_after'] == observed['muon_m']['hp_sha256'])
        for name in ['SOURCE_REVISION.json', 'HEAD_GEMM_ROUTE.json', 'ROUTER_NS_DVM_FACTOR.json',
                     'SHARED_UP_FP32_FACTOR.json', 'ADAMW_ARCHIVE_ARITHMETIC.json']:
            check['unchanged_' + name] = json.loads((directory / name).read_text()) == json.loads((baseline / name).read_text())
        manifest = json.loads((directory / 'PROBE_SOURCE_MANIFEST.json').read_text())
        check['49_sources'] = len(manifest['files']) == 49 and all(row['sha256'] == source_sha(row['name']) for row in manifest['files'])
        for name in ['HEAD_GEMM_ROUTE_CALLS.jsonl', 'ROUTER_NS_DVM_CALLS.jsonl']:
            check[name] = read_rows(directory / name) == read_rows(baseline / name)
        observer = json.loads((directory / 'MODEL_HASH_OBSERVER.json').read_text())
        check['observer_source'] = observer['source_sha256'] == source_sha('hp_frontier208_model_hash.py')
        check['observer_helper'] = observer['helper_sha256'] == source_sha('model_hash_stream.py')
        config = yaml.safe_load((ROOT / 'configs' / (run + '.yaml')).read_text())
        config['checkpoint']['checkpoint_dir'] = base_config['checkpoint']['checkpoint_dir']
        check['same_recipe'] = config == base_config
        hashes[mode] = model_hashes(run)
        check['model_hash_steps'] = list(hashes[mode]) == [208, 209, 210]
        check['all48_state208_exact_declared_scope'] = all(
            hashes[mode][208][name] == hashes['mf' if name in names else 'control'][208][name]
            for name in hashes['control'][208])
        checks[mode] = check
    if not all(all(check.values()) for check in checks.values()):
        raise ValueError(checks)
    series = {mode: read_metrics(ROOT / 'runs' / run / 'metrics.jsonl', 'hp', 1e-4, .3) for mode, run in RUNS.items()}
    series['mf'] = read_metrics(ROOT / 'runs/mf-resume/metrics.jsonl', 'mf', 1e-4, .3)
    rows = [{'step': step, 'absolute_error': {
        mode: {key: abs(series[mode][step][key] - series['mf'][step][key]) for key in KEYS} for mode in RUNS},
        'signed_interaction': {key: series['q_router'][step][key] - series['q'][step][key]
                               - series['router'][step][key] + series['control'][step][key] for key in KEYS},
        'actual': {mode: series[mode][step] for mode in series}} for step in range(201, 227)]
    summary = {mode: {key: {'first': rows[0]['absolute_error'][mode][key],
                           'max': max(row['absolute_error'][mode][key] for row in rows),
                           'mean': sum(row['absolute_error'][mode][key] for row in rows) / 26,
                           'bad_steps': [row['step'] for row in rows if row['absolute_error'][mode][key] > .005]}
                     for key in KEYS} for mode in RUNS}
    first_bad = {mode: next((row['step'] for row in rows if max(row['absolute_error'][mode].values()) > .005), None)
                 for mode in RUNS}
    models = [{'mode': mode, 'step': step, 'different_vs_mf': {key: [
        name for name, record in hashes[mode][step].items() if record[key] != hashes['mf'][step][name][key]]
        for key in ['fp32_sha256', 'bf16_as_fp32_sha256']}}
        for mode in RUNS for step in [208, 209, 210]]
    report = {'complete': True, 'eligible_for_acceptance': False, 'full200_count_unchanged': 76,
              'fidelity': checks, 'runs': RUNS, 'summary': summary, 'first_bad': first_bad, 'models': models,
              'rows': rows, 'scope': 'One post208 teacher weight patch; actual equal Muon buffers untouched. '
              '26-step causal diagnostics, not independent acceptance or a product fix.'}
    (ROOT / 'evidence/frontier208-2x2-causal-audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({key: report[key] for key in ['complete', 'first_bad', 'summary', 'models']}))


if __name__ == '__main__':
    main()
