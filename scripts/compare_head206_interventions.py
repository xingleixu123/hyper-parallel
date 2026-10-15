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

"""Audit the head/router factorial diagnostic without treating teacher patches as fixes."""
import hashlib
import json
from pathlib import Path

from compare_vocab_optimizer_trace import fidelity, read_rows
from compare_loss_components import read_metrics

ROOT = Path('/alignment/phase12')
RUNS = {
    'control': 'hp-jtf637-sharedup-modelhash226',
    'router': 'hp-jtf637-sharedup-router206diag226',
    'head': 'hp-jtf637-sharedup-head206diag226',
    'head_router': 'hp-jtf637-sharedup-headrouter206diag226',
}
KEYS = ['total_loss', 'lm_loss', 'mtp_weighted', 'mtp_raw', 'aux_weighted', 'aux_raw']
STEPS = list(range(201, 227))


def source_sha(name):
    """Read the exact current frozen diagnostic source."""
    return hashlib.sha256((ROOT / 'tools-f637' / name).read_bytes()).hexdigest()


def model_hashes(run):
    """Read complete named model records and reject duplicate tensors."""
    result = {}
    for row in read_rows(ROOT / 'runs' / run / 'MODEL_HASHES.jsonl'):
        records = {item['name']: item for item in row['records']}
        if len(records) != 48 or len(row['records']) != 48 or row['step'] in result:
            raise ValueError('Incomplete or duplicated model records: ' + run)
        result[row['step']] = records
    return result


def main():
    """Validate interventions, report actual magnitudes and signed interaction effects."""
    head = json.loads((ROOT / 'evidence/head206-model-comparison.json').read_text())
    previous = json.loads((ROOT / 'evidence/router206-causal-audit.json').read_text())
    checks = {'previous_router': previous['fidelity']}
    hashes = {family: model_hashes(run) for family, run in RUNS.items() if family != 'router'}
    hashes['mf'] = model_hashes('mf-modelhash226')
    for mode in ['head', 'head_router']:
        run = RUNS[mode]
        directory = ROOT / 'runs' / run
        check = fidelity(run, RUNS['control'], 26)
        check.pop('all_metrics_grad_norm_exact')
        check['exit_zero'] = (ROOT / 'runs' / (run + '.exit')).read_text().strip() == '0'
        control_rows = read_rows(ROOT / 'runs' / RUNS['control'] / 'metrics.jsonl')
        actual_rows = read_rows(directory / 'metrics.jsonl')
        strip = lambda rows: [{k: v for k, v in row.items() if k != 'wall_time'} for row in rows]
        check['first6_metrics_and_norm_exact_control'] = strip(control_rows[:6]) == strip(actual_rows[:6])
        marker = json.loads((directory / 'DIAGNOSTIC_ONLY.json').read_text())
        change = json.loads((directory / 'MODEL206_INTERVENTION.json').read_text())
        check['explicit_diagnostic_only'] = marker['diagnostic_only'] and not marker['eligible_for_acceptance']
        check['one_step206_mode'] = marker['step'] == change['step'] == 206 and marker['mode'] == change['mode'] == mode
        check['no_optimizer_intervention'] = not marker['optimizer_state_modified'] and not change['optimizer_state_modified']
        check['intervention_source'] = marker['source_sha256'] == source_sha('hp_head206_intervention.py')
        check['router_helper_source'] = marker['router_helper_sha256'] == source_sha('hp_router206_intervention.py')
        check['compact_patch'] = marker['patch_sha256'] == head['patch_sha256']
        expected = ['lm_head.weight'] + (['model.layers.1.mlp.gate.weight'] if mode == 'head_router' else [])
        check['only_requested_tensors'] = sorted(change['changed']) == sorted(expected)
        for name in expected:
            patch = change['changed'][name]
            count = head['different_fp32'] if name == 'lm_head.weight' else 12
            check[name + '_patch'] = patch['changed_elements'] == count and patch['other_elements_unchanged']
            check[name + '_before_hash'] = patch['before_sha256'] == hashes['control'][206][name]['fp32_sha256']
            check[name + '_after_hash'] = patch['after_sha256'] == hashes['mf'][206][name]['fp32_sha256']
        manifest = json.loads((directory / 'PROBE_SOURCE_MANIFEST.json').read_text())
        check['49_base_sources'] = len(manifest['files']) == 49 and all(
            row['sha256'] == source_sha(row['name']) for row in manifest['files'])
        observer = json.loads((directory / 'MODEL_HASH_OBSERVER.json').read_text())
        check['hash_observer_source'] = observer['source_sha256'] == source_sha('hp_frontier206_model_hash.py')
        check['hash_helper_source'] = observer['helper_sha256'] == source_sha('model_hash_stream.py')
        factor = json.loads((directory / 'SHARED_UP_FP32_FACTOR.json').read_text())
        check['shared_up_factor_source'] = factor['source_sha256'] == source_sha('hp_shared_up_fp32_factor.py')
        check['exact_hash_steps'] = list(hashes[mode]) == [206, 207]
        checks[mode] = check
    if not all(all(row.values()) for row in checks.values()):
        raise ValueError(checks)
    series = {family: read_metrics(ROOT / 'runs' / run / 'metrics.jsonl', 'hp', 1e-4, .3)
              for family, run in RUNS.items()}
    series['mf'] = read_metrics(ROOT / 'runs/mf-resume/metrics.jsonl', 'mf', 1e-4, .3)
    rows = []
    for step in STEPS:
        errors = {family: {key: abs(series[family][step][key] - series['mf'][step][key]) for key in KEYS}
                  for family in RUNS}
        interaction = {key: series['head_router'][step][key] - series['head'][step][key]
                      - series['router'][step][key] + series['control'][step][key] for key in KEYS}
        rows.append({'step': step, 'absolute_error': errors, 'signed_interaction': interaction,
                     'actual': {family: series[family][step] for family in series}})
    summary = {family: {key: {'max': max(row['absolute_error'][family][key] for row in rows),
                             'mean': sum(row['absolute_error'][family][key] for row in rows) / len(rows),
                             'bad_steps': [row['step'] for row in rows if row['absolute_error'][family][key] > .005]}
                       for key in KEYS} for family in RUNS}
    first_bad = {family: next((row['step'] for row in rows if max(row['absolute_error'][family].values()) > .005), None)
                 for family in RUNS}
    model_rows = []
    for step in [206, 207]:
        for family in ['control', 'head', 'head_router']:
            left, right = hashes[family][step], hashes['mf'][step]
            if set(left) != set(right):
                raise ValueError('Inconsistent model names')
            model_rows.append({'step': step, 'family': family, 'different_vs_mf': {
                key: [name for name in left if left[name][key] != right[name][key]]
                for key in ['fp32_sha256', 'bf16_as_fp32_sha256']}})
    report = {'complete': True, 'eligible_for_acceptance': False, 'fidelity': checks,
              'runs': RUNS, 'summary': summary, 'first_bad': first_bad, 'models': model_rows, 'rows': rows,
              'scope': 'One post206 teacher model patch per diagnostic. Optimizer slots are not patched; '
                       'all later updates run independently. This is a26-step causal diagnostic, not a product fix or200-step acceptance.'}
    (ROOT / 'evidence/head206-2x2-causal-audit.json').write_text(json.dumps(report, indent=2))
    compact = {key: report[key] for key in ['complete', 'first_bad', 'summary', 'models']}
    print(json.dumps(compact))


if __name__ == '__main__':
    main()
