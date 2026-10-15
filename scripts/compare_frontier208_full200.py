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

"""Audit all-step independent Q/router normalization factors and their interaction."""
import json
from pathlib import Path

import yaml
from compare_vocab_optimizer_trace import fidelity, read_rows
from compare_loss_components import read_metrics
from compare_head206_interventions import model_hashes, source_sha, KEYS

ROOT = Path('/alignment/phase12')
BASE = 'hp-jtf637-headrouter-control400'
SHORT_CONTROL = 'hp-jtf637-sharedup-headlegacynz-routerdvm226'
RUNS = {'control': BASE, **{mode: 'hp-jtf637-norm208-' + mode.replace('_', '-') + '-independent400'
                         for mode in ['q', 'q_router']}}


def main():
    """Verify no teacher values, full restores, all inputs, exact sources and actual factor calls."""
    checks, hashes = {}, {'mf': model_hashes('mf-modelhash226')}
    baseline = ROOT / 'runs' / SHORT_CONTROL
    base_config = yaml.safe_load((ROOT / 'configs' / (BASE + '.yaml')).read_text())
    for mode, run in RUNS.items():
        directory = ROOT / 'runs' / run
        check = fidelity(run, 'hp-jtf637-adamw-all-sharedupfp32400', 200)
        check.pop('all_metrics_grad_norm_exact')
        check['exit_zero'] = (ROOT / 'runs' / (run + '.exit')).read_text().strip() == '0'
        actual = read_rows(directory / 'metrics.jsonl')
        check['all200_steps'] = [row['step'] for row in actual] == list(range(201, 401))
        check['no_teacher_marker'] = not (directory / 'DIAGNOSTIC_ONLY.json').exists()
        for name in ['SOURCE_REVISION.json', 'HEAD_GEMM_ROUTE.json', 'SHARED_UP_FP32_FACTOR.json',
                     'ADAMW_ARCHIVE_ARITHMETIC.json']:
            check['unchanged_' + name] = json.loads((directory / name).read_text()) == json.loads((baseline / name).read_text())
        manifest = json.loads((directory / 'PROBE_SOURCE_MANIFEST.json').read_text())
        check['49_sources'] = len(manifest['files']) == 49 and all(row['sha256'] == source_sha(row['name']) for row in manifest['files'])
        head_rows = read_rows(directory / 'HEAD_GEMM_ROUTE_CALLS.jsonl')
        check['400_head_calls'] = [row['cumulative_backward_calls'] for row in head_rows] == list(range(2, 401, 2))
        expected_storage = read_rows(baseline / 'HEAD_GEMM_ROUTE_CALLS.jsonl')[0]['storage_metadata']
        check['all200_actual_storage'] = all(row['storage_metadata'] == expected_storage and row['mode'] == 'legacy_nz' for row in head_rows)
        router_name = 'ROUTER_NS_NPU_SQRT' if mode in ('router', 'q_router') else 'ROUTER_NS_DVM'
        router_source = 'router_ns_npu_sqrt_factor.py' if mode in ('router', 'q_router') else 'router_ns_dvm_factor.py'
        marker = json.loads((directory / (router_name + '_FACTOR.json')).read_text())
        check['router_factor_source'] = marker['source_sha256'] == source_sha(router_source)
        check['router_helper_source'] = marker['reduction_helper_sha256'] == source_sha('observe_hp_preclip_norm_adjacent.py')
        check['router_no_teacher'] = not marker['teacher_values_used']
        check['400_router_calls'] = read_rows(directory / (router_name + '_CALLS.jsonl')) == [
            {'step': step, 'total_router_calls': 2 * (step - 200)} for step in range(201, 401)]
        if mode in ('q', 'q_router'):
            marker = json.loads((directory / 'Q_NS_ADJACENT_FACTOR.json').read_text())
            check['q_source'] = marker['source_sha256'] == source_sha('muon_q_adjacent_factor.py')
            check['q_control_wrapper'] = marker['control_wrapper_sha256'] == source_sha('muon_q_native_tile_factor.py')
            check['q_half_tree_source'] = marker['half_tree_helper_sha256'] == source_sha('muon_q_norm_probe.py')
            check['q_no_teacher_or_special_case'] = not marker['teacher_values_used'] and not marker['coordinate_or_step_special_case']
            check['600_q_calls'] = read_rows(directory / 'Q_NS_ADJACENT_CALLS.jsonl') == [
                {'step': step, 'total_q_calls': 3 * (step - 200)} for step in range(201, 401)]
        else:
            check['q_factor_not_installed'] = not (directory / 'Q_NS_ADJACENT_FACTOR.json').exists()
        observer = json.loads((directory / 'MODEL_HASH_OBSERVER.json').read_text())
        check['observer_source'] = observer['source_sha256'] == source_sha('hp_frontier208_model_hash.py' if mode == 'control' else 'hp_normalization208_model_hash.py')
        check['observer_helper'] = observer['helper_sha256'] == source_sha('model_hash_stream.py')
        config = yaml.safe_load((ROOT / 'configs' / (run + '.yaml')).read_text())
        config['checkpoint']['checkpoint_dir'] = base_config['checkpoint']['checkpoint_dir']
        check['same_recipe'] = config == base_config
        hashes[mode] = model_hashes(run)
        check['model_hash_steps'] = list(hashes[mode]) == ([208, 209, 210] if mode == 'control' else [201, 207, 208, 209, 210])
        short_run = SHORT_CONTROL if mode == 'control' else run.replace('400', '226')
        short_rows = read_rows(ROOT / 'runs' / short_run / 'metrics.jsonl')
        strip = lambda rows: [{k: v for k, v in row.items() if k != 'wall_time'} for row in rows]
        check['first26_exact_short_metrics_norm'] = strip(actual[:26]) == strip(short_rows)
        checks[mode] = check
    if not all(all(check.values()) for check in checks.values()):
        raise ValueError(checks)
    series = {mode: read_metrics(ROOT / 'runs' / run / 'metrics.jsonl', 'hp', 1e-4, .3) for mode, run in RUNS.items()}
    series['mf'] = read_metrics(ROOT / 'runs/mf-resume/metrics.jsonl', 'mf', 1e-4, .3)
    rows = [{'step': step, 'absolute_error': {
        mode: {key: abs(series[mode][step][key] - series['mf'][step][key]) for key in KEYS} for mode in RUNS},
        'actual': {mode: series[mode][step] for mode in series}} for step in range(201, 401)]
    summary = {mode: {key: {'first': rows[0]['absolute_error'][mode][key],
                           'max': max(row['absolute_error'][mode][key] for row in rows),
                           'mean': sum(row['absolute_error'][mode][key] for row in rows) / 200,
                           'bad_steps': [row['step'] for row in rows if row['absolute_error'][mode][key] > .005]}
                     for key in KEYS} for mode in RUNS}
    first_bad = {mode: next((row['step'] for row in rows if max(row['absolute_error'][mode].values()) > .005), None)
                 for mode in RUNS}
    models = [{'mode': mode, 'step': step, 'different_vs_mf': {key: [
        name for name, record in hashes[mode][step].items() if record[key] != hashes['mf'][step][name][key]]
        for key in ['fp32_sha256', 'bf16_as_fp32_sha256']}}
        for mode in ['q', 'q_router'] for step in [201, 207, 208, 209, 210]]
    report = {'complete': True, 'independent': True, 'acceptance_complete': any(first_bad[mode] is None and all(summary[mode][key]['first'] <= .001 for key in KEYS) for mode in RUNS), 'completed_independent_full200_after_this_batch': 79,
              'fidelity': checks, 'runs': RUNS, 'summary': summary, 'first_bad': first_bad, 'models': models,
              'rows': rows, 'scope': '200-step independent control, Q and Q+router factors; no recorded future tensors. '
              'Archive compatibility is distinct from universal numerical correctness.'}
    (ROOT / 'evidence/frontier208-independent200-audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({key: report[key] for key in ['complete', 'first_bad', 'summary']}))


if __name__ == '__main__':
    main()
