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

"""Audit independent weighted-reduction scopes and retain all six loss/error outcomes."""
import hashlib
import json
from pathlib import Path

from compare_head206_interventions import KEYS, model_hashes, source_sha
from compare_loss_components import read_metrics
from compare_vocab_optimizer_trace import fidelity, read_rows

ROOT = Path('/alignment/phase12')
BASE = 'hp-jtf637-norm208-q-router-independent226'
RUNS = {scope: 'hp-jtf637-weighted-dvm-' + scope + '226' for scope in ['main', 'both']}


def main() -> None:
    checks = {}
    model_differences, preclip, summaries, error_rows = {}, {}, {}, {}
    reference_models = model_hashes('mf-modelhash226')
    reference_metrics = read_metrics(ROOT / 'runs/mf-resume/metrics.jsonl', 'mf', 1e-4, .3)
    for scope, run in RUNS.items():
        directory, baseline = ROOT / 'runs' / run, ROOT / 'runs' / BASE
        check = fidelity(run, BASE, 26)
        check.pop('all_metrics_grad_norm_exact')
        check['exit_zero'] = (ROOT / 'runs' / (run + '.exit')).read_text().strip() == '0'
        check['no_diagnostic_marker'] = not (directory / 'DIAGNOSTIC_ONLY.json').exists()
        metadata = json.loads((directory / 'WEIGHTED_DVM_FACTOR.json').read_text())
        check['scope_exact'] = metadata['scope'] == scope
        check['no_teacher_or_step_special_case'] = not metadata['teacher_values_used'] and not metadata['coordinate_or_step_special_case']
        check['factor_source'] = metadata['source_sha256'] == source_sha('hp_weighted_dvm_factor.py')
        check['half_tree_helper'] = metadata['helper_sha256'] == source_sha('muon_q_norm_probe.py')
        calls = read_rows(directory / 'WEIGHTED_DVM_CALLS.jsonl')
        check['all26_live_calls'] = calls == [{'step': step, 'calls': {
            'main': step - 200, 'mtp': step - 200 if scope == 'both' else 0}} for step in range(201, 227)]
        for name in ['SOURCE_REVISION.json', 'PROBE_SOURCE_MANIFEST.json', 'HEAD_GEMM_ROUTE.json',
                     'SHARED_UP_FP32_FACTOR.json', 'ADAMW_ARCHIVE_ARITHMETIC.json',
                     'Q_NS_ADJACENT_FACTOR.json', 'ROUTER_NS_NPU_SQRT_FACTOR.json',
                     'HEAD_GEMM_ROUTE_CALLS.jsonl', 'Q_NS_ADJACENT_CALLS.jsonl', 'ROUTER_NS_NPU_SQRT_CALLS.jsonl']:
            check['unchanged_' + name] = (directory / name).read_bytes() == (baseline / name).read_bytes()
        manifest = json.loads((directory / 'PROBE_SOURCE_MANIFEST.json').read_text())
        check['49_frozen_sources'] = len(manifest['files']) == 49 and all(
            row['sha256'] == source_sha(row['name']) for row in manifest['files'])
        models = model_hashes(run)
        check['all26_model_observations'] = list(models) == list(range(201, 227))
        observer = json.loads((directory / 'MODEL_HASH_OBSERVER.json').read_text())
        check['model_observer_source'] = observer['source_sha256'] == source_sha('hp_weighted212_model_hash.py')
        check['model_hash_helper_source'] = observer['helper_sha256'] == source_sha('model_hash_stream.py')
        model_differences[scope] = {step: {key: [name for name, row in model.items()
                                                         if row[key] != reference_models[step][name][key]]
                                                 for key in ['fp32_sha256', 'bf16_as_fp32_sha256']}
                                    for step, model in models.items()}
        preclip[scope] = {}
        for step in [211, 212]:
            current = json.loads((directory / ('PRECLIP-' + str(step) + '.json')).read_text())
            reference = json.loads((ROOT / 'runs/mf-preclip212' / ('PRECLIP-' + str(step) + '.json')).read_text())
            expected = {row['name']: row['sha256'] for row in reference['records']}
            check['preclip_source_' + str(step)] = current['source_sha256'] == source_sha('hp_preclip212_hashes.py')
            check['all46_preclip_' + str(step)] = len(current['records']) == 46
            preclip[scope][step] = {'exact': sum(row['sha256'] == expected[row['name']] for row in current['records']),
                                    'norm': current['norm']}
        checks[scope] = check
    if not all(all(check.values()) for check in checks.values()):
        raise ValueError(checks)
    for label, run in {'control': BASE, **RUNS}.items():
        metrics = read_metrics(ROOT / 'runs' / run / 'metrics.jsonl', 'hp', 1e-4, .3)
        if sorted(metrics) != list(range(201, 227)):
            raise ValueError('Expected all26 independent steps')
        rows = [{'step': step, 'errors': {key: abs(metrics[step][key] - reference_metrics[step][key]) for key in KEYS}}
                for step in range(201, 227)]
        error_rows[label] = rows
        summaries[label] = {key: {'first': rows[0]['errors'][key],
                                   'max': max(row['errors'][key] for row in rows),
                                   'mean': sum(row['errors'][key] for row in rows) / 26,
                                   'bad_steps': [row['step'] for row in rows if row['errors'][key] > .005]}
                            for key in KEYS}
    clean = lambda row: {key: value for key, value in row.items() if key != 'wall_time'}
    metrics_rows = {label: [clean(row) for row in read_rows(ROOT / 'runs' / run / 'metrics.jsonl')]
                    for label, run in {**RUNS, 'teacher': 'hp-jtf637-router212-dy-teacher226'}.items()}
    main_models = model_hashes(RUNS['main'])
    teacher_models = model_hashes('hp-jtf637-router212-dy-teacher226')
    equivalence = {'main_both_all26_metrics_norm_exact': metrics_rows['main'] == metrics_rows['both'],
                   'main_both_all26_model_hashes_exact': main_models == model_hashes(RUNS['both']),
                   'main_teacher_all26_metrics_norm_exact': metrics_rows['main'] == metrics_rows['teacher'],
                   'main_teacher_observed4_model_hashes_exact': all(main_models[step] == model
                                                                    for step, model in teacher_models.items())}
    report = {'complete': True, 'independent': True, 'fidelity': checks, 'models': model_differences,
              'preclip': preclip, 'summary': summaries, 'rows': error_rows, 'equivalence': equivalence,
              'first_bad': {label: next((row['step'] for row in rows if max(row['errors'].values()) > .005), None)
                            for label, rows in error_rows.items()},
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'scope': 'Two independent26-step scope ablations; all full200 requirements still pending.'}
    (ROOT / 'evidence/weighted212-independent26-audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'complete': True, 'first_bad': report['first_bad'], 'preclip': preclip,
                      'first_model_difference': {scope: next((step for step, row in models.items()
                                                                 if row['fp32_sha256']), None)
                                                 for scope, models in model_differences.items()},
                      'summary': summaries}))


if __name__ == '__main__':
    main()
