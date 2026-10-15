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

"""Audit two complete independent weighted-reduction runs against the frozen200-step control."""
import hashlib
import json
from pathlib import Path

from compare_head206_interventions import KEYS, source_sha
from compare_loss_components import read_metrics
from compare_vocab_optimizer_trace import fidelity, read_rows

ROOT = Path('/alignment/phase12')
BASE = 'hp-jtf637-norm208-q-router-independent400'
RUNS = {scope: 'hp-jtf637-weighted-dvm-' + scope + '400' for scope in ['main', 'both']}


def main() -> None:
    """Require all200 steps, source/config/restore/input checks and exact short-window prefixes."""
    checks = {}
    summaries, error_rows = {}, {}
    reference = read_metrics(ROOT / 'runs/mf-resume/metrics.jsonl', 'mf', 1e-4, .3)
    clean = lambda row: {key: value for key, value in row.items() if key != 'wall_time'}
    for scope, run in RUNS.items():
        directory, baseline = ROOT / 'runs' / run, ROOT / 'runs' / BASE
        if (ROOT / 'runs' / (run + '.exit')).read_text().strip() != '0':
            raise ValueError('Full run is incomplete or failed')
        check = fidelity(run, BASE, 200)
        check.pop('all_metrics_grad_norm_exact')
        check['exit_zero'] = True
        check['no_diagnostic_marker'] = not (directory / 'DIAGNOSTIC_ONLY.json').exists()
        short = ROOT / 'runs' / ('hp-jtf637-weighted-dvm-' + scope + '226')
        actual = read_rows(directory / 'metrics.jsonl')
        check['all200_steps'] = [row['step'] for row in actual] == list(range(201, 401))
        check['first26_metrics_norm_exact'] = [clean(row) for row in actual[:26]] == [
            clean(row) for row in read_rows(short / 'metrics.jsonl')]
        check['same_recipe_except_output_path'] = (ROOT / 'configs' / (BASE + '.yaml')).read_text().replace(
            BASE, run) == (ROOT / 'configs' / (run + '.yaml')).read_text()
        factor = json.loads((directory / 'WEIGHTED_DVM_FACTOR.json').read_text())
        check['factor_same_as_short'] = factor == json.loads((short / 'WEIGHTED_DVM_FACTOR.json').read_text())
        check['scope'] = factor['scope'] == scope
        check['no_teacher_or_step_special_case'] = not factor['teacher_values_used'] and not factor['coordinate_or_step_special_case']
        check['factor_source'] = factor['source_sha256'] == source_sha('hp_weighted_dvm_factor.py')
        check['half_tree_helper_source'] = factor['helper_sha256'] == source_sha('muon_q_norm_probe.py')
        launch = json.loads((ROOT / 'evidence' / (run + '-launch.json')).read_text())
        check['full_entry_source'] = launch['entry_sha256'] == source_sha('capture_hp_weighted_dvm_full.py')
        check['launch_config_source'] = launch['config_sha256'] == hashlib.sha256(
            (ROOT / 'configs' / (run + '.yaml')).read_bytes()).hexdigest()
        check['all200_live_calls'] = read_rows(directory / 'WEIGHTED_DVM_CALLS.jsonl') == [
            {'step': step, 'calls': {'main': step - 200, 'mtp': step - 200 if scope == 'both' else 0}}
            for step in range(201, 401)]
        for name in ['SOURCE_REVISION.json', 'PROBE_SOURCE_MANIFEST.json', 'HEAD_GEMM_ROUTE.json',
                     'SHARED_UP_FP32_FACTOR.json', 'ADAMW_ARCHIVE_ARITHMETIC.json',
                     'Q_NS_ADJACENT_FACTOR.json', 'ROUTER_NS_NPU_SQRT_FACTOR.json',
                     'HEAD_GEMM_ROUTE_CALLS.jsonl', 'Q_NS_ADJACENT_CALLS.jsonl', 'ROUTER_NS_NPU_SQRT_CALLS.jsonl']:
            check['unchanged_' + name] = (directory / name).read_bytes() == (baseline / name).read_bytes()
        manifest = json.loads((directory / 'PROBE_SOURCE_MANIFEST.json').read_text())
        check['49_frozen_sources'] = len(manifest['files']) == 49 and all(
            row['sha256'] == source_sha(row['name']) for row in manifest['files'])
        checks[scope] = check
    if not all(all(row.values()) for row in checks.values()):
        raise ValueError(checks)
    for label, run in {'control': BASE, **RUNS}.items():
        metrics = read_metrics(ROOT / 'runs' / run / 'metrics.jsonl', 'hp', 1e-4, .3)
        if sorted(metrics) != list(range(201, 401)):
            raise ValueError('Complete200-step metric coverage required')
        rows = [{'step': step, 'errors': {key: abs(metrics[step][key] - reference[step][key]) for key in KEYS}}
                for step in range(201, 401)]
        error_rows[label] = rows
        summaries[label] = {key: {'first': rows[0]['errors'][key], 'last': rows[-1]['errors'][key],
                                   'max': max(row['errors'][key] for row in rows),
                                   'max_step': max(rows, key=lambda row: row['errors'][key])['step'],
                                   'mean': sum(row['errors'][key] for row in rows) / 200,
                                   'bad_steps': [row['step'] for row in rows if row['errors'][key] > .005],
                                   'first_under_strict_0005': rows[0]['errors'][key] <= .0005,
                                   'passed': rows[0]['errors'][key] <= .001 and all(
                                       row['errors'][key] <= .005 for row in rows)} for key in KEYS}
    same_trajectory = [clean(row) for row in read_rows(ROOT / 'runs' / RUNS['main'] / 'metrics.jsonl')] == [
        clean(row) for row in read_rows(ROOT / 'runs' / RUNS['both'] / 'metrics.jsonl')]
    report = {'complete': True, 'independent': True, 'fidelity': checks, 'summary': summaries, 'rows': error_rows,
              'main_both_all200_metrics_and_grad_norm_exact': same_trajectory,
              'all_six_passed': {label: all(row['passed'] for row in summary.values())
                                 for label, summary in summaries.items()},
              'first_bad': {label: next((row['step'] for row in rows if max(row['errors'].values()) > .005), None)
                            for label, rows in error_rows.items()},
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'scope': 'Complete independent201-400; all six criteria, exact26 prefixes; no observer performance claims.'}
    (ROOT / 'evidence/weighted212-independent200-audit.json').write_text(json.dumps(report, indent=2))
    compact = {label: {key: {name: value for name, value in row.items() if name != 'bad_steps'} | {
        'bad_step_count': len(row['bad_steps'])} for key, row in summary.items()}
               for label, summary in summaries.items()}
    print(json.dumps({'complete': True, 'all_six_passed': report['all_six_passed'],
                      'first_bad': report['first_bad'], 'summary': compact}))


if __name__ == '__main__':
    main()
