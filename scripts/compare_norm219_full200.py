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

"""Audit independent position-reset and MTP-boundary factorial experiments."""
import hashlib
import json
from pathlib import Path

from compare_head206_interventions import KEYS, source_sha
from compare_loss_components import read_metrics
from compare_vocab_optimizer_trace import fidelity, read_rows

ROOT = Path('/alignment/phase12')
BASE = 'hp-jtf637-packed217-both400'
RUNS = {mode: 'hp-jtf637-norm219-' + mode + '400' for mode in ['dense', 'expert', 'both']}


def clean(row: dict) -> dict:
    """Discard time only; preserve all training metrics and gradient norm."""
    return {key: value for key, value in row.items() if key != 'wall_time'}


def main() -> None:
    """Require restoration, inputs, frozen sources and every live invocation."""
    checks = {}
    baseline = ROOT / 'runs' / BASE
    for mode, run in RUNS.items():
        directory = ROOT / 'runs' / run
        check = fidelity(run, BASE, 200)
        check.pop('all_metrics_grad_norm_exact')
        check['exit_zero'] = (ROOT / 'runs' / (run + '.exit')).read_text().strip() == '0'
        actual = read_rows(directory / 'metrics.jsonl')
        check['exact200_steps'] = [row['step'] for row in actual] == list(range(201, 401))
        check['first26_metrics_norm_exact'] = [clean(row) for row in actual[:26]] == [
            clean(row) for row in read_rows(ROOT / 'runs' / (run[:-3] + '226') / 'metrics.jsonl')]
        check['config_only_run_path'] = (ROOT / 'configs' / (BASE + '.yaml')).read_text().replace(
            BASE, run) == (ROOT / 'configs' / (run + '.yaml')).read_text()
        factor = json.loads((directory / 'NORM219_FACTOR.json').read_text())
        check['factor_source'] = factor['source_sha256'] == source_sha('hp_norm219_factor.py')
        check['factor_mode'] = factor['mode'] == mode
        check['no_teacher_or_step_special_case'] = not (
            factor['teacher_values_used'] or factor['coordinate_or_step_special_case'])
        expected = {'dense': 1 if mode in ['dense', 'both'] else 0,
                    'expert_pieces': 6 if mode in ['expert', 'both'] else 0}
        check['all200_live_calls'] = read_rows(directory / 'NORM219_CALLS.jsonl') == [
            {'step': step, 'calls': {key: value * (step - 200) for key, value in expected.items()}}
            for step in range(201, 401)]
        for name in ['SOURCE_REVISION.json', 'PROBE_SOURCE_MANIFEST.json', 'HEAD_GEMM_ROUTE.json',
                     'SHARED_UP_FP32_FACTOR.json', 'ADAMW_ARCHIVE_ARITHMETIC.json',
                     'Q_NS_ADJACENT_FACTOR.json', 'ROUTER_NS_NPU_SQRT_FACTOR.json',
                     'WEIGHTED_DVM_FACTOR.json', 'HEAD_GEMM_ROUTE_CALLS.jsonl',
                     'Q_NS_ADJACENT_CALLS.jsonl', 'ROUTER_NS_NPU_SQRT_CALLS.jsonl',
                     'WEIGHTED_DVM_CALLS.jsonl', 'PACKED_SEMANTICS_FACTOR.json', 'PACKED_SEMANTICS_CALLS.jsonl']:
            check['unchanged_' + name] = (directory / name).read_bytes() == (baseline / name).read_bytes()
        manifest = json.loads((directory / 'PROBE_SOURCE_MANIFEST.json').read_text())
        check['49_frozen_sources'] = len(manifest['files']) == 49 and all(
            row['sha256'] == source_sha(row['name']) for row in manifest['files'])
        launch = json.loads((ROOT / 'evidence' / (run + '-launch.json')).read_text())
        check['launch_config_hash'] = launch['config_sha256'] == hashlib.sha256(
            (ROOT / 'configs' / (run + '.yaml')).read_bytes()).hexdigest()
        check['launch_script_hash'] = launch['launcher_sha256'] == hashlib.sha256(
            (ROOT / ('run-' + run + '.sh')).read_bytes()).hexdigest()
        checks[mode] = check
    if not all(all(check.values()) for check in checks.values()):
        raise ValueError(checks)
    reference = read_metrics(ROOT / 'runs/mf-resume/metrics.jsonl', 'mf', 1e-4, .3)
    errors, summary, at219 = {}, {}, {}
    for label, run in {'control': BASE, **RUNS}.items():
        metrics = read_metrics(ROOT / 'runs' / run / 'metrics.jsonl', 'hp', 1e-4, .3)
        rows = [{'step': step, 'errors': {key: abs(metrics[step][key] - reference[step][key]) for key in KEYS}}
                for step in range(201, 401)]
        errors[label] = rows
        at219[label] = rows[18]['errors']
        summary[label] = {key: {
            'first': rows[0]['errors'][key], 'max': max(row['errors'][key] for row in rows),
            'max_step': max(rows, key=lambda row: row['errors'][key])['step'],
            'mean': sum(row['errors'][key] for row in rows) / 200,
            'bad_steps': [row['step'] for row in rows if row['errors'][key] > .005]}
            for key in KEYS}
    report = {'complete': True, 'fidelity': checks, 'step219': at219, 'summary': summary, 'rows': errors,
              'all_six_passed': {label: all(row['first'] <= .001 and not row['bad_steps'] for row in record.values()) for label, record in summary.items()},
              'first_bad': {label: next((row['step'] for row in values if max(row['errors'].values()) > .005), None) for label, values in errors.items()},
              'scope': 'Complete200 independent archive-compatibility factorial. Keeps previous packed compatibility; not the unmodified library recipe.',
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (ROOT / 'evidence/norm219-independent200-audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'fidelity': True, 'step219': at219, 'summary': summary}))


if __name__ == '__main__':
    main()
