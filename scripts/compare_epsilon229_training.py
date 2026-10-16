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

"""Audit one independent epsilon factor against its unchanged CE-tail control."""
import hashlib
import json
from pathlib import Path
import sys

from compare_head206_interventions import KEYS, source_sha
from compare_loss_components import read_metrics
from compare_vocab_optimizer_trace import fidelity, read_rows

ROOT = Path('/alignment/phase12')
EXPECTED_CALLS = {'(12288, 5120)': 2, '(16384, 1536)': 3, '(8192, 1536)': 3,
                  '(16384, 512)': 6, '(3072, 5120)': 4, '(5120, 3072)': 2,
                  '(512, 5120)': 3, '(64, 5120)': 3}


def digest(path: Path) -> str:
    """Hash a source or provenance file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_comparison(run: str) -> list:
    """Compare all model states against the two audited MF observation windows."""
    actual = read_rows(ROOT / 'runs' / run / 'MODEL_HASHES.jsonl')
    reference = read_rows(ROOT / 'runs/mf-modelhash226/MODEL_HASHES.jsonl')
    reference += read_rows(ROOT / 'runs/mf-modelhash230/MODEL_HASHES.jsonl')
    if [row['step'] for row in actual] != list(range(201, 231)):
        raise ValueError('Incomplete actual model hashes')
    if [row['step'] for row in reference] != list(range(201, 231)):
        raise ValueError('Incomplete reference model hashes')
    rows = []
    for left, right in zip(actual, reference):
        a = {row['name']: row for row in left['records']}
        b = {row['name']: row for row in right['records']}
        if len(a) != 48 or set(a) != set(b) or any(a[name]['shape'] != b[name]['shape'] for name in a):
            raise ValueError('Unexpected model layout')
        different = {key: [name for name in a if a[name][key] != b[name][key]]
                     for key in ['fp32_sha256', 'bf16_as_fp32_sha256']}
        rows.append({'step': left['step'], 'different': different})
    return rows


def main() -> None:
    """Keep total-loss acceptance separate from diagnostic component statistics."""
    steps = int(sys.argv[1])
    if steps not in [30, 200]:
        raise ValueError('Expected30 or200 completed steps')
    end = str(200 + steps)
    run, control = 'hp-jtf637-epsilon' + end, 'hp-jtf637-ce-tail' + end
    directory, baseline = ROOT / 'runs' / run, ROOT / 'runs' / control
    checks = fidelity(run, control, steps)
    checks.pop('all_metrics_grad_norm_exact')
    actual = read_rows(directory / 'metrics.jsonl')
    checks['complete_steps'] = [row['step'] for row in actual] == list(range(201, 201 + steps))
    checks['exit_zero'] = (ROOT / 'runs' / (run + '.exit')).read_text().strip() == '0'
    checks['config_only_run_path'] = (ROOT / 'configs' / (control + '.yaml')).read_text().replace(
        control, run) == (ROOT / 'configs' / (run + '.yaml')).read_text()
    unchanged = ['SOURCE_REVISION.json', 'PROBE_SOURCE_MANIFEST.json', 'HEAD_GEMM_ROUTE.json',
                 'SHARED_UP_FP32_FACTOR.json', 'ADAMW_ARCHIVE_ARITHMETIC.json',
                 'Q_NS_ADJACENT_FACTOR.json', 'ROUTER_NS_NPU_SQRT_FACTOR.json', 'WEIGHTED_DVM_FACTOR.json',
                 'HEAD_GEMM_ROUTE_CALLS.jsonl', 'Q_NS_ADJACENT_CALLS.jsonl', 'ROUTER_NS_NPU_SQRT_CALLS.jsonl',
                 'WEIGHTED_DVM_CALLS.jsonl', 'PACKED_SEMANTICS_FACTOR.json', 'PACKED_SEMANTICS_CALLS.jsonl',
                 'NORM219_FACTOR.json', 'NORM219_CALLS.jsonl', 'Q_NPU_SQRT222_FACTOR.json',
                 'CE_SUM_TREE.json', 'CE_TAIL226_FACTOR.json', 'CE_TAIL226_CALLS.jsonl']
    for name in unchanged:
        checks['unchanged_' + name] = (directory / name).read_bytes() == (baseline / name).read_bytes()
    manifest = json.loads((directory / 'PROBE_SOURCE_MANIFEST.json').read_text())
    checks['49_frozen_sources'] = len(manifest['files']) == 49 and all(
        row['sha256'] == source_sha(row['name']) for row in manifest['files'])
    launch = json.loads((ROOT / 'evidence' / (run + '-launch.json')).read_text())
    checks['config_hash'] = launch['config_sha256'] == digest(ROOT / 'configs' / (run + '.yaml'))
    checks['launcher_hash'] = launch['launcher_sha256'] == digest(ROOT / ('run-' + run + '.sh'))
    entry = 'capture_hp_epsilon230.py' if steps == 30 else 'capture_hp_epsilon229.py'
    checks['entry_hash'] = launch['entry_sha256'] == source_sha(entry)
    factor = json.loads((directory / 'NS_EPSILON229_FACTOR.json').read_text())
    checks['factor_source'] = factor['source_sha256'] == source_sha('hp_ns_epsilon229_factor.py')
    checks['factor_base'] = factor['base_source_sha256'] == source_sha('muon_jit_probe.py')
    checks['no_reference_or_step_behavior'] = not (factor['teacher_values_used'] or factor['step_specific_behavior'])
    old = 'current.norm(dim=(-2, -1), keepdim=True) + 1e-7'
    new = 'current.norm(dim=(-2, -1), keepdim=True) + (P12_EPSILON229 if current.ndim == 2 else 1e-7)'
    checks['one_scalar_conversion_only'] = factor['candidate_source'] == factor['original_source'].replace(old, new)
    checks['exact_epsilon'] = factor['epsilon'] == 1.0011717677116394e-7
    checks['all_live_calls'] = read_rows(directory / 'NS_EPSILON229_CALLS.jsonl') == [
        {'step': step, 'cumulative_calls': {shape: count * (step - 200) for shape, count in EXPECTED_CALLS.items()}}
        for step in range(201, 201 + steps)]
    if steps == 30:
        observer = json.loads((directory / 'MODEL_HASH_OBSERVER.json').read_text())
        checks['observer_source'] = observer['source_sha256'] == source_sha('hp_epsilon230_model_hash.py')
        checks['helper_source'] = observer['helper_sha256'] == source_sha('model_hash_stream.py')
    else:
        short = read_rows(ROOT / 'runs/hp-jtf637-epsilon230/metrics.jsonl')
        checks['first30_exact_short'] = len(short) == 30 and actual[:30] == short
    if not all(checks.values()):
        raise ValueError(checks)
    reference = read_metrics(ROOT / 'runs/mf-resume/metrics.jsonl', 'mf', 1e-4, .3)
    errors, summary = {}, {}
    for label, name in [('control', control), ('epsilon', run)]:
        metrics = read_metrics(ROOT / 'runs' / name / 'metrics.jsonl', 'hp', 1e-4, .3)
        errors[label] = [{'step': step, 'errors': {key: abs(metrics[step][key] - reference[step][key]) for key in KEYS}}
                         for step in range(201, 201 + steps)]
        summary[label] = {key: {
            'first': errors[label][0]['errors'][key],
            'max': max(row['errors'][key] for row in errors[label]),
            'max_step': max(errors[label], key=lambda row: row['errors'][key])['step'],
            'mean': sum(row['errors'][key] for row in errors[label]) / steps,
            'bad_steps': [row['step'] for row in errors[label] if row['errors'][key] > .005]}
            for key in KEYS}
    models = model_comparison(run) if steps == 30 else []
    total = summary['epsilon']['total_loss']
    report = {'complete': True, 'steps': steps, 'fidelity': checks, 'rows': errors, 'summary': summary,
              'models': models, 'first_model_difference': next((row for row in models if any(row['different'].values())), None),
              'total_loss_passed': total['first'] <= .001 and not total['bad_steps'],
              'acceptance_complete': steps == 200 and total['first'] <= .001 and not total['bad_steps'],
              'source_sha256': digest(Path(__file__)),
              'scope': 'One independent archived epsilon compatibility factor; total loss only is the current acceptance gate.'}
    (ROOT / ('evidence/frontier229-epsilon-independent' + str(steps) + '-audit.json')).write_text(json.dumps(report, indent=2))
    print(json.dumps({'fidelity': True, 'summary': summary, 'first_model_difference': report['first_model_difference'],
                      'acceptance_complete': report['acceptance_complete']}))


if __name__ == '__main__':
    main()
