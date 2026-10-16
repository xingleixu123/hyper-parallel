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

"""Audit independent 200-step MTP weighted-gradient reduction against its matched control."""
import hashlib
import json
from pathlib import Path
import re

from compare_head206_interventions import KEYS, source_sha
from compare_loss_components import read_metrics
from compare_vocab_optimizer_trace import fidelity, read_rows

ROOT = Path('/alignment/phase12')
RUN = 'hp-jtf637-weighted-mtp400'
CONTROL = 'hp-jtf637-global-sqrt400'


def digest(path: Path) -> str:
    """Hash a bounded source or provenance file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized_launcher(text: str, run: str) -> str:
    """Remove only run destinations and resource allocation from a launcher comparison."""
    text = text.replace(run, 'RUN')
    text = re.sub(r'ASCEND_RT_VISIBLE_DEVICES=\d+', 'ASCEND_RT_VISIBLE_DEVICES=CARD', text)
    return re.sub(r'--master_port=\d+', '--master_port=PORT', text)


def main() -> None:
    """Separate configuration fidelity, diagnostic components, and total-loss acceptance."""
    directory, baseline = ROOT / 'runs' / RUN, ROOT / 'runs' / CONTROL
    checks = fidelity(RUN, CONTROL, 200)
    checks.pop('all_metrics_grad_norm_exact')
    actual = read_rows(directory / 'metrics.jsonl')
    checks['complete_steps'] = [row['step'] for row in actual] == list(range(201, 401))
    checks['exit_zero'] = (ROOT / 'runs' / (RUN + '.exit')).read_text().strip() == '0'
    checks['config_only_run_path'] = (ROOT / 'configs' / (CONTROL + '.yaml')).read_text().replace(
        CONTROL, RUN) == (ROOT / 'configs' / (RUN + '.yaml')).read_text()
    unchanged = ['RESTORE.json', 'SOURCE_REVISION.json', 'PROBE_SOURCE_MANIFEST.json', 'HEAD_GEMM_ROUTE.json',
                 'SHARED_UP_FP32_FACTOR.json', 'ADAMW_ARCHIVE_ARITHMETIC.json',
                 'Q_NS_ADJACENT_FACTOR.json', 'ROUTER_NS_NPU_SQRT_FACTOR.json',
                 'HEAD_GEMM_ROUTE_CALLS.jsonl', 'Q_NS_ADJACENT_CALLS.jsonl', 'ROUTER_NS_NPU_SQRT_CALLS.jsonl',
                 'PACKED_SEMANTICS_FACTOR.json', 'PACKED_SEMANTICS_CALLS.jsonl',
                 'NORM219_FACTOR.json', 'NORM219_CALLS.jsonl', 'Q_NPU_SQRT222_FACTOR.json',
                 'CE_SUM_TREE.json', 'CE_TAIL226_FACTOR.json', 'CE_TAIL226_CALLS.jsonl',
                 'NS_EPSILON229_FACTOR.json', 'NS_EPSILON229_CALLS.jsonl', 'CLIP_REDUCTION_POLICY.json',
                 'GLOBAL_NPU_SQRT235_FACTOR.json']
    for name in unchanged:
        checks['unchanged_' + name] = (directory / name).read_bytes() == (baseline / name).read_bytes()
    manifest = json.loads((directory / 'PROBE_SOURCE_MANIFEST.json').read_text())
    checks['49_frozen_sources'] = len(manifest['files']) == 49 and all(
        row['sha256'] == source_sha(row['name']) for row in manifest['files'])
    launch = json.loads((ROOT / 'evidence' / (RUN + '-launch.json')).read_text())
    checks['config_hash'] = launch['config_sha256'] == digest(ROOT / 'configs' / (RUN + '.yaml'))
    checks['launcher_hash'] = launch['launcher_sha256'] == digest(ROOT / ('run-' + RUN + '.sh'))
    checks['entry_hash'] = launch['entry_sha256'] == source_sha('capture_hp_global_sqrt400.py')
    checks['factor_hash'] = launch['factor_sha256'] == source_sha('hp_weighted_dvm_factor.py')
    expected = normalized_launcher((ROOT / ('run-' + CONTROL + '.sh')).read_text(), CONTROL)
    expected = expected.replace('P12_WEIGHTED_DVM_SCOPE=main', 'P12_WEIGHTED_DVM_SCOPE=both')
    checks['only_scope_launch_change'] = expected == normalized_launcher(
        (ROOT / ('run-' + RUN + '.sh')).read_text(), RUN)
    policy = json.loads((directory / 'WEIGHTED_DVM_FACTOR.json').read_text())
    prior_policy = json.loads((baseline / 'WEIGHTED_DVM_FACTOR.json').read_text())
    checks['weighted_scope_only'] = policy == {**prior_policy, 'scope': 'both'}
    checks['no_reference_or_step_behavior'] = not (
        policy['teacher_values_used'] or policy['coordinate_or_step_special_case'])
    checks['weighted_both_calls'] = read_rows(directory / 'WEIGHTED_DVM_CALLS.jsonl') == [
        {'step': step, 'calls': {'main': step - 200, 'mtp': step - 200}} for step in range(201, 401)]
    short = read_rows(ROOT / 'runs/hp-jtf637-weighted-mtp236/metrics.jsonl')
    checks['first36_exact_short'] = len(short) == 36 and actual[:36] == short
    clips = read_rows(directory / 'CLIP_REDUCTION.jsonl')
    checks['all_live_clipping_steps'] = [row['step'] for row in clips] == list(range(201, 401))
    checks['first36_exact_short_clips'] = clips[:36] == read_rows(
        ROOT / 'runs/hp-jtf637-weighted-mtp236/CLIP_REDUCTION.jsonl')
    if not all(checks.values()):
        raise ValueError(checks)
    reference = read_metrics(ROOT / 'runs/mf-resume/metrics.jsonl', 'mf', 1e-4, .3)
    errors, summary = {}, {}
    for label, name in [('control', CONTROL), ('weighted_mtp', RUN)]:
        metrics = read_metrics(ROOT / 'runs' / name / 'metrics.jsonl', 'hp', 1e-4, .3)
        errors[label] = [{'step': step, 'errors': {key: abs(metrics[step][key] - reference[step][key])
                                                for key in KEYS}} for step in range(201, 401)]
        summary[label] = {key: {
            'first': errors[label][0]['errors'][key],
            'max': max(row['errors'][key] for row in errors[label]),
            'max_step': max(errors[label], key=lambda row: row['errors'][key])['step'],
            'mean': sum(row['errors'][key] for row in errors[label]) / 200,
            'bad_steps': [row['step'] for row in errors[label] if row['errors'][key] > .005]}
            for key in KEYS}
    mf_rows = read_rows(ROOT / 'runs/mf-resume/metrics.jsonl')
    norm_differences = [{'step': row['step'], 'hp': row['grad_norm'], 'mf': ref['outputs'][4][0]}
                        for row, ref in zip(actual, mf_rows)
                        if row['grad_norm'] != ref['outputs'][4][0]]
    total = summary['weighted_mtp']['total_loss']
    passed = total['first'] <= .001 and not total['bad_steps']
    report = {'complete': True, 'steps': 200, 'fidelity': checks, 'rows': errors, 'summary': summary,
              'first_grad_norm_difference': next(iter(norm_differences), None),
              'total_loss_passed': passed, 'acceptance_complete': passed,
              'source_sha256': digest(Path(__file__)),
              'scope': 'Independent main-to-both weighted-gradient reduction scope; total loss is the current gate. '
                       'MTP and Aux losses are diagnostic. No reference values are injected into training.'}
    (ROOT / 'evidence/frontier236-weighted-mtp-independent200-audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'fidelity': True, 'summary': summary,
                      'first_grad_norm_difference': report['first_grad_norm_difference'],
                      'acceptance_complete': passed}))


if __name__ == '__main__':
    main()
