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

"""Audit a full continuation after the JT_dev wrapper-identification update."""
import hashlib
import json
from pathlib import Path

from compare_head206_interventions import KEYS, source_sha
from compare_loss_components import read_metrics
from compare_q_up258_full200 import normalized_launcher
from compare_vocab_optimizer_trace import fidelity

ROOT = Path('/alignment/phase12')
RUN = 'hp-jte213-qup-nope400'
CONTROL = 'hp-jtf637-qup-nope400'


def digest(path: Path) -> str:
    """Hash bounded source and metadata bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Require independent input/state restoration and every metric to match."""
    directory, baseline = ROOT / 'runs' / RUN, ROOT / 'runs' / CONTROL
    checks = fidelity(RUN, CONTROL, 200)
    checks['exit_zero'] = (ROOT / 'runs' / (RUN + '.exit')).read_text().strip() == '0'
    checks['config_only_paths'] = (ROOT / 'configs' / (CONTROL + '.yaml')).read_text().replace(
        CONTROL, RUN) == (ROOT / 'configs' / (RUN + '.yaml')).read_text()
    expected = normalized_launcher((ROOT / ('run-' + CONTROL + '.sh')).read_text(), CONTROL)
    expected = expected.replace('jt-f637-pr192-venv', 'jt-e213-pr192-venv')
    expected = expected.replace('PYTHONPATH=/alignment/phase12/tools-f637:',
        'PYTHONPATH=/alignment/phase12/tools-e213:/alignment/phase12/tools-f637:')
    expected = expected.replace('/tools-f637/capture_hp_q_up258.py', '/tools-e213/capture_hp_q_up258.py')
    checks['only_source_runtime_launcher_changed'] = expected == normalized_launcher(
        (ROOT / ('run-' + RUN + '.sh')).read_text(), RUN)
    names = ['PROBE_SOURCE_MANIFEST.json', 'HEAD_GEMM_ROUTE.json', 'SHARED_UP_FP32_FACTOR.json',
             'ADAMW_ARCHIVE_ARITHMETIC.json', 'Q_NS_ADJACENT_FACTOR.json',
             'ROUTER_NS_NPU_SQRT_FACTOR.json', 'WEIGHTED_DVM_FACTOR.json',
             'PACKED_SEMANTICS_FACTOR.json', 'NORM219_FACTOR.json', 'Q_NPU_SQRT222_FACTOR.json',
             'CE_TAIL226_FACTOR.json', 'NS_EPSILON229_FACTOR.json', 'GLOBAL_NPU_SQRT235_FACTOR.json',
             'QUP258_FACTOR.json', 'CLIP_REDUCTION_POLICY.json', 'ARITHMETIC_MODE.json']
    names += [path.name for path in baseline.glob('*CALLS.jsonl')]
    names += ['CLIP_REDUCTION.jsonl', 'metrics.jsonl']
    for name in names:
        checks['unchanged_' + name] = (directory / name).read_bytes() == (baseline / name).read_bytes()
    probe = json.loads((directory / 'PROBE_SOURCE_MANIFEST.json').read_text())
    checks['49_frozen_probes'] = len(probe['files']) == 49 and all(
        row['sha256'] == source_sha(row['name']) for row in probe['files'])
    source = json.loads((directory / 'SOURCE_REVISION.json').read_text())
    checks['new_source_manifest'] = source == json.loads(
        (ROOT / 'evidence/jt-e213-pr192-source-manifest.json').read_text())
    checks['new_upstream_head'] = source['upstream_commit'] == 'e21302f9e7b47d41b61f5f4987fc34c000f7c978'
    checks['all1947_runtime_sources'] = len(source['files']) == 1947 and all(
        row['sha256'] == digest(ROOT / 'hp-jt-e213-pr192' / row['path']) for row in source['files'])
    differences = [row['path'] for row in source['files']
                   if row['sha256'] != digest(ROOT / 'hp-jt-f637-pr192' / row['path'])]
    checks['only_two_source_changes'] = sorted(differences) == [
        'hyper_parallel/tools/model_integration/integration_validator.py',
        'tests/ut/model_integration/test_model_integration.py']
    override = json.loads((directory / 'PROBE_RUNTIME_OVERRIDE.json').read_text())
    checks['guard_hash'] = override['current_guard_sha256'] == digest(
        ROOT / 'tools-e213/capture_hp_pr180_192.py')
    checks['original_guard_hash'] = override['frozen_probe_guard_sha256'] == source_sha('capture_hp_pr180_192.py')
    checks['entry_unchanged'] = override['entry_sha256'] == source_sha('capture_hp_q_up258.py')
    checks['no_teacher_values'] = not override['teacher_values_used']
    first = json.loads((directory / 'CANN_RUNTIME.json').read_text())
    second = json.loads((baseline / 'CANN_RUNTIME.json').read_text())
    for key in ['torch', 'torch_npu', 'torch_path', 'torch_npu_path', 'cann_version_info', 'packages']:
        checks['runtime_' + key] = first[key] == second[key]
    checks['editable_new_import'] = first['hyper_parallel_path'].startswith('/alignment/phase12/hp-jt-e213-pr192/')
    launch = json.loads((ROOT / 'evidence' / (RUN + '-launch.json')).read_text())
    checks['config_hash'] = launch['config_sha256'] == digest(ROOT / 'configs' / (RUN + '.yaml'))
    checks['launcher_hash'] = launch['launcher_sha256'] == digest(ROOT / ('run-' + RUN + '.sh'))
    checks['source_manifest_hash'] = launch['new_source_manifest_sha256'] == digest(
        ROOT / 'evidence/jt-e213-pr192-source-manifest.json')
    if not all(checks.values()):
        raise ValueError(checks)
    hp = read_metrics(directory / 'metrics.jsonl', 'hp', 1e-4, .3)
    mf = read_metrics(ROOT / 'runs/mf-resume/metrics.jsonl', 'mf', 1e-4, .3)
    rows = [{'step': step, 'errors': {key: abs(hp[step][key] - mf[step][key]) for key in KEYS}}
            for step in range(201, 401)]
    summary = {key: {'first': rows[0]['errors'][key], 'last': rows[-1]['errors'][key],
                    'max': max(row['errors'][key] for row in rows),
                    'max_step': max(rows, key=lambda row: row['errors'][key])['step'],
                    'mean': sum(row['errors'][key] for row in rows) / 200,
                    'bad_steps': [row['step'] for row in rows if row['errors'][key] > .005]} for key in KEYS}
    total = summary['total_loss']
    passed = total['first'] <= .001 and not total['bad_steps']
    report = {'complete': True, 'steps': 200, 'run': RUN, 'control': CONTROL, 'fidelity': checks,
              'source_files_changed': differences, 'upstream_commit': source['upstream_commit'],
              'integration_commit': source['commit'], 'summary': summary, 'rows': rows,
              'total_loss_passed': passed, 'acceptance_complete': passed,
              'source_sha256': digest(Path(__file__)),
              'scope': 'Same independent continuation on the new JT_dev revision; no new arithmetic factor.'}
    (ROOT / 'evidence/jt-e213-independent200-audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'fidelity': True, 'summary': summary, 'acceptance_complete': passed}))


if __name__ == '__main__':
    main()
