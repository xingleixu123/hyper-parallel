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

"""Audit the independent router normalization candidate without accepting a short window as200 steps."""
import hashlib
import json
from pathlib import Path

from compare_vocab_optimizer_trace import fidelity, read_rows
from compare_loss_components import read_metrics

ROOT = Path('/alignment/phase12')
RUN = 'hp-jtf637-sharedup-routernorm226'
CONTROL = 'hp-jtf637-sharedup-modelhash226'
KEYS = ['total_loss', 'lm_loss', 'mtp_weighted', 'mtp_raw', 'aux_weighted', 'aux_raw']


def main():
    checks = fidelity(RUN, CONTROL, 26)
    metrics_equal = checks.pop('all_metrics_grad_norm_exact')
    checks['exit_zero'] = (ROOT / 'runs' / (RUN + '.exit')).read_text().strip() == '0'
    checks['no_diagnostic_marker'] = not (ROOT / 'runs' / RUN / 'DIAGNOSTIC_ONLY.json').exists()
    marker = json.loads((ROOT / 'runs' / RUN / 'ROUTER_NS_SQUARE_SUM_FACTOR.json').read_text())
    checks['no_teacher'] = not marker['teacher_values_used']
    checks['factor_source_exact'] = marker['source_sha256'] == hashlib.sha256(
        (ROOT / 'tools-f637/router_ns_square_sum_factor.py').read_bytes()).hexdigest()
    calls = read_rows(ROOT / 'runs' / RUN / 'ROUTER_NS_SQUARE_SUM_CALLS.jsonl')
    checks['two_router_calls_per_step'] = calls == [
        {'step': step, 'total_router_calls': 2 * (step - 200)} for step in range(201, 227)]
    manifest = json.loads((ROOT / 'runs' / RUN / 'PROBE_SOURCE_MANIFEST.json').read_text())
    checks['49_base_sources_exact'] = len(manifest['files']) == 49 and all(
        row['sha256'] == hashlib.sha256((ROOT / 'tools-f637' / row['name']).read_bytes()).hexdigest()
        for row in manifest['files'])
    observer = json.loads((ROOT / 'runs' / RUN / 'MODEL_HASH_OBSERVER.json').read_text())
    checks['observer_source_exact'] = observer['source_sha256'] == hashlib.sha256(
        (ROOT / 'tools-f637/hp_router_norm_model_hash.py').read_bytes()).hexdigest()
    checks['hash_helper_exact'] = observer['helper_sha256'] == hashlib.sha256(
        (ROOT / 'tools-f637/model_hash_stream.py').read_bytes()).hexdigest()
    series = {run: read_metrics(ROOT / 'runs' / run / 'metrics.jsonl', family, 1e-4, .3)
              for run, family in [(RUN, 'hp'), (CONTROL, 'hp'), ('mf-resume', 'mf')]}
    rows = []
    for step in range(201, 227):
        rows.append({'step': step,
                     'control_error': {key: abs(series[CONTROL][step][key] - series['mf-resume'][step][key]) for key in KEYS},
                     'candidate_error': {key: abs(series[RUN][step][key] - series['mf-resume'][step][key]) for key in KEYS}})
    hashes = {run: {row['step']: {item['name']: item for item in row['records']}
                   for row in read_rows(ROOT / 'runs' / run / 'MODEL_HASHES.jsonl')}
              for run in [RUN, CONTROL, 'mf-modelhash226']}
    checks['exact7_hash_steps'] = list(hashes[RUN]) == list(range(201, 208))
    model_rows = []
    for step in range(201, 208):
        if any(len(hashes[run][step]) != 48 for run in hashes):
            raise ValueError('Expected48 complete model tensors')
        model_rows.append({'step': step, 'different_vs_control': [name for name in hashes[RUN][step]
            if hashes[RUN][step][name]['fp32_sha256'] != hashes[CONTROL][step][name]['fp32_sha256']],
            'different_vs_mf': [name for name in hashes[RUN][step]
            if hashes[RUN][step][name]['fp32_sha256'] != hashes['mf-modelhash226'][step][name]['fp32_sha256']]})
    if not all(checks.values()):
        raise ValueError(checks)
    summary = {family: {key: {'max': max(row[family][key] for row in rows),
                              'mean': sum(row[family][key] for row in rows) / 26,
                              'bad_steps': [row['step'] for row in rows if row[family][key] > .005]} for key in KEYS}
               for family in ['control_error', 'candidate_error']}
    report = {'complete': True, 'fidelity': checks, 'independent': True, 'window_steps': 26,
              'metrics_and_norm_exact_control': metrics_equal, 'rows': rows, 'summary': summary,
              'models': model_rows,
              'first_bad_control': next((row['step'] for row in rows if max(row['control_error'].values()) > .005), None),
              'first_bad_candidate': next((row['step'] for row in rows if max(row['candidate_error'].values()) > .005), None),
              'scope': 'One independent formula across both8x5120 router NS inputs; short window only, not full200 acceptance.'}
    (ROOT / 'evidence/router-norm-independent226-audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({key: report[key] for key in ['complete', 'first_bad_control', 'first_bad_candidate', 'summary']}))


if __name__ == '__main__':
    main()
