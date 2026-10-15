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

"""Measure the effect of one bounded router206 model-state intervention."""
import hashlib
import json
from pathlib import Path

from compare_vocab_optimizer_trace import fidelity, read_rows
from compare_loss_components import read_metrics

ROOT = Path('/alignment/phase12')
RUN = 'hp-jtf637-sharedup-router206diag226'
CONTROL = 'hp-jtf637-sharedup-modelhash226'
KEYS = ['total_loss', 'lm_loss', 'mtp_weighted', 'mtp_raw', 'aux_weighted', 'aux_raw']


def main():
    checks = fidelity(RUN, CONTROL, 26)
    metrics_equal = checks.pop('all_metrics_grad_norm_exact')
    checks['exit_zero'] = (ROOT / 'runs' / (RUN + '.exit')).read_text().strip() == '0'
    control = read_rows(ROOT / 'runs' / CONTROL / 'metrics.jsonl')
    actual = read_rows(ROOT / 'runs' / RUN / 'metrics.jsonl')
    checks['before_intervention_all6_exact'] = control[:6] == actual[:6]
    marker = json.loads((ROOT / 'runs' / RUN / 'DIAGNOSTIC_ONLY.json').read_text())
    change = json.loads((ROOT / 'runs' / RUN / 'ROUTER206_INTERVENTION.json').read_text())
    checks['explicit_diagnostic_only'] = marker['diagnostic_only'] and not marker['eligible_for_acceptance']
    checks['source_exact'] = marker['source_sha256'] == hashlib.sha256(
        (ROOT / 'tools-f637/hp_router206_intervention.py').read_bytes()).hexdigest()
    checks['exactly12_coordinates'] = change['changed_elements'] == 12 and change['all_other_elements_unchanged']
    checks['optimizer_unchanged'] = not change['optimizer_state_modified']
    checks['mf_weight_restored'] = change['after_sha256'] == change['native_mf_sha256']
    for family, run in [('before', 'hp-jtf637-sharedup-router206-state'), ('after', 'mf-router206-state')]:
        state = json.loads((ROOT / 'runs' / run / 'ROUTER_STATE-206.json').read_text())
        weight = next(row for row in state['records'] if row['kind'] == 'weight')
        checks[family + '_weight_matches_probe'] = weight['sha256'] == change[family + '_sha256']
    manifest = json.loads((ROOT / 'runs' / RUN / 'PROBE_SOURCE_MANIFEST.json').read_text())
    checks['49_base_sources_exact'] = len(manifest['files']) == 49 and all(
        row['sha256'] == hashlib.sha256((ROOT / 'tools-f637' / row['name']).read_bytes()).hexdigest()
        for row in manifest['files'])
    if not all(checks.values()):
        raise ValueError(checks)
    series = {run: read_metrics(ROOT / 'runs' / run / 'metrics.jsonl', family, 1e-4, .3)
              for run, family in [(RUN, 'hp'), (CONTROL, 'hp'), ('mf-resume', 'mf')]}
    rows = []
    for step in range(201, 227):
        rows.append({'step': step,
                     'control_error': {key: abs(series[CONTROL][step][key] - series['mf-resume'][step][key]) for key in KEYS},
                     'diagnostic_error': {key: abs(series[RUN][step][key] - series['mf-resume'][step][key]) for key in KEYS},
                     'change_from_control': {key: series[RUN][step][key] - series[CONTROL][step][key] for key in KEYS}})
    summary = {}
    for family in ['control_error', 'diagnostic_error']:
        summary[family] = {key: {'max': max(row[family][key] for row in rows),
                                  'mean': sum(row[family][key] for row in rows) / 26,
                                  'bad_steps': [row['step'] for row in rows if row[family][key] > .005]}
                           for key in KEYS}
    report = {'complete': True, 'fidelity': checks, 'eligible_for_acceptance': False,
              'metrics_and_norm_exact_control': metrics_equal, 'rows': rows, 'summary': summary,
              'first_bad_control': next((row['step'] for row in rows if max(row['control_error'].values()) > .005), None),
              'first_bad_diagnostic': next((row['step'] for row in rows if max(row['diagnostic_error'].values()) > .005), None),
              'scope': 'Only one post206 router model-state intervention; no claim of independent implementation or complete200 acceptance.'}
    (ROOT / 'evidence/router206-causal-audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({key: report[key] for key in ['complete', 'metrics_and_norm_exact_control', 'first_bad_control', 'first_bad_diagnostic', 'summary']}))


if __name__ == '__main__':
    main()
