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

"""Measure a one-time router derivative intervention without accepting it as independent training."""
import hashlib
import json
from pathlib import Path

from compare_head206_interventions import KEYS, model_hashes, source_sha
from compare_loss_components import read_metrics
from compare_vocab_optimizer_trace import fidelity, read_rows

ROOT = Path('/alignment/phase12')
RUN = 'hp-jtf637-router212-dy-teacher226'
BASE = 'hp-jtf637-norm208-q-router-independent226'


def main() -> None:
    """Verify the pre-intervention trajectory and report all positive/negative outcomes."""
    checks = fidelity(RUN, BASE, 26)
    checks.pop('all_metrics_grad_norm_exact')
    directory, baseline = ROOT / 'runs' / RUN, ROOT / 'runs' / BASE
    checks['exit_zero'] = (ROOT / 'runs' / (RUN + '.exit')).read_text().strip() == '0'
    actual, control = read_rows(directory / 'metrics.jsonl'), read_rows(baseline / 'metrics.jsonl')
    clean = lambda row: {key: value for key, value in row.items() if key != 'wall_time'}
    checks['201_211_metrics_norm_exact'] = [clean(row) for row in actual[:11]] == [clean(row) for row in control[:11]]
    checks['all26_steps'] = [row['step'] for row in actual] == list(range(201, 227))
    marker = json.loads((directory / 'DIAGNOSTIC_ONLY.json').read_text())
    checks['excluded_from_acceptance'] = marker['teacher_values_used'] and not marker['eligible_for_alignment_acceptance']
    checks['intervention_source'] = marker['source_sha256'] == source_sha('hp_router212_dy_intervention.py')
    effect = json.loads((directory / 'ROUTER212_DY_INTERVENTION.json').read_text())
    patch = json.loads((ROOT / 'evidence/router212-dy-patch.json').read_text())
    checks['two_only_at212_complete_hashes'] = effect['step'] == 212 and effect['replaced_elements'] == 2 and (
        effect['before_sha256'] == patch['hp_sha256'] and effect['after_sha256'] == patch['mf_sha256'])
    for name in ['SOURCE_REVISION.json', 'PROBE_SOURCE_MANIFEST.json', 'HEAD_GEMM_ROUTE.json',
                 'SHARED_UP_FP32_FACTOR.json', 'ADAMW_ARCHIVE_ARITHMETIC.json',
                 'Q_NS_ADJACENT_FACTOR.json', 'ROUTER_NS_NPU_SQRT_FACTOR.json',
                 'HEAD_GEMM_ROUTE_CALLS.jsonl', 'Q_NS_ADJACENT_CALLS.jsonl', 'ROUTER_NS_NPU_SQRT_CALLS.jsonl']:
        checks['unchanged_' + name] = (directory / name).read_bytes() == (baseline / name).read_bytes()
    manifest = json.loads((directory / 'PROBE_SOURCE_MANIFEST.json').read_text())
    checks['49_frozen_sources'] = len(manifest['files']) == 49 and all(
        row['sha256'] == source_sha(row['name']) for row in manifest['files'])
    models, mf_models = model_hashes(RUN), model_hashes('mf-modelhash226')
    checks['model_hash_steps'] = list(models) == [211, 212, 213, 226]
    observer = json.loads((directory / 'MODEL_HASH_OBSERVER.json').read_text())
    checks['model_observer_source'] = observer['source_sha256'] == source_sha('hp_router212_causal_model_hash.py')
    preclip = {}
    for step in [211, 212]:
        rows = json.loads((directory / ('PRECLIP-' + str(step) + '.json')).read_text())
        reference = {row['name']: row for row in json.loads((ROOT / 'runs/mf-preclip212' / (
            'PRECLIP-' + str(step) + '.json')).read_text())['records']}
        checks['preclip_source_' + str(step)] = rows['source_sha256'] == source_sha('hp_preclip212_hashes.py')
        preclip[step] = {'exact': [row['name'] for row in rows['records']
                                 if row['sha256'] == reference[row['name']]['sha256']],
                        'different': [row['name'] for row in rows['records']
                                      if row['sha256'] != reference[row['name']]['sha256']], 'norm': rows['norm']}
    checks['preclip211_all46_exact'] = len(preclip[211]['exact']) == 46
    if not all(checks.values()):
        raise ValueError(checks)
    metrics = {label: read_metrics(ROOT / 'runs' / run / 'metrics.jsonl', kind, 1e-4, .3)
               for label, run, kind in [('control', BASE, 'hp'), ('teacher', RUN, 'hp'), ('mf', 'mf-resume', 'mf')]}
    rows = [{'step': step, 'errors': {label: {key: abs(metrics[label][step][key] - metrics['mf'][step][key])
                                            for key in KEYS} for label in ['control', 'teacher']}}
            for step in range(201, 227)]
    summary = {label: {key: {'first': rows[0]['errors'][label][key],
                            'max': max(row['errors'][label][key] for row in rows),
                            'mean': sum(row['errors'][label][key] for row in rows) / 26,
                            'bad_steps': [row['step'] for row in rows if row['errors'][label][key] > .005]}
                      for key in KEYS} for label in ['control', 'teacher']}
    model_diff = {step: {key: [name for name, row in model.items() if row[key] != mf_models[step][name][key]]
                        for key in ['fp32_sha256', 'bf16_as_fp32_sha256']} for step, model in models.items()}
    report = {'complete': True, 'diagnostic_only': True, 'eligible_for_alignment_acceptance': False,
              'fidelity': checks, 'preclip': preclip, 'models': model_diff, 'summary': summary, 'rows': rows,
              'first_bad': {label: next((row['step'] for row in rows if max(row['errors'][label].values()) > .005), None)
                            for label in ['control', 'teacher']},
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'scope': 'One-time two router derivative values at212. No model or optimizer state replacement; no new full200.'}
    (ROOT / 'evidence/router212-dy-causal-audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'complete': True, 'preclip_exact': {step: len(row['exact']) for step, row in preclip.items()},
                      'model_fp32_differences': {step: len(row['fp32_sha256']) for step, row in model_diff.items()},
                      'first_bad': report['first_bad'], 'summary': summary}))


if __name__ == '__main__':
    main()
