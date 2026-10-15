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

"""Audit all 26 model-hash steps and isolate the measured norm206 intervention."""
import hashlib
import json
from pathlib import Path

from compare_vocab_optimizer_trace import fidelity, read_rows
from compare_loss_components import read_metrics

ROOT = Path('/alignment/phase12')
MF = 'mf-modelhash226'
HP = 'hp-jtf637-sharedup-modelhash226'
DIAG = 'hp-jtf637-sharedup-norm206diag226'
STEPS = list(range(201, 227))
KEYS = ['total_loss', 'lm_loss', 'mtp_weighted', 'mtp_raw', 'aux_weighted', 'aux_raw']


def hashes(run):
    """Require one full native logical model record per completed step."""
    rows = read_rows(ROOT / 'runs' / run / 'MODEL_HASHES.jsonl')
    if [row['step'] for row in rows] != STEPS:
        raise ValueError('Incomplete model observation: ' + run)
    result = {}
    for row in rows:
        records = {record['name']: record for record in row['records']}
        if len(records) != 48 or len(row['records']) != 48:
            raise ValueError('Expected 48 unique model tensors')
        result[row['step']] = records
    return result


def compare_hashes(left, right):
    """Report hashes only; equality failure does not imply a large numerical error."""
    result = []
    for step in STEPS:
        if set(left[step]) != set(right[step]):
            raise ValueError('Model names differ')
        differences = {}
        for key in ['fp32_sha256', 'bf16_as_fp32_sha256']:
            differences[key] = [name for name in left[step] if left[step][name][key] != right[step][name][key]]
        result.append({'step': step, 'different': differences})
    return result


def verify_sources(run, family):
    """Verify the source hashes recorded by the running observer."""
    directory = ROOT / 'runs' / run
    observed = json.loads((directory / 'MODEL_HASH_OBSERVER.json').read_text())
    base = ROOT / ('tools' if family == 'mf' else 'tools-f637')
    checks = {
        'observer_source': observed['source_sha256'] == hashlib.sha256(
            (base / (family + '_model_hash_stream.py')).read_bytes()).hexdigest(),
        'helper_source': observed['helper_sha256'] == hashlib.sha256(
            (base / 'model_hash_stream.py').read_bytes()).hexdigest(),
    }
    if family == 'mf':
        converter = ROOT / 'hyper-parallel/scripts/checkpoint_conversion/convert_mf_ckpt_to_hp.py'
        checks['converter_source'] = observed['converter_sha256'] == hashlib.sha256(converter.read_bytes()).hexdigest()
    else:
        manifest = json.loads((directory / 'PROBE_SOURCE_MANIFEST.json').read_text())
        checks['49_base_sources'] = len(manifest['files']) == 49 and all(
            row['sha256'] == hashlib.sha256((base / row['name']).read_bytes()).hexdigest()
            for row in manifest['files'])
        factor = json.loads((directory / 'SHARED_UP_FP32_FACTOR.json').read_text())
        checks['independent_factor_source'] = factor['source_sha256'] == hashlib.sha256(
            (base / 'hp_shared_up_fp32_factor.py').read_bytes()).hexdigest()
    return checks


def main():
    """Keep the diagnostic ineligible even when its measurements reproduce the control."""
    checks = {}
    for run, family, base in [(MF, 'mf', 'mf-resume'), (HP, 'hp', 'hp-jtf637-adamw-all-sharedupfp32400')]:
        checks[run] = fidelity(run, base, 26)
        checks[run].update(verify_sources(run, family))
        checks[run]['exit_zero'] = (ROOT / 'runs' / (run + '.exit')).read_text().strip() == '0'
    diag_fidelity = fidelity(DIAG, HP, 26)
    checks[DIAG] = {key: value for key, value in diag_fidelity.items() if key != 'all_metrics_grad_norm_exact'}
    checks[DIAG].update(verify_sources(DIAG, 'hp'))
    checks[DIAG]['exit_zero'] = (ROOT / 'runs' / (DIAG + '.exit')).read_text().strip() == '0'
    marker = json.loads((ROOT / 'runs' / DIAG / 'DIAGNOSTIC_ONLY.json').read_text())
    intervention = json.loads((ROOT / 'runs' / DIAG / 'NORM206_INTERVENTION.json').read_text())
    checks[DIAG]['marked_ineligible'] = marker['diagnostic_only'] and not marker['eligible_for_acceptance']
    checks[DIAG]['intervention_source'] = marker['source_sha256'] == hashlib.sha256(
        (ROOT / 'tools-f637/hp_norm206_intervention.py').read_bytes()).hexdigest()
    checks[DIAG]['exact_one_change'] = intervention['changed_elements'] == 1 and intervention['all_other_elements_unchanged']
    checks[DIAG]['native_gradient_restored'] = intervention['after_sha256'] == intervention['native_mf_sha256']
    hp_rows = read_rows(ROOT / 'runs' / HP / 'metrics.jsonl')
    diag_rows = read_rows(ROOT / 'runs' / DIAG / 'metrics.jsonl')
    checks[DIAG]['pre_intervention_exact'] = hp_rows[:5] == diag_rows[:5]
    if not all(all(row.values()) for row in checks.values()):
        raise ValueError(checks)
    series = {run: read_metrics(ROOT / 'runs' / run / 'metrics.jsonl', family, 1e-4, .3)
              for run, family in [(MF, 'mf'), (HP, 'hp'), (DIAG, 'hp')]}
    rows = []
    for step in STEPS:
        rows.append({'step': step,
                     'independent_error': {key: abs(series[HP][step][key] - series[MF][step][key]) for key in KEYS},
                     'diagnostic_error': {key: abs(series[DIAG][step][key] - series[MF][step][key]) for key in KEYS},
                     'diagnostic_control_delta': {key: series[DIAG][step][key] - series[HP][step][key] for key in KEYS}})
    model = {run: hashes(run) for run in [MF, HP, DIAG]}
    control_hashes = compare_hashes(model[HP], model[DIAG])
    report = {'complete': True, 'fidelity': checks, 'diagnostic_eligible_for_acceptance': False,
              'diagnostic_all_metrics_and_norm_exact_control': diag_fidelity['all_metrics_grad_norm_exact'],
              'diagnostic_all48_model_fp32_and_bf16_exact_control': all(
                  not names for row in control_hashes for names in row['different'].values()),
              'native_vs_control_hashes': compare_hashes(model[MF], model[HP]),
              'control_vs_diagnostic_hashes': control_hashes, 'losses': rows,
              'first_bad_independent': next((row['step'] for row in rows if max(row['independent_error'].values()) > .005), None),
              'first_bad_diagnostic': next((row['step'] for row in rows if max(row['diagnostic_error'].values()) > .005), None),
              'scope': 'All48 postupdate model tensors and six losses at201-226; no optimizer-moment equality or later-step conclusion.'}
    (ROOT / 'evidence/modelhash226-norm206-causal-audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({key: report[key] for key in ['complete', 'diagnostic_all_metrics_and_norm_exact_control',
          'diagnostic_all48_model_fp32_and_bf16_exact_control', 'first_bad_independent', 'first_bad_diagnostic']}))


if __name__ == '__main__':
    main()
