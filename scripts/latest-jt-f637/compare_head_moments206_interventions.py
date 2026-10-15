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

"""Separate accumulated head moments from model-state differences in actual short trajectories."""
import hashlib
import json
from pathlib import Path

from compare_vocab_optimizer_trace import fidelity, read_rows
from compare_loss_components import read_metrics
from compare_head206_interventions import model_hashes, source_sha, KEYS

ROOT = Path('/alignment/phase12')
RUNS = {
    'control': 'hp-jtf637-sharedup-modelhash226',
    'head': 'hp-jtf637-sharedup-head206diag226',
    'head_router': 'hp-jtf637-sharedup-headrouter206diag226',
    'moments': 'hp-jtf637-sharedup-moments206diag226',
    'head_moments': 'hp-jtf637-sharedup-headmoments206diag226',
    'head_router_moments': 'hp-jtf637-sharedup-headroutermoments206diag226',
}
MODEL_REFERENCE = {'moments': 'control', 'head_moments': 'head', 'head_router_moments': 'head_router'}
MF = 'mf-head207-momenthash'


def without_time(rows: list) -> list:
    """Compare numerical metrics independently of observation overhead."""
    return [{key: value for key, value in row.items() if key != 'wall_time'} for row in rows]


def main() -> None:
    """Require the207 preupdate trajectory to match its model-only reference exactly."""
    comparison = json.loads((ROOT / 'evidence/head-moments206-comparison.json').read_text())
    previous = json.loads((ROOT / 'evidence/head206-2x2-causal-audit.json').read_text())
    if not previous['complete'] or not all(all(row.values()) for row in previous['fidelity'].values()):
        raise ValueError('Prior model-only diagnostics are not audited')
    hashes = {mode: model_hashes(run) for mode, run in RUNS.items()}
    hashes['mf'] = model_hashes('mf-modelhash226')
    checks = {MF: fidelity(MF, 'mf-resume', 7)}
    native = json.loads((ROOT / 'runs' / MF / 'HEAD_MOMENTS207_HASH.json').read_text())
    check = checks[MF]
    check['exit_zero'] = (ROOT / 'runs' / (MF + '.exit')).read_text().strip() == '0'
    check['native_step'] = native['step'] == native['optimizer_step'] == 207
    check['native_head'] = native['model_fp32_sha256'] == hashes['mf'][207]['lm_head.weight']['fp32_sha256']
    check['source'] = native['source_sha256'] == hashlib.sha256((ROOT / 'tools/mf_head_moments207_hash.py').read_bytes()).hexdigest()
    check['helper'] = native['helper_sha256'] == hashlib.sha256((ROOT / 'tools/head_moments206_observation.py').read_bytes()).hexdigest()
    native_moments = {row['kind']: row['sha256'] for row in native['records']}
    check['two_native_slots'] = set(native_moments) == {'exp_avg', 'exp_avg_sq'}
    moment_rows = []
    for mode, reference in MODEL_REFERENCE.items():
        run = RUNS[mode]
        directory = ROOT / 'runs' / run
        check = fidelity(run, RUNS['control'], 26)
        check.pop('all_metrics_grad_norm_exact')
        check['exit_zero'] = (ROOT / 'runs' / (run + '.exit')).read_text().strip() == '0'
        actual = read_rows(directory / 'metrics.jsonl')
        ref = read_rows(ROOT / 'runs' / RUNS[reference] / 'metrics.jsonl')
        check['first7_metrics_and_norm_exact_model_reference'] = without_time(actual[:7]) == without_time(ref[:7])
        marker = json.loads((directory / 'DIAGNOSTIC_ONLY.json').read_text())
        change = json.loads((directory / 'HEAD_MOMENTS206_INTERVENTION.json').read_text())
        check['diagnostic_only'] = marker['diagnostic_only'] and not marker['eligible_for_acceptance'] and marker['teacher_moment_values_used']
        check['mode'] = marker['mode'] == change['mode'] == mode
        check['one206_step'] = marker['step'] == change['step'] == change['optimizer_step_before'] == change['optimizer_step_after'] == 206
        check['counter_and_other_slots_not_patched'] = not marker['optimizer_counter_modified'] and not change['other_optimizer_slots_modified']
        check['moment_source'] = marker['source_sha256'] == source_sha('hp_head_moments206_intervention.py')
        check['patch_helper_source'] = marker['patch_helper_sha256'] == source_sha('hp_head206_intervention.py')
        check['exact_two_slots'] = set(change['changed']) == {'exp_avg', 'exp_avg_sq'}
        for kind in ['exp_avg', 'exp_avg_sq']:
            patch = change['changed'][kind]
            check[kind + '_count'] = patch['changed_elements'] == comparison['moments'][kind]['different'] and patch['other_elements_unchanged']
            check[kind + '_patch_sha'] = marker['patches'][kind] == comparison['moments'][kind]['patch_sha256']
            for family, side in [('hp', 'before'), ('mf', 'after')]:
                expected = next(row['sha256'] for row in comparison['metadata'][family]['records'] if row['kind'] == kind)
                check[kind + '_' + side] = patch[side + '_sha256'] == expected
        model_marker = marker['model_intervention']
        if mode == 'moments':
            check['no_model_intervention'] = model_marker is None and not (directory / 'MODEL206_INTERVENTION.json').exists()
        else:
            check['model_intervention_source'] = model_marker['source_sha256'] == source_sha('hp_head206_intervention.py')
            check['router_helper_source'] = model_marker['router_helper_sha256'] == source_sha('hp_router206_intervention.py')
            check['model_mode'] = model_marker['mode'] == ('head' if mode == 'head_moments' else 'head_router')
            model_change = json.loads((directory / 'MODEL206_INTERVENTION.json').read_text())
            expected_names = ['lm_head.weight'] + (['model.layers.1.mlp.gate.weight'] if mode == 'head_router_moments' else [])
            check['model_scope'] = sorted(model_change['changed']) == sorted(expected_names)
        check['hash_steps'] = list(hashes[mode]) == [206, 207]
        check['all48_model206_exact_model_reference'] = hashes[mode][206] == hashes[reference][206]
        check['all47_nonhead207_exact_model_reference'] = all(
            hashes[mode][207][name] == row for name, row in hashes[reference][207].items() if name != 'lm_head.weight')
        check['head206_unchanged_by_moment_patch'] = change['weight_fp32_sha256'] == hashes[reference][206]['lm_head.weight']['fp32_sha256']
        manifest = json.loads((directory / 'PROBE_SOURCE_MANIFEST.json').read_text())
        check['49_base_sources'] = len(manifest['files']) == 49 and all(row['sha256'] == source_sha(row['name']) for row in manifest['files'])
        observer = json.loads((directory / 'MODEL_HASH_OBSERVER.json').read_text())
        check['model_hash_observer'] = observer['source_sha256'] == source_sha('hp_frontier206_model_hash.py')
        check['model_hash_helper'] = observer['helper_sha256'] == source_sha('model_hash_stream.py')
        current = json.loads((directory / 'HEAD_MOMENTS207_HASH.json').read_text())
        check['moment207_source'] = current['source_sha256'] == source_sha('hp_head_moments206_intervention.py')
        check['moment207_step'] = current['step'] == current['optimizer_step'] == 207
        current_moments = {row['kind']: row['sha256'] for row in current['records']}
        check['moment207_slots'] = set(current_moments) == {'exp_avg', 'exp_avg_sq'}
        moment_rows.append({'mode': mode, 'step': 207, 'head_moments_exact_native': {
            kind: current_moments[kind] == native_moments[kind] for kind in native_moments},
            'head_model_fp32_exact_native': hashes[mode][207]['lm_head.weight']['fp32_sha256'] == hashes['mf'][207]['lm_head.weight']['fp32_sha256']})
        checks[mode] = check
    if not all(all(row.values()) for row in checks.values()):
        raise ValueError(checks)
    series = {mode: read_metrics(ROOT / 'runs' / run / 'metrics.jsonl', 'hp', 1e-4, .3) for mode, run in RUNS.items()}
    series['mf'] = read_metrics(ROOT / 'runs/mf-resume/metrics.jsonl', 'mf', 1e-4, .3)
    rows = []
    for step in range(201, 227):
        error = {mode: {key: abs(series[mode][step][key] - series['mf'][step][key]) for key in KEYS} for mode in RUNS}
        interaction = {key: series['head_moments'][step][key] - series['head'][step][key]
                       - series['moments'][step][key] + series['control'][step][key] for key in KEYS}
        rows.append({'step': step, 'absolute_error': error, 'head_moment_signed_interaction': interaction})
    summary = {mode: {key: {'max': max(row['absolute_error'][mode][key] for row in rows),
                           'mean': sum(row['absolute_error'][mode][key] for row in rows) / 26,
                           'bad_steps': [row['step'] for row in rows if row['absolute_error'][mode][key] > .005]}
                     for key in KEYS} for mode in RUNS}
    first_bad = {mode: next((row['step'] for row in rows if max(row['absolute_error'][mode].values()) > .005), None) for mode in RUNS}
    models = [{'mode': mode, 'step': step, 'different_vs_mf': {key: [
        name for name, record in hashes[mode][step].items() if record[key] != hashes['mf'][step][name][key]]
        for key in ['fp32_sha256', 'bf16_as_fp32_sha256']}}
        for mode in MODEL_REFERENCE for step in [206, 207]]
    report = {'complete': True, 'eligible_for_acceptance': False, 'window_steps': 26,
              'fidelity': checks, 'runs': RUNS, 'first_bad': first_bad, 'summary': summary,
              'moment_hashes207': moment_rows, 'models': models, 'rows': rows,
              'scope': 'One206 head-moment teacher intervention, optionally with prior model factors; '
                       'no claim of all64 optimizer slots matching, mathematical superiority or independent200-step acceptance.'}
    (ROOT / 'evidence/head-moments206-causal-audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({key: report[key] for key in ['complete', 'first_bad', 'summary', 'moment_hashes207']}))


if __name__ == '__main__':
    main()
