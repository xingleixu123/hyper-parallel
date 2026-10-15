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

"""Audit independent head GEMM arithmetic routes against the fixed26-step control."""
import hashlib
import json
from pathlib import Path

import yaml

from compare_vocab_optimizer_trace import fidelity, read_rows
from compare_loss_components import read_metrics
from compare_head206_interventions import model_hashes, source_sha, KEYS

ROOT = Path('/alignment/phase12')
RUNS = {'legacy_nz': 'hp-jtf637-headrouter-frontier210'}
BASE = 'hp-jtf637-sharedup-headlegacynz-routerdvm226'


def main() -> None:
    """Reject wrapper drift and verify source, complete restoration, inputs and actual route uses."""
    checks = {}
    hashes = {'control': model_hashes('hp-jtf637-sharedup-modelhash226'), 'mf': model_hashes('mf-modelhash226')}
    base_source = json.loads((ROOT / 'runs' / BASE / 'SOURCE_REVISION.json').read_text())
    for mode, run in RUNS.items():
        directory = ROOT / 'runs' / run
        check = fidelity(run, BASE, 10)
        actual = read_rows(directory / 'metrics.jsonl')
        check['exact10_steps'] = [row['step'] for row in actual] == list(range(201, 211))
        check['exit_zero'] = (ROOT / 'runs' / (run + '.exit')).read_text().strip() == '0'
        check['source_revision'] = json.loads((directory / 'SOURCE_REVISION.json').read_text()) == base_source
        marker = json.loads((directory / 'HEAD_GEMM_ROUTE.json').read_text())
        check['mode'] = marker['mode'] == mode
        check['independent'] = not marker['teacher_values_used'] and not marker['coordinate_or_step_special_case']
        check['all_steps_factor'] = marker['all_resumed_steps']
        check['no_teacher_marker'] = not (directory / 'DIAGNOSTIC_ONLY.json').exists()
        factor = 'hp_head_gemm_route_factor.py' if mode == 'mm' else 'hp_head_gemm_legacy_nz_factor.py'
        check['route_source'] = marker['source_sha256'] == source_sha(factor)
        if mode == 'legacy_nz':
            check['helper_sources'] = all(digest == source_sha(name) for name, digest in marker['helper_sha256'].items())
            check['actual_storage_every_step'] = all(row['storage_metadata'] == [
                {'logical_shape': [4096, 152064], 'storage_shape': [9504, 256, 16, 16], 'format': 29, 'buffer_bytes': 1245708288},
                {'logical_shape': [4096, 5120], 'storage_shape': [320, 256, 16, 16], 'format': 29, 'buffer_bytes': 41943040},
                {'logical_shape': [152064, 5120], 'storage_shape': [320, 9504, 16, 16], 'format': 29, 'buffer_bytes': 1557135360}]
                for row in read_rows(directory / 'HEAD_GEMM_ROUTE_CALLS.jsonl'))
        calls = read_rows(directory / 'HEAD_GEMM_ROUTE_CALLS.jsonl')
        check['20_actual_backward_calls'] = [{key: row[key] for key in ['step', 'mode', 'cumulative_backward_calls']} for row in calls] == [
            {'step': step, 'mode': mode, 'cumulative_backward_calls': 2 * (step - 200)}
            for step in range(201, 211)]
        if mode == 'legacy_nz':
            router = json.loads((directory / 'ROUTER_NS_DVM_FACTOR.json').read_text())
            check['router_no_teacher'] = not router['teacher_values_used']
            check['router_source'] = router['source_sha256'] == source_sha('router_ns_dvm_factor.py')
            check['router_helper_source'] = router['reduction_helper_sha256'] == source_sha('observe_hp_preclip_norm_adjacent.py')
            check['20_router_calls'] = read_rows(directory / 'ROUTER_NS_DVM_CALLS.jsonl') == [
                {'step': step, 'total_router_calls': 2 * (step - 200)} for step in range(201, 211)]
        manifest = json.loads((directory / 'PROBE_SOURCE_MANIFEST.json').read_text())
        check['49_base_sources'] = len(manifest['files']) == 49 and all(
            row['sha256'] == source_sha(row['name']) for row in manifest['files'])
        observer = json.loads((directory / 'MODEL_HASH_OBSERVER.json').read_text())
        check['model_hash_observer'] = observer['source_sha256'] == source_sha('hp_frontier208_model_hash.py')
        check['model_hash_helper'] = observer['helper_sha256'] == source_sha('model_hash_stream.py')
        config = yaml.safe_load((ROOT / 'configs' / (run + '.yaml')).read_text())
        base_config = yaml.safe_load((ROOT / 'configs' / (BASE + '.yaml')).read_text())
        check['end_step210'] = config['training']['train_iters'] == 210
        config['training']['train_iters'] = base_config['training']['train_iters']
        config['checkpoint']['checkpoint_dir'] = base_config['checkpoint']['checkpoint_dir']
        check['recipe_otherwise_unchanged'] = config == base_config
        hashes[mode] = model_hashes(run)
        check['hash_steps'] = list(hashes[mode]) == [208, 209, 210]
        if mode == 'mm':
            check['all48_models206207_exact_control'] = all(
                hashes[mode][step] == hashes['control'][step] for step in [208, 209, 210])
        checks[mode] = check
    if not all(all(check.values()) for check in checks.values()):
        raise ValueError(checks)
    series = {mode: read_metrics(ROOT / 'runs' / run / 'metrics.jsonl', 'hp', 1e-4, .3)
              for mode, run in RUNS.items()}
    series['mf'] = read_metrics(ROOT / 'runs/mf-resume/metrics.jsonl', 'mf', 1e-4, .3)
    rows = [{'step': step, 'absolute_error': {
        mode: {key: abs(series[mode][step][key] - series['mf'][step][key]) for key in KEYS}
        for mode in RUNS}} for step in range(201, 211)]
    summary = {mode: {key: {'first': rows[0]['absolute_error'][mode][key],
                           'max': max(row['absolute_error'][mode][key] for row in rows),
                           'mean': sum(row['absolute_error'][mode][key] for row in rows) / 10,
                           'bad_steps': [row['step'] for row in rows if row['absolute_error'][mode][key] > .005]}
                     for key in KEYS} for mode in RUNS}
    first_bad = {mode: next((row['step'] for row in rows if max(row['absolute_error'][mode].values()) > .005), None)
                 for mode in RUNS}
    models = [{'mode': mode, 'step': step, 'different_vs_mf': {key: [
        name for name, record in hashes[mode][step].items() if record[key] != hashes['mf'][step][name][key]]
        for key in ['fp32_sha256', 'bf16_as_fp32_sha256']}} for mode in RUNS for step in [208, 209, 210]]
    report = {'complete': True, 'independent': True, 'acceptance_complete': False, 'window_steps': 10,
              'fidelity': checks, 'runs': RUNS, 'first_bad': first_bad, 'summary': summary, 'models': models,
              'rows': rows, 'scope': 'Read-only model hashes208-210; ten-step loss and gradient norms must remain exactly identical to the independent combination26 trajectory.'}
    (ROOT / 'evidence/head208-frontier210-audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({key: report[key] for key in ['complete', 'fidelity', 'first_bad', 'summary', 'models']}))


if __name__ == '__main__':
    main()
