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

"""Join actual state replay fidelity with native normalization statistics."""
import csv
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path('/alignment/phase12')
STATE = Path('/state/jt-recipe-20261014')


def digest(path: Path) -> str:
    """Hash a bounded source or evidence file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Do not infer internal values from rounded printed statistics."""
    control = json.loads((ROOT / 'evidence/frontier306-hp-control-replay.json').read_text())
    candidate = json.loads((ROOT / 'evidence/frontier306-hp-both-replay.json').read_text())
    native = json.loads((ROOT / 'evidence/frontier306-mf-builtin-stat.json').read_text())
    state = json.loads((ROOT / 'evidence/frontier306-state-comparison.json').read_text())
    checks = {
        'state_observation_fidelity': state['complete'] and all(
            all(row.values()) for row in state['fidelity'].values()),
        'control_full_state_exact': all(row['weight_exact_training'] and row['momentum_exact_training']
                                       for row in control['checks']),
        'candidate_full_state_exact_mf': all(row['weight_exact_mf_training'] and row['momentum_exact_training']
                                           for row in candidate['checks']),
        'native_full_state_exact': all(row['exact_actual_training'] for row in native['native_state_fidelity'].values()),
        'same_actual_inputs_and_trace': control['records'] == candidate['records'],
        'two_partitions': len(control['records']) == len(candidate['records']) == 2,
        'native_source_unchanged': not native['source_tensor_dump_inserted'],
    }
    for mode, report in [('control', control), ('both', candidate)]:
        run = 'frontier306-hp-' + mode + '-replay'
        launch = json.loads((ROOT / 'evidence' / (run + '-launch.json')).read_text())
        checks[mode + '_exit_zero'] = (ROOT / 'runs' / (run + '.exit')).read_text().strip() == '0'
        checks[mode + '_source_hash'] = report['source_sha256'] == launch['source_sha256'] == digest(
            ROOT / 'tools-f637/replay_hp_frontier306_muon.py')
        checks[mode + '_launcher_hash'] = launch['launcher_sha256'] == digest(ROOT / ('run-' + run + '.sh'))
        checks[mode + '_explicit_normalization_exact'] = all(
            row['explicit_normalization_exact'] for row in report['records'])
    run = 'frontier306-mf-builtin-stat'
    launch = json.loads((ROOT / 'evidence' / (run + '-launch.json')).read_text())
    checks['mf_exit_zero'] = (ROOT / 'runs' / (run + '.exit')).read_text().strip() == '0'
    checks['mf_source_hash'] = native['source_sha256'] == launch['source_sha256'] == digest(
        ROOT / 'tools/replay_mf_frontier306_builtin_dump.py')
    checks['mf_launcher_hash'] = launch['launcher_sha256'] == digest(ROOT / ('run-' + run + '.sh'))
    checks['mf_dump_config_hash'] = native['dump_config_sha256'] == launch['config_sha256'] == digest(
        ROOT / 'configs' / (run + '.json'))
    dump = STATE / run
    paths = list(dump.rglob('statistic.csv'))
    if len(paths) != 1:
        raise ValueError('Native statistics invocation is ambiguous')
    rows = list(csv.DictReader(paths[0].open()))
    checks['six_native_records'] = len(rows) == 6
    checks['statistics_only'] = not any(dump.rglob('*.npy')) and not any(dump.rglob('*.npz'))
    comparisons = []
    for piece, hp in enumerate(control['records']):
        shape = hp['pre_norm']['shape']
        inputs = [row for row in rows if row['IO'] == 'input' and json.loads(row['Shape']) == shape]
        if len(inputs) != 1:
            raise ValueError('Native normalization input layout is ambiguous')
        raw = inputs[0]
        selected = [row for row in rows if row['Op Name'] == raw['Op Name']]
        scalar = next(row for row in selected if row['IO'] == 'input' and row['Shape'] == '[]')
        output = next(row for row in selected if row['IO'] == 'output')
        if len(selected) != 3 or scalar['Data Type'] != 'float32' or output['Data Type'] != 'bfloat16':
            raise ValueError('Unexpected native fused boundary')
        record = {
            'piece': piece, 'scope': 'nope' if piece == 0 else 'rope', 'shape': shape,
            'pre_norm_exact_native': hp['pre_norm']['native_dump_sha1'] == raw['SHA1'],
            'square_sum_exact_native': hp['candidate_square_sum_sha1'] == scalar['SHA1'],
            'candidate_normalized_exact_native': hp['candidate_normalized']['native_dump_sha1'] == output['SHA1'],
            'control_normalized_exact_native': hp['control_normalized']['native_dump_sha1'] == output['SHA1'],
            'control_candidate_ns_exact': hp['control_ns']['fp32_sha256'] == hp['candidate_ns']['fp32_sha256'],
            'control_scalar': hp['ordinary_scalar'], 'candidate_scalar': hp['candidate_scalar'],
            'native_input_sha1': raw['SHA1'], 'native_square_sum_sha1': scalar['SHA1'],
            'native_output_sha1': output['SHA1'], 'precision': hp['precision_control_vs_candidate'],
        }
        comparisons.append(record)
    checks['all_native_inputs_and_candidates_exact'] = all(
        row['pre_norm_exact_native'] and row['square_sum_exact_native'] and row['candidate_normalized_exact_native']
        for row in comparisons)
    if not all(checks.values()):
        raise ValueError(checks)
    csv_path = ROOT / 'evidence/frontier306-native-normalization-stat.csv'
    shutil.copyfile(paths[0], csv_path)
    graph = dump / 'rank_0/graphs/ms_output_trace_code_graph_0.ir'
    shutil.copyfile(graph, ROOT / 'evidence/frontier306-native-update-graph.ir')
    dvm = list((STATE / ('work-' + run) / 'graph_kernel_dump').glob('dvm_kernel_*.txt'))
    if len(dvm) != 1:
        raise ValueError('Expected one native DVM program')
    shutil.copyfile(dvm[0], ROOT / 'evidence/frontier306-native-dvm.txt')
    report = {
        'complete': True, 'fidelity': checks, 'comparisons': comparisons,
        'native_statistics_sha256': digest(csv_path), 'native_graph_sha256': digest(graph),
        'native_dvm_sha256': digest(dvm[0]), 'source_sha256': digest(Path(__file__)),
        'observation_limitation': state['observation_limitation'],
        'scope': 'Original HP and MF replays reproduce their entire target weight/momentum. '
                 'Candidate uses only its own live input, and reproduces MF target state. '
                 'Native normalization is compared through full composed hashes; no tensor arrays saved.',
    }
    (ROOT / 'evidence/frontier306-muon-boundaries.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'fidelity': True, 'comparisons': comparisons}))


if __name__ == '__main__':
    main()
