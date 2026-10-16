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

"""Bind in-memory reduction precision to complete native Dump inputs and outputs."""
import csv
import hashlib
import json
from pathlib import Path
import re

from compare_vocab_optimizer_trace import fidelity

ROOT = Path('/alignment/phase12')
RUN = 'hp-jtf637-weighted-precision236'
CONTROL = 'hp-jtf637-global-sqrt-moe236'


def digest(path: Path) -> str:
    """Hash a bounded evidence file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Reject altered training before comparing the same-input FP64 reference."""
    directory = ROOT / 'runs' / RUN
    checks = fidelity(RUN, CONTROL, 36)
    checks['exit_zero'] = (ROOT / 'runs' / (RUN + '.exit')).read_text().strip() == '0'
    folder = ROOT / 'tools-f637'
    precision = json.loads((directory / 'WEIGHTED236_PRECISION.json').read_text())
    checks['precision_source'] = precision['source_sha256'] == digest(folder / 'hp_weighted236_precision.py')
    checks['base_source'] = precision['base_observer_sha256'] == digest(folder / 'hp_moe236_hashes.py')
    checks['hash_source'] = precision['hash_helper_sha256'] == digest(folder / 'mindspore272_stat_hash.py')
    checks['no_arrays'] = precision['raw_arrays_written'] is False
    launch = json.loads((ROOT / 'evidence' / (RUN + '-launch.json')).read_text())
    checks['entry_hash'] = launch['entry_sha256'] == digest(folder / 'capture_hp_weighted236_precision.py')
    checks['config_hash'] = launch['config_sha256'] == digest(ROOT / 'configs' / (RUN + '.yaml'))
    checks['launcher_hash'] = launch['launcher_sha256'] == digest(ROOT / ('run-' + RUN + '.sh'))
    checks['config_only_paths'] = (ROOT / 'configs' / (CONTROL + '.yaml')).read_text().replace(
        CONTROL, RUN) == (ROOT / 'configs' / (RUN + '.yaml')).read_text()
    for name in ['SOURCE_REVISION.json', 'PROBE_SOURCE_MANIFEST.json', 'RESTORE.json',
                 'WEIGHTED_DVM_FACTOR.json', 'WEIGHTED_DVM_CALLS.jsonl', 'NS_EPSILON229_FACTOR.json',
                 'GLOBAL_NPU_SQRT235_FACTOR.json', 'CE_TAIL226_FACTOR.json', 'NORM219_FACTOR.json',
                 'Q_NPU_SQRT222_FACTOR.json', 'CLIP_REDUCTION.jsonl', 'MOE236_HASHES.json']:
        checks['unchanged_' + name] = (directory / name).read_bytes() == (
            ROOT / 'runs' / CONTROL / name).read_bytes()
    manifest = json.loads((directory / 'PROBE_SOURCE_MANIFEST.json').read_text())
    checks['49_frozen_sources'] = len(manifest['files']) == 49 and all(
        row['sha256'] == digest(folder / row['name']) for row in manifest['files'])
    prior = json.loads((ROOT / 'evidence/frontier236-moe-statistics-comparison.json').read_text())
    checks['prior_statistics_fidelity'] = prior['complete'] and all(prior['fidelity'].values())
    root = ROOT / 'diagnostic-statistics/mf-native-boundaries236/rank_0'
    graph, statistics = root / 'graphs/ms_output_trace_code_graph_1.ir', root / 'train236/1/35/statistic.csv'
    checks['graph_identity'] = digest(graph) == prior['graph_sha256']
    checks['statistics_identity'] = digest(statistics) == prior['statistics_sha256']
    if not all(checks.values()):
        raise ValueError(checks)
    nodes = {match[1]: match[2] for match in re.finditer(
        r'^  %(\d+)\([^\n]*?\) = (.*?)(?=^  %|^  Return)', graph.read_text(), re.M | re.S)}
    with statistics.open() as stream:
        rows = list(csv.DictReader(stream))
    mapping = [('weighted.outputs_bf16', '771', 'input', 0, 'bfloat16', 4104 * 2 * 5120),
               ('weighted.dy_bf16_single_choice', '771', 'input', 1, 'bfloat16', 4104 * 5120),
               ('weighted.weights_bf16', '771', 'input', 2, 'bfloat16', 4104 * 2),
               ('actual_coefficients_dy_fp32', '775', 'output', 0, 'float32', 4096 * 2),
               ('candidate_coefficients_dy_fp32', '775', 'output', 0, 'float32', 4096 * 2)]
    comparisons = []
    for name, node, direction, slot, dtype, elements in mapping:
        opname = re.search(r'# Fullname with scope: \(([^\n]+)\)', nodes[node])[1]
        selected = [row for row in rows if row['Op Name'] == opname and row['IO'] == direction
                    and int(row['Slot']) == slot]
        if len(selected) != 1 or selected[0]['Data Type'] != dtype:
            raise ValueError('Ambiguous native weighted-product boundary: ' + name)
        native = selected[0]
        actual_elements = 1
        for size in json.loads(native['Shape']):
            actual_elements *= size
        if actual_elements != elements:
            raise ValueError('Unexpected native weighted-product dimensions')
        actual = precision['canonical_hashes'][name]
        comparisons.append({'name': name, 'node': node, 'io': direction, 'slot': slot,
                            'native_shape': json.loads(native['Shape']), 'dtype': dtype,
                            'exact': actual == native['SHA1'], 'hp_sha1': actual, 'mf_sha1': native['SHA1']})
    same_inputs = all(row['exact'] for row in comparisons[:3])
    candidate_matches_mf = comparisons[-1]['exact']
    if not same_inputs or not candidate_matches_mf or comparisons[-2]['exact']:
        raise ValueError({'same_inputs': same_inputs, 'candidate_matches_mf': candidate_matches_mf,
                          'comparisons': comparisons})
    report = {'complete': True, 'fidelity': checks, 'comparisons': comparisons, 'same_inputs': same_inputs,
              'candidate_matches_mf': candidate_matches_mf, 'precision': precision['precision'],
              'raw_arrays_written': False, 'source_sha256': digest(Path(__file__)),
              'scope': 'Same native BF16 inputs verified over all elements; candidate coefficient gradients exactly '
                       'match native MF. FP64 reference retains the original BF16 product rounding. '
                       'No raw tensor values or reference substitutions are used in training.'}
    (ROOT / 'evidence/frontier236-weighted-precision-audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == '__main__':
    main()
