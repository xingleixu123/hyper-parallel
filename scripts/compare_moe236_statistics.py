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

"""Compare MTP routing hashes against already faithful native graph statistics."""
import csv
import hashlib
import json
from pathlib import Path
import re

from compare_vocab_optimizer_trace import fidelity, read_rows

ROOT = Path('/alignment/phase12')
HP = 'hp-jtf637-global-sqrt-moe236'
MF = 'mf-native-boundaries236'
GRAPH_SHA256 = '0cc512eb0d35dcdbd891da45eb3720e0d7bcb4b22f54b712aecd2e35ff279b15'


def digest(path: Path) -> str:
    """Hash evidence without tensor values."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Require unchanged trajectories and match operands rather than debug scope names."""
    directory = ROOT / 'runs' / HP
    checks = fidelity(HP, 'hp-jtf637-global-sqrt400', 36)
    checks['exit_zero'] = (ROOT / 'runs' / (HP + '.exit')).read_text().strip() == '0'
    folder = ROOT / 'tools-f637'
    observer = json.loads((directory / 'MOE236_OBSERVER.json').read_text())
    checks['observer_source'] = observer['source_sha256'] == digest(folder / 'hp_moe236_hashes.py')
    checks['hash_source'] = observer['hash_helper_sha256'] == digest(folder / 'mindspore272_stat_hash.py')
    checks['no_arrays'] = observer['raw_arrays_written'] is False
    launch = json.loads((ROOT / 'evidence' / (HP + '-launch.json')).read_text())
    checks['entry_hash'] = launch['entry_sha256'] == digest(folder / 'capture_hp_moe236.py')
    checks['config_hash'] = launch['config_sha256'] == digest(ROOT / 'configs' / (HP + '.yaml'))
    checks['launcher_hash'] = launch['launcher_sha256'] == digest(ROOT / ('run-' + HP + '.sh'))
    base = 'hp-jtf637-global-sqrt-preclip236'
    checks['config_only_paths'] = (ROOT / 'configs' / (base + '.yaml')).read_text().replace(
        base, HP) == (ROOT / 'configs' / (HP + '.yaml')).read_text()
    for name in ['SOURCE_REVISION.json', 'PROBE_SOURCE_MANIFEST.json', 'RESTORE.json',
                 'WEIGHTED_DVM_FACTOR.json', 'WEIGHTED_DVM_CALLS.jsonl', 'NS_EPSILON229_FACTOR.json',
                 'GLOBAL_NPU_SQRT235_FACTOR.json', 'CE_TAIL226_FACTOR.json', 'NORM219_FACTOR.json',
                 'Q_NPU_SQRT222_FACTOR.json', 'CLIP_REDUCTION.jsonl']:
        checks['unchanged_' + name] = (directory / name).read_bytes() == (ROOT / 'runs' / base / name).read_bytes()
    manifest = json.loads((directory / 'PROBE_SOURCE_MANIFEST.json').read_text())
    checks['49_frozen_sources'] = len(manifest['files']) == 49 and all(
        row['sha256'] == digest(folder / row['name']) for row in manifest['files'])
    prior = json.loads((ROOT / 'evidence/frontier236-boundary-statistics-comparison.json').read_text())
    checks['native_statistics_fidelity'] = prior['complete'] and all(prior['fidelity']['mf'].values())
    root = ROOT / 'diagnostic-statistics' / MF / 'rank_0'
    graph, statistics = root / 'graphs/ms_output_trace_code_graph_1.ir', root / 'train236/1/35/statistic.csv'
    checks['graph_identity'] = digest(graph) == GRAPH_SHA256 == prior['native_graph_sha256']
    checks['statistics_identity'] = digest(statistics) == prior['native_statistics_sha256']
    if not all(checks.values()):
        raise ValueError(checks)
    nodes = {match[1]: match[2] for match in re.finditer(
        r'^  %(\d+)\([^\n]*?\) = (.*?)(?=^  %|^  Return)', graph.read_text(), re.M | re.S)}
    with statistics.open() as stream:
        rows = list(csv.DictReader(stream))
    observed = json.loads((directory / 'MOE236_HASHES.json').read_text())
    hp = {row['name']: row for row in observed['records']}
    if not observed['complete'] or len(hp) != len(observed['records']):
        raise ValueError('Incomplete or repeated observations')
    # These node IDs are bound to the exact verified final graph above.
    mapping = [
        ('router.x', '721', 'input', 0), ('router.weight', '721', 'input', 1),
        ('router.logits', '721', 'output', 0), ('sigmoid.logits', '721', 'output', 0),
        ('sigmoid.scores', '722', 'output', 0), ('normalization.scores', '722', 'output', 0),
        ('normalization.coefficients', '739', 'output', 1),
        ('normalization.coefficients_dy', '775', 'output', 0),
        ('sigmoid.dx', '798', 'output', 0), ('router.dy', '799', 'output', 0),
        ('router.dx', '800', 'output', 0), ('pair.router_dx', '800', 'output', 0),
        ('router.dw', '1702', 'output', 0),
    ]
    comparisons = []
    for name, node, direction, slot in mapping:
        opname = re.search(r'# Fullname with scope: \(([^\n]+)\)', nodes[node])[1]
        candidates = [row for row in rows if row['Op Name'] == opname and row['IO'] == direction
                      and int(row['Slot']) == slot]
        if len(candidates) != 1:
            raise ValueError(('Ambiguous native boundary', name, opname, len(candidates)))
        native, actual = candidates[0], hp[name]
        if native['Data Type'] != 'float32' or actual['dtype'] != 'torch.float32':
            raise ValueError(('Expected the same FP32 boundary', name))
        native_shape = json.loads(native['Shape'])
        if [n for n in native_shape if n != 1] != [n for n in actual['shape'] if n != 1]:
            if name != 'pair.router_dx' or actual['shape'] != [1, 4104, 5120]:
                raise ValueError(('Shape mismatch', name, actual['shape'], native_shape))
            continue
        comparisons.append({'name': name, 'node': node, 'native_op': opname, 'io': direction, 'slot': slot,
                            'shape': actual['shape'], 'exact': actual['native_dump_sha1'] == native['SHA1'],
                            'hp_sha1': actual['native_dump_sha1'], 'mf_sha1': native['SHA1']})
    report = {'complete': True, 'fidelity': checks, 'comparisons': comparisons,
              'captured_records': len(hp), 'source_sha256': digest(Path(__file__)),
              'graph_sha256': digest(graph), 'statistics_sha256': digest(statistics),
              'scope': 'Exact FP32 node boundaries on MTP router. Expert-major weighted buffers require layout mapping '
                       'before comparison; hash inequality does not quantify elementwise error.'}
    (ROOT / 'evidence/frontier236-moe-statistics-comparison.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'fidelity': checks, 'comparisons': comparisons}))


if __name__ == '__main__':
    main()
