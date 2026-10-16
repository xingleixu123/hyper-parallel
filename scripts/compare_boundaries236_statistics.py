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

"""Audit native forward and final-norm derivative fingerprints without tensor dumps."""
import csv
import hashlib
import json
from pathlib import Path
import re

from compare_vocab_optimizer_trace import fidelity, read_rows

ROOT = Path('/alignment/phase12')
HP = 'hp-jtf637-global-sqrt-boundaries236-r2'
MF = 'mf-native-boundaries236'
DUMP = ROOT / 'diagnostic-statistics' / MF / 'rank_0'


def digest(path: Path) -> str:
    """Hash an evidence file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scopes() -> dict[str, str]:
    """Map logical HP norms to their native MF forward scopes."""
    decoder = 'decoder-TransformerBlock/'
    mtp = 'mtp-MultiTokenPredictionBlock/layers-CellList/0-MultiTokenPredictionLayer/'
    result = {'model.norm': decoder + 'final_layernorm-FusedRMSNorm/',
              'mtp.layers.0.final_layernorm': mtp + 'final_layernorm-FusedRMSNorm/',
              'mtp.layers.0.enorm': mtp + 'enorm-FusedRMSNorm/',
              'mtp.layers.0.hnorm': mtp + 'hnorm-FusedRMSNorm/'}
    pairs = [(f'model.layers.{i}', decoder + f'layers-CellList/{i}-TransformerLayer/') for i in [0, 1]]
    pairs.append(('mtp.layers.0.transformer_layer', mtp + 'transformer_layer-TransformerLayer/'))
    for hp, native in pairs:
        for left, right in [
                ('input_layernorm', 'input_layernorm-FusedRMSNorm/'),
                ('post_attention_layernorm', 'pre_mlp_layernorm-FusedRMSNorm/'),
                ('self_attn.q_a_layernorm', 'self_attention-MLASelfAttention/q_layernorm-FusedRMSNorm/'),
                ('self_attn.kv_a_layernorm', 'self_attention-MLASelfAttention/kv_layernorm-FusedRMSNorm/')]:
            result[hp + '.' + left] = native + right
    return result


def main() -> None:
    """Require exact independent trajectories before comparing typed graph boundaries."""
    checks = {}
    for family, run, base in [('hp', HP, 'hp-jtf637-global-sqrt400'), ('mf', MF, 'mf-resume')]:
        folder = ROOT / ('tools-f637' if family == 'hp' else 'tools')
        directory = ROOT / 'runs' / run
        check = fidelity(run, base, 36)
        check['exit_zero'] = (ROOT / 'runs' / (run + '.exit')).read_text().strip() == '0'
        launch = json.loads((ROOT / 'evidence' / (run + '-launch.json')).read_text())
        entry = 'capture_hp_boundaries236.py' if family == 'hp' else 'capture_mf_model_hash236.py'
        check['entry_hash'] = launch['entry_sha256'] == digest(folder / entry)
        check['config_hash'] = launch['config_sha256'] == digest(ROOT / 'configs' / (run + '.yaml'))
        check['launcher_hash'] = launch['launcher_sha256'] == digest(ROOT / ('run-' + run + '.sh'))
        prior = 'hp-jtf637-global-sqrt-preclip236' if family == 'hp' else 'mf-native-stat236'
        check['config_only_paths'] = (ROOT / 'configs' / (prior + '.yaml')).read_text().replace(
            prior, run) == (ROOT / 'configs' / (run + '.yaml')).read_text()
        if family == 'hp':
            manifest = json.loads((directory / 'PROBE_SOURCE_MANIFEST.json').read_text())
            check['49_frozen_sources'] = len(manifest['files']) == 49 and all(
                row['sha256'] == digest(folder / row['name']) for row in manifest['files'])
            for name in ['RESTORE.json', 'SOURCE_REVISION.json', 'PROBE_SOURCE_MANIFEST.json',
                         'CE_TAIL226_FACTOR.json', 'NS_EPSILON229_FACTOR.json',
                         'NORM219_FACTOR.json', 'Q_NPU_SQRT222_FACTOR.json', 'GLOBAL_NPU_SQRT235_FACTOR.json']:
                check['unchanged_' + name] = (directory / name).read_bytes() == (
                    ROOT / 'runs' / base / name).read_bytes()
            check['all_clip_calls'] = read_rows(directory / 'CLIP_REDUCTION.jsonl') == read_rows(
                ROOT / 'runs' / base / 'CLIP_REDUCTION.jsonl')[:36]
            for file, source in [('FORWARD236.json', 'hp_forward236_hashes.py'),
                                 ('BACKWARD236_OBSERVER.json', 'hp_backward236_hashes.py')]:
                observer = json.loads((directory / file).read_text())
                check[file + '_source'] = observer['source_sha256'] == digest(folder / source)
                check[file + '_helper'] = observer['hash_helper_sha256'] == digest(folder / 'mindspore272_stat_hash.py')
                check[file + '_no_arrays'] = observer['raw_arrays_written'] is False
            check['forward_coverage'] = len(json.loads((directory / 'FORWARD236.json').read_text())['records']) == 42
            check['backward_coverage'] = len(json.loads((directory / 'BACKWARD236_BOUNDARIES.json').read_text())['records']) == 16
        else:
            check['dump_config_hash'] = launch['dump_config_sha256'] == digest(ROOT / 'configs' / (run + '.json'))
            check['model_hashes_exact_reference_probe'] = (directory / 'MODEL_HASHES.jsonl').read_bytes() == (
                ROOT / 'runs/mf-native-stat236/MODEL_HASHES.jsonl').read_bytes()
            check['no_dump_arrays'] = not any(DUMP.rglob('*.npy')) and not any(DUMP.rglob('*.npz'))
        checks[family] = check
    if not all(all(values.values()) for values in checks.values()):
        raise ValueError(checks)
    graph = DUMP / 'graphs/ms_output_trace_code_graph_1.ir'
    nodes = {match[1]: match[2] for match in re.finditer(
        r'^  %(\d+)\([^\n]*?\) = (.*?)(?=^  %|^  Return)', graph.read_text(), re.M | re.S)}
    names = {node: re.search(r'# Fullname with scope: \(([^\n]+)\)', text)[1] for node, text in nodes.items()
             if '# Fullname with scope:' in text}
    csv_path = DUMP / 'train236/1/35/statistic.csv'
    with csv_path.open() as stream:
        rows = list(csv.DictReader(stream))
    indexed = {}
    for row in rows:
        indexed.setdefault((row['Op Name'], row['IO'], int(row['Slot'])), []).append(row)
    forward = {(row['name'], row['kind']): row for row in json.loads(
        (ROOT / 'runs' / HP / 'FORWARD236.json').read_text())['records']}
    backward = {(row['name'], row['kind']): row for row in json.loads(
        (ROOT / 'runs' / HP / 'BACKWARD236_BOUNDARIES.json').read_text())['records']}
    comparisons = []

    def compare(hp, node, direction, slot, boundary):
        candidates = indexed.get((names[node], direction, slot), [])
        if len(candidates) != 1:
            raise ValueError(('Expected unique native norm boundary', node, names[node], direction, slot, len(candidates)))
        native = candidates[0]
        if [n for n in hp['shape'] if n != 1] != [n for n in json.loads(native['Shape']) if n != 1]:
            raise ValueError(('Non-unit layout differs', hp['name'], hp['shape'], native['Shape']))
        key = 'fp32_native_dump_sha1' if native['Data Type'] == 'float32' else 'native_dump_sha1'
        if native['Data Type'] not in ['float32', 'bfloat16']:
            raise ValueError('Unsupported native boundary dtype')
        if native['Data Type'] == 'bfloat16' and hp['dtype'] != 'torch.bfloat16':
            raise ValueError(('A BF16 cast needs explicit HP hash', boundary))
        comparisons.append({'name': hp['name'], 'kind': hp['kind'], 'boundary': boundary,
                            'native_node': node, 'native_op': names[node], 'io': direction, 'slot': slot,
                            'hp_dtype': hp['dtype'], 'mf_dtype': native['Data Type'], 'shape': hp['shape'],
                            'exact': hp[key] == native['SHA1'], 'hp_sha1': hp[key], 'mf_sha1': native['SHA1']})

    def cast_output(parent, tensor_slot):
        items = [n for n, text in nodes.items() if text.startswith(f'TupleGetItem(%{parent}, I64({tensor_slot}))')]
        casts = [n for n, text in nodes.items() if any(
            text.startswith(f'PrimFunc_Cast(%{item}, I64(45))') for item in items)]
        return casts

    mapped = {}
    for hp_name, suffix in scopes().items():
        selected = [n for n, text in nodes.items() if text.startswith('PrimFunc_RmsNorm(')
                    and '/model-GPTModel/' + suffix + 'RmsNorm-' in names[n]]
        if not selected:
            raise ValueError('No native scope for ' + hp_name)
        mapped[hp_name] = selected
        for n in selected:
            compare(forward[hp_name, 'input'], n, 'input', 0, 'norm_forward_input')
            for cast in cast_output(n, 0):
                if forward[hp_name, 'output']['dtype'] == 'torch.bfloat16':
                    compare(forward[hp_name, 'output'], cast, 'output', 0, 'norm_forward_bf16_output')
    for hp_name in ['model.norm', 'mtp.layers.0.final_layernorm']:
        for n in mapped[hp_name]:
            for kind, io, slot in [('x', 'input', 0), ('weight', 'input', 1),
                                   ('y_fp32', 'output', 0), ('inverse_rms', 'output', 1)]:
                compare(backward[hp_name, kind], n, io, slot, 'final_norm_native_forward')
            refs = re.findall(r'%(\d+)', nodes[n].split(' primitive_attrs:')[0])[:2]
            gradients = [g for g, text in nodes.items() if text.startswith('PrimFunc_RmsNormGrad(')
                         and re.findall(r'%(\d+)', text.split(' primitive_attrs:')[0])[1::2] == refs]
            for gradient in gradients:
                compare(backward[hp_name, 'dy'], gradient, 'input', 0, 'final_norm_backward_dy')
                dx = backward[hp_name, 'dx']
                if dx['dtype'] == 'torch.float32':
                    compare(dx, gradient, 'output', 0, 'final_norm_backward_dx')
                else:
                    casts = cast_output(gradient, 0)
                    if len(casts) != 1:
                        raise ValueError('Expected final norm gradient cast')
                    compare(dx, casts[0], 'output', 0, 'final_norm_backward_dx')
    report = {'complete': True, 'fidelity': checks, 'comparisons': comparisons,
              'source_sha256': digest(Path(__file__)), 'native_graph_sha256': digest(graph),
              'native_statistics_sha256': digest(csv_path), 'native_statistics_bytes': csv_path.stat().st_size,
              'unmatched_backward_moe': [row for row in backward.values() if row['name'].endswith('.mlp')],
              'scope': 'Typed norm graph boundaries only; hash mismatch does not quantify elementwise error. '
                       'Repeated native recomputation must independently match. Other captured HP boundaries remain available.'}
    (ROOT / 'evidence/frontier236-boundary-statistics-comparison.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'fidelity': checks, 'compared': len(comparisons),
                      'different': [row for row in comparisons if not row['exact']]}))


if __name__ == '__main__':
    main()
