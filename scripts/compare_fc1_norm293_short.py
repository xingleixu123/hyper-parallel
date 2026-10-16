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

"""Compare complete preclip hashes using native graph edges and statistics."""
import csv
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import numpy as np

from compare_vocab_optimizer_trace import fidelity, read_rows

ROOT = Path('/alignment/phase12')
HP = 'hp-jte213-fc1norm-preclip294'
MF = 'mf-native-stat294'
DUMP = ROOT / 'diagnostic-statistics' / MF / 'rank_0'


def digest(path: Path) -> str:
    """Hash a bounded evidence file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scalar_sha1(value: float) -> str:
    """Encode the exact little-endian FP32 scalar, independent of printed precision."""
    return hashlib.sha1(np.asarray(value, dtype='<f4').tobytes()).hexdigest()


def graph_mapping(path: Path) -> tuple[list[dict[str, Any]], str, str]:
    """Trace AddN inputs through ExpandDims to gradient/square-sum producers."""
    text = path.read_text()
    nodes = {match[1]: match[2] for match in re.finditer(
        r'^  %(\d+)\([^\n]*?\) = (.*?)(?=^  %|^  Return)', text, re.M | re.S)}
    candidates = [node for node in nodes.values() if node.startswith('PrimFunc_AddN(')
                  and '/clip_grad_norm-ClipGradNorm/AddN-' in node]
    if len(candidates) != 1:
        raise ValueError('Native clip AddN is ambiguous')
    addn = candidates[0]
    references = re.findall(r'%(\d+)', addn.split(' primitive_attrs:')[0])
    order = json.loads((ROOT / 'configs/mf-gradient-layout-order.json').read_text())
    if len(references) != 46 or len(order) != 46:
        raise ValueError('Expected the full 46-input native norm')
    mapping = []
    for index, (reference, parameter) in enumerate(zip(references, order)):
        expand = nodes[reference]
        if not expand.startswith('PrimFunc_ExpandDims('):
            raise ValueError('Unexpected norm scalar expansion')
        if f'/clip_grad_norm-ClipGradNorm/ExpandDims-op{index})' not in expand:
            raise ValueError('Native ordered norm inputs changed')
        item_id = re.search(r'ExpandDims\(%(\d+),', expand)[1]
        item = re.match(r'TupleGetItem\(%(\d+), I64\(1\)\)', nodes[item_id])
        if item is None:
            raise ValueError('Expected square sum in tuple slot 1')
        producer = nodes[item[1]]
        if not producer.startswith('GraphKernel_Mul_Mul_ReduceSum_split('):
            raise ValueError('Unexpected native gradient and sum producer')
        opname = re.search(r'# Fullname with scope: \(([^\n]+)\)', producer)[1]
        mapping.append({**parameter, 'index': index, 'native_op': opname,
                        'gradient_output_slot': 0, 'square_sum_output_slot': 1})
    addn_name = re.search(r'# Fullname with scope: \(([^\n]+)\)', addn)[1]
    sqrt = [node for node in nodes.values()
            if node.startswith('GraphKernel_Sqrt_GreaterEqual_Select_Add_Div_Minimum_split(')]
    if len(sqrt) != 1:
        raise ValueError('Native clip norm/scale kernel is ambiguous')
    sqrt_name = re.search(r'# Fullname with scope: \(([^\n]+)\)', sqrt[0])[1]
    return mapping, addn_name, sqrt_name


def main() -> None:
    """Require unchanged trajectories before locating the next gradient boundary."""
    checks = {}
    for family, run, base in [('hp', HP, 'hp-jte213-fc1norm400'), ('mf', MF, 'mf-resume')]:
        directory = ROOT / 'runs' / run
        check = fidelity(run, base, 94)
        check['exit_zero'] = (ROOT / 'runs' / (run + '.exit')).read_text().strip() == '0'
        prior = 'hp-jte213-fc1norm400' if family == 'hp' else 'mf-modelhash235'
        config = ROOT / 'configs' / (run + '.yaml')
        check['config_only_paths'] = (ROOT / 'configs' / (prior + '.yaml')).read_text().replace(
            prior, run).replace('  train_iters: 400', '  train_iters: 294').replace(
                '  stop_step: 35', '  stop_step: 94') == config.read_text()
        launch = json.loads((ROOT / 'evidence' / (run + '-launch.json')).read_text())
        folder = ROOT / ('tools-f637' if family == 'hp' else 'tools')
        entry = 'capture_hp_fc1_preclip294.py' if family == 'hp' else 'capture_mf_model_hash294.py'
        check['config_hash'] = launch['config_sha256'] == digest(config)
        check['launcher_hash'] = launch['launcher_sha256'] == digest(ROOT / ('run-' + run + '.sh'))
        entry_folder = ROOT / 'tools-e213' if family == 'hp' else folder
        check['entry_hash'] = launch['entry_sha256'] == digest(entry_folder / entry)
        if family == 'hp':
            manifest = json.loads((directory / 'PROBE_SOURCE_MANIFEST.json').read_text())
            check['49_frozen_sources'] = len(manifest['files']) == 49 and all(
                row['sha256'] == digest(folder / row['name']) for row in manifest['files'])
            for name in ['RESTORE.json', 'SOURCE_REVISION.json', 'PROBE_SOURCE_MANIFEST.json',
                         'WEIGHTED_DVM_FACTOR.json', 'CE_TAIL226_FACTOR.json', 'NS_EPSILON229_FACTOR.json',
                         'NORM219_FACTOR.json', 'Q_NPU_SQRT222_FACTOR.json', 'GLOBAL_NPU_SQRT235_FACTOR.json', 'QUP258_FACTOR.json', 'FC1293_FACTOR.json']:
                check['unchanged_' + name] = (directory / name).read_bytes() == (
                    ROOT / 'runs' / base / name).read_bytes()
            check['weighted_both_calls'] = read_rows(directory / 'WEIGHTED_DVM_CALLS.jsonl') == [
                {'step': step, 'calls': {'main': step - 200, 'mtp': step - 200}} for step in range(201, 295)]
            check['unchanged_all_clip_calls'] = read_rows(directory / 'CLIP_REDUCTION.jsonl') == read_rows(
                ROOT / 'runs' / base / 'CLIP_REDUCTION.jsonl')[:94]
            for step in [293, 294]:
                observed = json.loads((directory / f'PRECLIP-{step}.json').read_text())
                check[f'preclip_{step}_source'] = observed['source_sha256'] == digest(
                    folder / 'hp_preclip294_native_hashes.py')
                check[f'preclip_{step}_helper'] = observed['helper_sha256'] == digest(
                    folder / 'frontier212_array_io.py')
                check[f'preclip_{step}_native_hash_helper'] = observed['native_hash_helper_sha256'] == digest(
                    folder / 'mindspore272_stat_hash.py')
                check[f'preclip_{step}_complete'] = observed['complete'] and len(observed['records']) == 46
        else:
            observer = json.loads((directory / 'MODEL_HASH_OBSERVER.json').read_text())
            check['converter_hash'] = observer['converter_sha256'] == digest(
                ROOT / 'hyper-parallel/scripts/checkpoint_conversion/convert_mf_ckpt_to_hp.py')
            check['dump_config_hash'] = launch['dump_config_sha256'] == digest(
                ROOT / 'configs' / (run + '.json'))
            check['no_array_files'] = not any(DUMP.rglob('*.npy')) and not any(DUMP.rglob('*.npz'))
        observer = json.loads((directory / 'MODEL_HASH_OBSERVER.json').read_text())
        check['model_observer_hash'] = observer['source_sha256'] == digest(folder / (family + '_tail294_model_hash.py'))
        check['model_helper_hash'] = observer['helper_sha256'] == digest(folder / 'model_hash_stream.py')
        checks[family] = check
    if not all(all(values.values()) for values in checks.values()):
        raise ValueError(checks)
    graph = DUMP / 'graphs/ms_output_trace_code_graph_1.ir'
    mapping, addn, sqrt = graph_mapping(graph)
    clips = {row['step']: row for row in read_rows(ROOT / 'runs' / HP / 'CLIP_REDUCTION.jsonl')}
    metrics = {row['step']: row for row in read_rows(ROOT / 'runs' / MF / 'metrics.jsonl')}
    comparisons = []
    for step in [293, 294]:
        csv_path = DUMP / 'train294' / '1' / str(step - 201) / 'statistic.csv'
        with csv_path.open() as stream:
            rows = list(csv.DictReader(stream))
        norm_rows = [row for row in rows if row['Op Name'] == sqrt]
        if len(norm_rows) != 2:
            raise ValueError('Ambiguous native clipping invocation')
        cutoff = max(int(row['Timestamp']) for row in norm_rows)
        selected = {item['native_op'] for item in mapping} | {addn, sqrt}
        # Subgraph statistics share a CSV; stop at clipping to exclude optimizer name reuse.
        selected_rows = [row for row in rows if row['Op Name'] in selected
                         and int(row['Timestamp']) <= cutoff]
        indexed = {(row['Op Name'], row['IO'], int(row['Slot'])): row for row in selected_rows}
        if len(indexed) != len(selected_rows):
            raise ValueError('Duplicate native preclip tensor identities')
        observed = json.loads((ROOT / 'runs' / HP / f'PRECLIP-{step}.json').read_text())
        hp_records = {row['name']: row for row in observed['records']}
        norm = indexed[sqrt, 'output', 0]
        if norm['SHA1'] != scalar_sha1(metrics[step]['outputs'][4][0]):
            raise ValueError('Native Dump iteration does not match the actual training norm')
        records = []
        for item in mapping:
            grad = indexed[item['native_op'], 'output', 0]
            square = indexed[item['native_op'], 'output', 1]
            hp = hp_records[item['name']]
            if json.loads(grad['Shape']) != item['shape'] or grad['Data Type'] != 'float32':
                raise ValueError('Native gradient layout differs from the verified parameter order')
            if hp['shape'] != item['shape']:
                raise ValueError('HP logical gradient shape differs')
            records.append({**item, 'hp_native_dump_sha1': hp['native_dump_sha1'], 'mf_gradient_sha1': grad['SHA1'],
                            'gradient_exact': hp['native_dump_sha1'] == grad['SHA1'],
                            'square_sum_exact': scalar_sha1(clips[step]['per_parameter_square_sums'][
                                item['index']]) == square['SHA1'],
                            'hp_square_sum': clips[step]['per_parameter_square_sums'][item['index']],
                            'mf_square_sum_sha1': square['SHA1'],
                            'mf_printed_statistics': {key: grad[key] for key in
                                                      ['Min Value', 'Max Value', 'L2Norm Value']}})
        total = indexed[addn, 'output', 0]
        summary = {
            'step': step, 'dump_iteration': step - 201, 'native_norm_matches_training_sha1': True,
            'gradient_exact_count': sum(row['gradient_exact'] for row in records),
            'square_sum_exact_count': sum(row['square_sum_exact'] for row in records),
            'total_square_sum_exact': scalar_sha1(clips[step]['square_sum']) == total['SHA1'],
            'global_norm_exact': scalar_sha1(clips[step]['norm']) == norm['SHA1'],
            'hp_norm': clips[step]['norm'], 'mf_norm': metrics[step]['outputs'][4][0],
            'hp_square_sum': clips[step]['square_sum'], 'mf_square_sum_sha1': total['SHA1'],
            'mf_norm_sha1': norm['SHA1'], 'mf_scale_sha1': indexed[sqrt, 'output', 1]['SHA1'],
            'records': records, 'statistics_sha256': digest(csv_path), 'statistics_rows': len(rows),
        }
        comparisons.append(summary)
    models = {}
    for family, run in [('hp', HP), ('mf', MF)]:
        rows = read_rows(ROOT / 'runs' / run / 'MODEL_HASHES.jsonl')
        if [row['step'] for row in rows] != [293, 294]:
            raise ValueError('Incomplete model boundary capture')
        models[family] = {}
        for row in rows:
            records = {item['name']: item for item in row['records']}
            if len(records) != 48 or len(row['records']) != 48:
                raise ValueError('Expected all 48 model records')
            models[family][row['step']] = records
    model_comparisons = []
    for step in [293, 294]:
        left, right = models['hp'][step], models['mf'][step]
        if set(left) != set(right) or any(left[name]['shape'] != right[name]['shape'] for name in left):
            raise ValueError('Model identities or layouts differ')
        model_comparisons.append({'step': step, 'different': {
            key: [name for name in left if left[name][key] != right[name][key]]
            for key in ['fp32_sha256', 'bf16_as_fp32_sha256']}})
    report = {
        'complete': True, 'fidelity': checks, 'comparisons': comparisons, 'models': model_comparisons,
        'source_sha256': digest(Path(__file__)), 'native_graph_sha256': digest(graph),
        'scope': 'Native composed SHA1 over all elements of 46 preclip gradients and square sums; printed statistics '
                 'are not exact values. Iteration mapping is checked against actual FP32 norm hashes. '
                 'Hash inequality alone does not quantify gradient differences.',
    }
    (ROOT / 'evidence/frontier293-fc1-native-statistics-comparison.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'fidelity': checks, 'models': model_comparisons, 'comparisons': [
        {key: value for key, value in row.items() if key != 'records'} for row in comparisons]}))


if __name__ == '__main__':
    main()
