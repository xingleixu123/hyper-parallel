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

"""Audit all selected vocabulary rows and their actual backward outputs."""
import hashlib
import json
from pathlib import Path

import numpy as np

from compare_preclip212 import compare_arrays
from compare_head206_interventions import source_sha
from compare_vocab_optimizer_trace import fidelity, read_rows

ROOT = Path('/alignment/phase12')
STATE = Path('/state/jt-recipe-20261014')
HP = 'hp-jtf637-qsqrt222-head226-r3'
MF = 'mf-head226'
BASE = 'hp-jtf637-qsqrt222-both226'


def main() -> None:
    """Require native trajectory fidelity before comparing retained observations."""
    checks = {'hp': fidelity(HP, BASE, 26), 'mf': fidelity(MF, 'mf-resume', 26)}
    for side, run in [('hp', HP), ('mf', MF)]:
        checks[side]['exit_zero'] = (ROOT / 'runs' / (run + '.exit')).read_text().strip() == '0'
    checks['hp']['config_exact_except_length_path'] = (ROOT / 'configs' / (BASE + '.yaml')).read_text().replace(
        BASE, HP).replace('train_iters: 226', 'train_iters: 226') == (ROOT / 'configs' / (HP + '.yaml')).read_text()
    checks['mf']['config_exact_except_path'] = (ROOT / 'configs/mf-preclip217.yaml').read_text().replace(
        'mf-preclip217', MF).replace('  stop_step: 17', '  stop_step: 26') == (ROOT / 'configs' / (MF + '.yaml')).read_text()
    for name in ['SOURCE_REVISION.json', 'PROBE_SOURCE_MANIFEST.json', 'HEAD_GEMM_ROUTE.json',
                 'SHARED_UP_FP32_FACTOR.json', 'ADAMW_ARCHIVE_ARITHMETIC.json', 'Q_NS_ADJACENT_FACTOR.json',
                 'ROUTER_NS_NPU_SQRT_FACTOR.json', 'WEIGHTED_DVM_FACTOR.json', 'PACKED_SEMANTICS_FACTOR.json', 'NORM219_FACTOR.json', 'Q_NPU_SQRT222_FACTOR.json']:
        checks['hp']['unchanged_' + name] = (ROOT / 'runs' / HP / name).read_bytes() == (ROOT / 'runs' / BASE / name).read_bytes()
    for name in ['HEAD_GEMM_ROUTE_CALLS.jsonl', 'Q_NS_ADJACENT_CALLS.jsonl', 'ROUTER_NS_NPU_SQRT_CALLS.jsonl',
                 'CLIP_REDUCTION.jsonl', 'WEIGHTED_DVM_CALLS.jsonl', 'PACKED_SEMANTICS_CALLS.jsonl', 'NORM219_CALLS.jsonl']:
        checks['hp']['exact26_' + name] = read_rows(ROOT / 'runs' / HP / name) == read_rows(ROOT / 'runs' / BASE / name)[:26]
    frozen = json.loads((ROOT / 'runs' / HP / 'PROBE_SOURCE_MANIFEST.json').read_text())
    checks['hp']['49_frozen_sources'] = len(frozen['files']) == 49 and all(
        row['sha256'] == source_sha(row['name']) for row in frozen['files'])
    hp_report = json.loads((ROOT / 'runs' / HP / 'HEAD226_OPERANDS.json').read_text())
    mf_rows = read_rows(ROOT / 'runs' / MF / 'HEAD226_OPERANDS.jsonl')
    mf_manifest = json.loads((STATE / MF / 'deep/manifest.json').read_text())
    hp_metadata = json.loads((ROOT / 'runs' / HP / 'HEAD226_OBSERVER.json').read_text())
    selection_path = ROOT / 'evidence/head226-selected-coordinates.json'
    selection_sha = hashlib.sha256(selection_path.read_bytes()).hexdigest()
    selection = json.loads(selection_path.read_text())
    checks['hp']['observer_source'] = hp_metadata['source_sha256'] == source_sha('hp_head226_observer_r3.py')
    checks['mf']['observer_source'] = mf_manifest['source_sha256'] == hashlib.sha256(
        (ROOT / 'tools/mf_head226_tap.py').read_bytes()).hexdigest()
    checks['hp']['selection_exact'] = hp_metadata['selection_sha256'] == selection_sha
    checks['mf']['selection_exact'] = mf_manifest['selection_sha256'] == selection_sha
    checks['mf']['all26_observations'] = [r['step'] for r in mf_rows] == list(range(201, 227))
    checks['hp']['all10_boundaries'] = len(hp_report['records']) == 10
    checks['mf']['all10_boundaries'] = len({(r['name'], r['kind']) for r in mf_rows[-1]['records']}) == 10
    if not all(all(row.values()) for row in checks.values()):
        raise ValueError(checks)
    arrays = {}
    for family, records, directory, key_name in [
            ('hp', hp_report['records'], STATE / HP / 'head', 'call'),
            ('mf', mf_rows[-1]['records'], STATE / MF / 'deep', 'name')]:
        for record in records:
            key = family, record[key_name], record['kind']
            array = np.load(directory / record['file'], mmap_mode='r')
            if list(array.shape) != record['shape'] or hashlib.sha256(array.tobytes()).hexdigest() != record['sha256']:
                raise ValueError('Observed head operand changed')
            if key in arrays and not np.array_equal(arrays[key], array):
                raise ValueError('Head recomputation differs')
            arrays[key] = array
    observations = []
    mapping = {}
    for module in mf_manifest['modules']:
        name = module['name']
        candidates = [call for call in [1, 2] if np.array_equal(arrays['mf', name, 'x'], arrays['hp', call, 'x'])]
        if len(candidates) != 1:
            raise ValueError('Head branch mapping not unique')
        call = candidates[0]
        mapping[name] = call
        for mf_kind, hp_kind in [('x', 'x'), ('logits', 'logits'), ('dy_output', 'dy'), ('dy_matmul', 'dy'), ('dx', 'dx')]:
            observations.append({'mf_module': name, 'hp_call': call, 'mf_kind': mf_kind, 'hp_kind': hp_kind,
                                 'numerically_different': int(np.count_nonzero(arrays['hp', call, hp_kind] != arrays['mf', name, mf_kind])),
                                 **compare_arrays(arrays['hp', call, hp_kind], arrays['mf', name, mf_kind])})
    main_name = next(name for name in mapping if '.mtp.' not in name)
    call = mapping[main_name]
    hp_dx = arrays['hp', call, 'dx']
    mf_dx = arrays['mf', main_name, 'dx']
    hp_dy = arrays['hp', call, 'dy']
    mf_dy = arrays['mf', main_name, 'dy_matmul']
    weight = arrays['hp', call, 'weight_columns']
    full_audit = json.loads((ROOT / 'evidence/backward226-comparison.json').read_text())
    if not full_audit['complete']:
        raise ValueError('Full boundary fidelity missing')
    hp_full = json.loads((ROOT / 'runs/hp-jtf637-qsqrt222-backward226/BACKWARD226_BOUNDARIES.json').read_text())
    mf_full = read_rows(ROOT / 'runs/mf-backward226/BACKWARD226_BOUNDARIES.jsonl')[-1]
    for family, records, module_name, actual, directory in [
            ('hp', hp_full['records'], 'model.norm', hp_dx, STATE / 'hp-jtf637-qsqrt222-backward226/norms'),
            ('mf', mf_full['records'], 'model.decoder.final_layernorm', mf_dx, STATE / 'mf-backward226/deep')]:
        record = next(r for r in records if r['name'] == module_name and r['kind'] == 'dy')
        full = np.load(directory / record['file'], mmap_mode='r').reshape(4096, 5120)
        checks[family]['crossrun_dx_exact'] = np.array_equal(actual, full[selection['token_rows']])
    if not all(all(row.values()) for row in checks.values()):
        raise ValueError(checks)
    precision = []
    if np.array_equal(hp_dy, mf_dy):
        for token, column in selection['coordinates']:
            row = selection['token_rows'].index(token)
            col = selection['hidden_columns'].index(column)
            reference = float(np.dot(hp_dy[row].astype(np.float64), weight[:, col].astype(np.float64)))
            left, right = float(hp_dx[row, column]), float(mf_dx[row, column])
            precision.append({'token': token, 'column': column,
                              'hp_error_fp64': abs(left - reference), 'mf_error_fp64': abs(right - reference),
                              'closer': 'hp' if abs(left-reference) < abs(right-reference) else 'mf'})
    report = {'complete': True, 'fidelity': checks, 'mapping_from_actual_input': mapping,
              'selection': selection, 'observations': observations, 'selected_dx_fp64': precision,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'scope': 'Four complete token rows, all vocabulary entries and all nine full-observation dx differences; not a full-logits comparison.',
              'weight_basis': 'Actual HP consumed BF16 weight columns; previous full-model audit confirms both baseline step225 entire weights exact; current probes preserve both26step trajectories.'}
    (ROOT / 'evidence/head226-selected-comparison.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == '__main__':
    main()
