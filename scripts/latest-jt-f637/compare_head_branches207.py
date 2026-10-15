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

"""Compare actual207 branch operands, GEMM outputs and full postclip reconstruction."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from compare_head206_interventions import model_hashes
from compare_vocab_optimizer_trace import fidelity

ROOT = Path('/alignment/phase12')
STATE = Path('/state/jt-recipe-20261014')
MF = 'mf-head207-nativebranches'
HP = 'hp-jtf637-headroutermoments-branches207-r2'
BLOCK = 1_000_000


def read(run: str, file: str) -> dict:
    """Read one small observation record."""
    return json.loads((ROOT / 'runs' / run / file).read_text())


def fp32(value: np.ndarray) -> np.ndarray:
    """Interpret preserved BF16 storage or native FP32 dump without arithmetic conversion."""
    if value.dtype == np.uint16:
        return (value.astype(np.uint32) << np.uint32(16)).view(np.float32)
    if value.dtype != np.float32:
        raise ValueError('Unexpected boundary storage dtype')
    return value


def canonical_sha(array: np.ndarray) -> str:
    """Hash logical BF16 bits while checking native dump values are exactly representable."""
    flat = array.reshape(-1)
    result = hashlib.sha256()
    for start in range(0, flat.size, BLOCK):
        values = flat[start:start + BLOCK]
        if values.dtype == np.float32:
            bits = values.view(np.uint32)
            if np.any(bits & np.uint32(0xffff)):
                raise ValueError('Native BF16 dump contains non-BF16 values')
            values = (bits >> np.uint32(16)).astype(np.uint16)
        elif values.dtype != np.uint16:
            raise ValueError('Expected preserved BF16 boundary')
        result.update(memoryview(np.ascontiguousarray(values)).cast('B'))
    return result.hexdigest()


def differences(left: np.ndarray, right: np.ndarray, path: Path | None = None) -> dict:
    """Measure every logical element and retain a bounded list of differing output coordinates."""
    if left.shape != right.shape:
        raise ValueError('Boundary shapes differ')
    left, right = left.reshape(-1), right.reshape(-1)
    count, maximum, error_sq, reference_sq = 0, 0.0, 0.0, 0.0
    selected = []
    for start in range(0, left.size, BLOCK):
        mf, hp = fp32(left[start:start + BLOCK]), fp32(right[start:start + BLOCK])
        diff = hp.astype(np.float64) - mf.astype(np.float64)
        indices = np.flatnonzero(mf != hp)
        count += indices.size
        if path is not None and count < 1_000_000:
            selected.append(indices + start)
        maximum = max(maximum, float(np.max(np.abs(diff))))
        error_sq += float(np.dot(diff, diff))
        reference = mf.astype(np.float64)
        reference_sq += float(np.dot(reference, reference))
    result = {'size': int(left.size), 'different': int(count), 'max_abs': maximum,
              'relative_l2': float(np.sqrt(error_sq / reference_sq)) if reference_sq else None}
    if path is not None and count < 1_000_000:
        indices = np.concatenate(selected).astype(np.int64)
        if indices.size != count:
            raise ValueError('Incomplete selected coordinate evidence')
        np.save(path, indices, allow_pickle=False)
        result['coordinate_file'] = str(path)
        result['coordinate_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def main() -> None:
    """Require unmodified trajectories before attributing differences to actual branch GEMMs."""
    checks = {'mf': fidelity(MF, 'mf-resume', 7),
              'hp': fidelity(HP, 'hp-jtf637-headroutermoments-gradient207', 7)}
    for family, run, reference in [('mf', MF, 'mf-head207-nativegradient'),
                                   ('hp', HP, 'hp-jtf637-headroutermoments-gradient207')]:
        checks[family]['exit_zero'] = (ROOT / 'runs' / (run + '.exit')).read_text().strip() == '0'
        checks[family]['moment207_hash_exact'] = read(run, 'HEAD_MOMENTS207_HASH.json') == read(
            reference, 'HEAD_MOMENTS207_HASH.json')
    checks['hp']['models206207_exact'] = model_hashes(HP) == model_hashes('hp-jtf637-headroutermoments-gradient207')
    checks['hp']['postclip_gradient_hash_exact'] = read(HP, 'HEAD_GRADIENT207.json')['sha256'] == read(
        'hp-jtf637-headroutermoments-gradient207', 'HEAD_GRADIENT207.json')['sha256']
    marker = read(HP, 'HEAD_BRANCH_OBSERVER.json')
    checks['hp']['observer_source'] = marker['source_sha256'] == hashlib.sha256(
        (ROOT / 'tools-f637/hp_head_branches207_probe_r2.py').read_bytes()).hexdigest()
    checks['hp']['original_source'] = marker['original_source_sha256'] == hashlib.sha256(
        (ROOT / 'tools-f637/backward_precision_builder.py').read_bytes()).hexdigest()
    manifest = read(HP, 'PROBE_SOURCE_MANIFEST.json')
    checks['hp']['49_sources'] = len(manifest['files']) == 49 and all(
        row['sha256'] == hashlib.sha256((ROOT / 'tools-f637' / row['name']).read_bytes()).hexdigest()
        for row in manifest['files'])
    records = [json.loads(line) for line in (ROOT / 'runs' / HP / 'HEAD_BRANCHES207.jsonl').read_text().splitlines()]
    native = {}
    directory = STATE / (MF + '-kernels/rank_0/HeadBranches207/1/6')
    for filename, original in csv.reader((directory / 'mapping.csv').read_text().splitlines()):
        if '.bool.npy' in original:
            continue
        number = 2 if 'MatMul-op2.' in original else 3 if 'MatMul-op3.' in original else None
        kind = next((kind for key, kind in [('input.0', 'dy'), ('input.1', 'x'), ('output.0', 'dw')]
                     if '.' + key + '.' in original), None)
        if number is None or kind is None:
            raise ValueError('Unexpected native branch kernel')
        native.setdefault(number, {})[kind] = {'path': str(directory / filename), 'original_name': original}
    observed, arrays = {}, {'mf': {}, 'hp': {}}
    for row in records:
        array = np.load(row['path'], mmap_mode='r')
        if hashlib.sha256(memoryview(array).cast('B')).hexdigest() != row['sha256']:
            raise ValueError('HP branch data changed')
        observed.setdefault(row['call'], {})[row['kind']] = row
    calls = [call for call, rows in observed.items() if set(rows) == {'x', 'dy', 'dw'}]
    checks['hp']['two_complete_backward_branches'] = len(calls) == 2
    for family, source in [('mf', native), ('hp', {call: observed[call] for call in calls})]:
        for call, values in source.items():
            arrays[family][call] = {}
            for kind, row in values.items():
                array = np.load(row['path'], mmap_mode='r')
                row['bf16_sha256'] = canonical_sha(array)
                row['shape'] = list(array.shape)
                row['storage_dtype'] = str(array.dtype)
                arrays[family][call][kind] = array
    pairs = []
    for number, rows in native.items():
        matches = [call for call in calls if rows['x']['bf16_sha256'] == observed[call]['x']['bf16_sha256']]
        if len(matches) != 1:
            raise ValueError('Cannot uniquely map native branch by complete input identity')
        pairs.append((number, matches[0]))
    checks['mf']['two_unique_branch_matches'] = len(pairs) == len({call for _, call in pairs}) == 2
    if not all(all(row.values()) for row in checks.values()):
        raise ValueError(checks)
    target = STATE / 'head-branches207-differences'
    target.mkdir(exist_ok=False)
    comparisons = []
    for number, call in pairs:
        measured = {kind: differences(arrays['mf'][number][kind], arrays['hp'][call][kind],
                                     target / ('op' + str(number) + '-coordinates.npy') if kind == 'dw' else None)
                    for kind in ['x', 'dy', 'dw']}
        comparisons.append({'native_kernel': number, 'hp_forward_call': call, 'comparison': measured})
    gradient_report = json.loads((ROOT / 'evidence/head-moments207-gradient-comparison.json').read_text())
    postclip = {'mf': gradient_report['native_kernels']['input.0'], 'hp': read(HP, 'HEAD_GRADIENT207.json')}
    reconstruction = {}
    for family, run in [('mf', MF), ('hp', HP)]:
        metrics = [json.loads(line) for line in (ROOT / 'runs' / run / 'metrics.jsonl').read_text().splitlines()]
        norm = next(row['grad_norm'] for row in metrics if row['step'] == 207)
        coefficient = np.minimum(np.float32(1), np.float32(1) / (np.float32(norm) + np.float32(1e-6)))
        branches = [arrays[family][key]['dw'].reshape(-1) for key in arrays[family]]
        result = hashlib.sha256()
        for start in range(0, branches[0].size, BLOCK):
            value = (fp32(branches[0][start:start + BLOCK]) + fp32(branches[1][start:start + BLOCK])) * coefficient
            result.update(memoryview(value).cast('B'))
        reconstruction[family] = {'grad_norm': norm, 'coefficient': float(coefficient),
                                  'reconstructed_sha256': result.hexdigest(),
                                  'actual_sha256': postclip[family]['sha256'],
                                  'all_postclip_elements_exact': result.hexdigest() == postclip[family]['sha256']}
    report = {'complete': True, 'fidelity': checks, 'mf_run': MF, 'hp_run': HP, 'comparisons': comparisons,
              'postclip_reconstruction': reconstruction, 'native': native, 'hp': observed,
              'hp_recomputation': read(HP, 'HEAD_BRANCH_CALLS.json'),
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'scope': 'Full actual BF16 operands and outputs, branches mapped by complete x identity. '
                       'Numerical differences do not alone prove mathematical incorrectness.'}
    (ROOT / 'evidence/head-moments207-branches-comparison.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'complete': True, 'fidelity': checks, 'comparisons': comparisons,
                      'postclip_reconstruction': reconstruction}))


if __name__ == '__main__':
    main()
