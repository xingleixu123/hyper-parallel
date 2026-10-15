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

"""Audit native preclip observations, then rank complete-array differences."""
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from compare_head206_interventions import source_sha
from compare_vocab_optimizer_trace import fidelity, read_rows
from frontier212_array_io import describe

ROOT = Path('/alignment/phase12')
STATE = Path('/state/jt-recipe-20261014')
HP = 'hp-jtf637-weighted-dvm-preclip217'
MF = 'mf-preclip217'
BASE = 'hp-jtf637-weighted-dvm-main226'


def bf16_bits(value: np.ndarray) -> np.ndarray:
    """Return finite FP32 values rounded to BF16 with ties to even."""
    bits = value.view(np.uint32)
    return ((bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def compare_arrays(hp: np.ndarray, mf: np.ndarray) -> dict[str, Any]:
    """Measure all values in bounded chunks; retain no raw coordinates in reports."""
    if hp.shape != mf.shape or hp.dtype != np.float32 or mf.dtype != np.float32:
        raise ValueError('Array layout or dtype differs')
    first, second = hp.reshape(-1), mf.reshape(-1)
    different, rounded_different, maximum = 0, 0, 0.
    squares, absolutes, reference_squares = [], [], []
    for start in range(0, first.size, 1_048_576):
        left, right = first[start:start + 1_048_576], second[start:start + 1_048_576]
        if not np.isfinite(left).all() or not np.isfinite(right).all():
            raise ValueError('Nonfinite actual preclip gradients')
        different += int(np.count_nonzero(left.view(np.uint32) != right.view(np.uint32)))
        rounded_different += int(np.count_nonzero(bf16_bits(left) != bf16_bits(right)))
        difference = left.astype(np.float64) - right.astype(np.float64)
        squares.append(float(np.sum(difference * difference)))
        absolutes.append(float(np.sum(np.abs(difference))))
        reference_squares.append(float(np.sum(np.square(right, dtype=np.float64))))
        maximum = max(maximum, float(np.max(np.abs(difference))))
    square_error = math.fsum(squares)
    reference = math.fsum(reference_squares)
    return {'elements': int(first.size), 'fp32_bitwise_different': different,
            'bf16_different': rounded_different, 'max_abs': maximum,
            'mean_abs': math.fsum(absolutes) / first.size, 'l2_error_squared': square_error,
            'relative_l2': math.sqrt(square_error / reference) if reference else None}


def main() -> None:
    """Reject changed trajectories before using any numerical comparison."""
    checks = {'hp': fidelity(HP, BASE, 17), 'mf': fidelity(MF, 'mf-resume', 17)}
    for side, run in [('hp', HP), ('mf', MF)]:
        checks[side]['exit_zero'] = (ROOT / 'runs' / (run + '.exit')).read_text().strip() == '0'
    checks['hp']['config_only_observation_length_path'] = (ROOT / 'configs' / (BASE + '.yaml')).read_text().replace(
        BASE, HP).replace('train_iters: 226', 'train_iters: 217') == (ROOT / 'configs' / (HP + '.yaml')).read_text()
    checks['mf']['config_only_observation_length_path'] = (ROOT / 'configs/mf-preclip212.yaml').read_text().replace(
        'mf-preclip212', MF).replace('stop_step: 12', 'stop_step: 17') == (ROOT / 'configs' / (MF + '.yaml')).read_text()
    directory, baseline = ROOT / 'runs' / HP, ROOT / 'runs' / BASE
    for name in ['SOURCE_REVISION.json', 'PROBE_SOURCE_MANIFEST.json', 'HEAD_GEMM_ROUTE.json',
                 'SHARED_UP_FP32_FACTOR.json', 'ADAMW_ARCHIVE_ARITHMETIC.json',
                 'Q_NS_ADJACENT_FACTOR.json', 'ROUTER_NS_NPU_SQRT_FACTOR.json', 'WEIGHTED_DVM_FACTOR.json']:
        checks['hp']['unchanged_' + name] = (directory / name).read_bytes() == (baseline / name).read_bytes()
    for name in ['HEAD_GEMM_ROUTE_CALLS.jsonl', 'Q_NS_ADJACENT_CALLS.jsonl',
                 'ROUTER_NS_NPU_SQRT_CALLS.jsonl', 'CLIP_REDUCTION.jsonl', 'WEIGHTED_DVM_CALLS.jsonl']:
        checks['hp']['exact17_' + name] = read_rows(directory / name) == read_rows(baseline / name)[:17]
    manifest = json.loads((directory / 'PROBE_SOURCE_MANIFEST.json').read_text())
    checks['hp']['49_frozen_sources'] = len(manifest['files']) == 49 and all(
        row['sha256'] == source_sha(row['name']) for row in manifest['files'])
    records = {}
    for side, run, source in [('hp', HP, 'hp_preclip217_probe.py'), ('mf', MF, 'mf_preclip217_tap.py')]:
        for step in [216, 217]:
            report = json.loads((ROOT / 'runs' / run / ('PRECLIP-' + str(step) + '.json')).read_text())
            folder = 'tools-f637' if side == 'hp' else 'tools'
            checks[side][f'source_{step}'] = report['source_sha256'] == hashlib.sha256(
                (ROOT / folder / source).read_bytes()).hexdigest()
            checks[side][f'helper_{step}'] = report['helper_sha256'] == hashlib.sha256(
                (ROOT / folder / 'frontier212_array_io.py').read_bytes()).hexdigest()
            records[side, step] = {row['name']: row for row in report['records']}
            checks[side][f'all46_{step}'] = len(records[side, step]) == 46
    output = {'complete': True, 'fidelity': checks, 'usable_as_native_training': all(
        all(check.values()) for check in checks.values()), 'runs': {'hp': HP, 'mf': MF}}
    destination = ROOT / 'evidence/preclip217-comparison.json'
    if not output['usable_as_native_training']:
        destination.write_text(json.dumps(output, indent=2))
        raise ValueError(checks)
    hashes = {step: [{'name': name, 'exact': row['sha256'] == records['mf', step][name]['sha256']}
                    for name, row in records['hp', step].items()] for step in [216, 217]}
    comparisons = []
    for name, hp_row in records['hp', 217].items():
        mf_row = records['mf', 217][name]
        hp = np.load(STATE / HP / hp_row['file'], mmap_mode='r')
        mf = np.load(STATE / MF / 'gradients' / mf_row['file'], mmap_mode='r')
        for value, row in [(hp, hp_row), (mf, mf_row)]:
            metadata = describe(value)
            if any(metadata[key] != row[key] for key in metadata):
                raise ValueError('Retained actual preclip array differs from its report')
        comparisons.append({'name': name, **compare_arrays(hp, mf),
                            'hp_square_sum_fp64': hp_row['square_sum_fp64'],
                            'mf_square_sum_fp64': mf_row['square_sum_fp64']})
        del hp, mf
    output.update(hash_comparison=hashes, comparisons=comparisons,
                  norms_fp64={side: math.sqrt(math.fsum(row['square_sum_fp64']
                                                      for row in records[side, 217].values()))
                              for side in ['hp', 'mf']},
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scope='Complete46 preclip gradients in native MF layouts; no optimizer or future values replaced.')
    destination.write_text(json.dumps(output, indent=2))
    print(json.dumps({'fidelity': output['usable_as_native_training'],
                      'exact_counts': {step: sum(row['exact'] for row in rows) for step, rows in hashes.items()},
                      'norms_fp64': output['norms_fp64'], 'different': [
                          row for row in comparisons if row['fp32_bitwise_different']]}))


if __name__ == '__main__':
    main()
