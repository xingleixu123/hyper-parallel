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

"""Verify weighted-product operands, native probability derivatives and high precision sums."""
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from compare_head_weight_boundaries import rounded_bf16
from compare_preclip212 import compare_arrays
from compare_router212_scores import branch_hashes, load_scores
from compare_vocab_optimizer_trace import fidelity

ROOT = Path('/alignment/phase12')
STATE = Path('/state/jt-recipe-20261014')
HP, MF = 'hp-jtf637-norm208-weighted212', 'mf-weighted212'


def load(run: str, side: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load complete actual operands after verifying hashes and shapes."""
    report = json.loads((ROOT / 'runs' / run / 'WEIGHTED_PRODUCT-212.json').read_text())
    directory = STATE / run / ('weighted' if side == 'hp' else 'deep')
    arrays = {}
    for row in report['records']:
        value = np.load(directory / row['file'], mmap_mode='r')
        if (hashlib.sha256(value.tobytes()).hexdigest() != row['sha256']
                or list(value.shape) != row['shape'] or str(value.dtype) != row['dtype']):
            raise ValueError('Changed weighted-product operand')
        if row['name'] in arrays and not np.array_equal(arrays[row['name']], value):
            raise ValueError('Recomputation changed an operand')
        arrays[row['name']] = value
    return arrays, report


def main() -> None:
    """Require observer fidelity before deriving precision comparisons from actual inputs."""
    checks = {}
    for side, run, control in [('hp', HP, 'hp-jtf637-norm208-routerscores212'), ('mf', MF, 'mf-routerscores212')]:
        checks[side + '_exit_zero'] = (ROOT / 'runs' / (run + '.exit')).read_text().strip() == '0'
        checks.update({side + '_' + key: value for key, value in fidelity(run, control, 12).items()})
        checks[side + '_all_main_branches_exact'] = branch_hashes(run, side) == branch_hashes(control, side)
        actual, _ = load_scores(run, side)
        previous, _ = load_scores(control, side)
        checks[side + '_all_score_boundaries_exact'] = actual.keys() == previous.keys() and all(
            np.array_equal(actual[name], previous[name]) for name in actual)
    hp, hp_report = load(HP, 'hp')
    mf, mf_report = load(MF, 'mf')
    for side, report, folder, name in [('hp', hp_report, 'tools-f637', 'hp_weighted212_observer.py'),
                                      ('mf', mf_report, 'tools', 'mf_weighted212_tap.py')]:
        checks[side + '_weighted_source'] = report['source_sha256'] == hashlib.sha256(
            (ROOT / folder / name).read_bytes()).hexdigest()
    checks['hp_round_weights'] = hp_report['round_weights']
    checks['mf_no_fused_permute'] = not mf_report['moe_permute_fusion']
    checks['hp_full_permutation'] = np.array_equal(np.sort(hp['order']), np.arange(hp['order'].size))
    if not all(checks.values()):
        raise ValueError(checks)
    inverse = np.argsort(hp['order'])
    hp_values = hp['outputs'][inverse].reshape(-1, 2, 5120)
    hp_dy = hp['dy'][inverse].reshape(-1, 2, 5120)
    hp_weights_dy = hp['weights_dy'][inverse].reshape(-1, 2)
    hp_weights = hp['weights'][inverse].reshape(-1, 2)
    mf_inverse = mf['unsort'].reshape(-1)[mf['outer_unsort'].reshape(-1)]
    mf_values = mf['tokens'].reshape(-1, 5120)[mf_inverse].reshape(-1, 2, 5120)
    padding = mf_report['padding']
    if padding:
        hp_values, hp_dy = hp_values[padding:], hp_dy[padding:]
        hp_weights, hp_weights_dy = hp_weights[padding:], hp_weights_dy[padding:]
        mf_values = mf_values[padding:]
    mf_dy = np.broadcast_to(mf['dy'].reshape(-1, 1, 5120), mf_values.shape)
    hp_scores, _ = load_scores(HP, 'hp')
    mf_scores, _ = load_scores(MF, 'mf')
    comparisons = {
        'expert_values': compare_arrays(hp_values, mf_values),
        'weighted_output_dy': compare_arrays(hp_dy, mf_dy),
        'weights': compare_arrays(hp_weights, mf['probs'].reshape(-1, 2)[padding:]),
        'hp_native_weight_dy_to_coefficients': compare_arrays(hp_weights_dy, hp_scores['coefficients_dy']),
        'native_weight_dy': compare_arrays(hp_weights_dy, mf_scores['coefficients_dy'])}
    same_inputs = all(comparisons[name]['fp32_bitwise_different'] == 0 for name in ['expert_values', 'weighted_output_dy'])
    if not same_inputs:
        raise ValueError('Weighted reduction operands differ; do not attribute to arithmetic')
    product = rounded_bf16(rounded_bf16(hp_dy) * hp_values)
    exact_sum = np.sum(product, axis=-1, dtype=np.float64)
    reference = rounded_bf16(exact_sum.astype(np.float32))
    hp_error = np.abs(hp_weights_dy.astype(np.float64) - exact_sum)
    mf_error = np.abs(mf_scores['coefficients_dy'].astype(np.float64) - exact_sum)
    changed = hp_weights_dy.view(np.uint32) != mf_scores['coefficients_dy'].view(np.uint32)
    candidates = {
        'fp64_sum_rounded_bf16': {'hp': compare_arrays(hp_weights_dy, reference),
                                  'mf': compare_arrays(mf_scores['coefficients_dy'], reference)},
        'numpy_fp32_sum_rounded_bf16': {
            'hp': compare_arrays(hp_weights_dy, rounded_bf16(np.sum(product, axis=-1, dtype=np.float32))),
            'mf': compare_arrays(mf_scores['coefficients_dy'], rounded_bf16(np.sum(product, axis=-1, dtype=np.float32)))}}
    correctness = {'scope': 'FP64 sum of actual rounded BF16 products; does not remove required intermediate rounding.',
                   'different_coordinates': int(np.count_nonzero(changed)),
                   'hp_closer_at_differences': int(np.count_nonzero(changed & (hp_error < mf_error))),
                   'mf_closer_at_differences': int(np.count_nonzero(changed & (mf_error < hp_error))),
                   'ties_at_differences': int(np.count_nonzero(changed & (mf_error == hp_error))),
                   'hp_max_abs_error': float(hp_error.max()), 'mf_max_abs_error': float(mf_error.max())}
    report = {'complete': True, 'fidelity': checks, 'comparisons': comparisons, 'candidates': candidates,
              'correctness': correctness, 'same_inputs': same_inputs,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'scope': 'Complete8192 expert choices, each5120 hidden elements, native12-step observer fidelity.'}
    (ROOT / 'evidence/weighted212-operands-comparison.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == '__main__':
    main()
