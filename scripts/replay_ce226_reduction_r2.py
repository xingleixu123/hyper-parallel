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

"""Separate MF full-shape exp/sum fusion from partial-tile and NumPy tree replays."""
import hashlib
import json
from pathlib import Path

import numpy as np
import mindspore as ms
from mindspore import nn, ops
from mindspore.ops.auto_generate import SumExt, SubExt, Exp

from replay_ce_sumx_ragged import reduce_tree
from compare_ce226 import load_arrays
from replay_cross_entropy_mf_o1_stages import CEStages

ROOT = Path('/alignment/phase12')
STATE = Path('/state/jt-recipe-20261014')


class SumOnly(nn.Cell):
    """Recompute native DVM sum from already observed exact FP32 exponents."""

    def construct(self, values):
        return SumExt()(values, -1, True)


class ShiftExpSum(nn.Cell):
    """Preserve max/sub/exp/sum graph for distinguishing a fused implementation."""

    def construct(self, logits):
        _, maximum = ops.ArgMaxWithValue(axis=1, keep_dims=True)(logits)
        shifted = SubExt()(logits, maximum)
        exponent = Exp()(shifted)
        return SumExt()(exponent, -1, True)


def measure(actual: np.ndarray, expected: np.ndarray) -> dict:
    """Compare bounded full selected reductions without publishing token values."""
    delta = actual.astype(np.float64) - expected.astype(np.float64)
    return {'different': int(np.count_nonzero(actual != expected)), 'max_abs': float(np.abs(delta).max())}


def tree_parts(exponent: np.ndarray) -> np.ndarray:
    """Compute the current explicit FP32 compatibility tree on CPU for diagnosis."""
    shaped = exponent.reshape(-1, 22, 108, 64)
    vertical = reduce_tree(shaped.transpose(0, 1, 3, 2), 'ragged-first')
    return reduce_tree(vertical, 'adjacent')


def sequential(parts: np.ndarray) -> np.ndarray:
    """Preserve ascending tile order in FP32."""
    result = parts[:, 0].copy()
    for index in range(1, parts.shape[1]):
        result = result + parts[:, index]
    return result[:, None]


def main() -> None:
    """Use full training shapes and original row locations; unspecified rows are controls."""
    audit = json.loads((ROOT / 'evidence/ce226-comparison.json').read_text())
    if not audit['complete']:
        raise ValueError('Native CE observation fidelity has not passed')
    hp = load_arrays(json.loads((ROOT / 'runs/hp-jtf637-qsqrt222-ce226/CE226_OPERANDS.json').read_text())['records'],
                     STATE / 'hp-jtf637-qsqrt222-ce226/ce', 'call')
    selection = json.loads((ROOT / 'evidence/head226-selected-coordinates.json').read_text())['token_rows']
    mf_rows = [json.loads(line) for line in (ROOT / 'runs/mf-ce226/CE226_OPERANDS.jsonl').read_text().splitlines()]
    mf = load_arrays(mf_rows[-1]['records'], STATE / 'mf-ce226/deep', 'name')
    output = STATE / 'ce226-reduction-replay-r2'
    output.mkdir(exist_ok=False)
    ms.set_context(mode=ms.GRAPH_MODE, device_target='Ascend', device_id=0, deterministic='ON',
                   jit_config={'jit_level': 'O1'})
    records = []
    for name, mapping in audit['mapping'].items():
        call = mapping['ce_call']
        expected = mf[name, 'sum']
        full = np.zeros((4096, 152064), np.float32)
        full[selection] = hp[call, 'exponent']
        tensor = ms.Tensor(full)
        direct = SumOnly()(tensor).asnumpy()[selection]
        parts = SumOnly()(ops.reshape(tensor, (4096, 22, 6912))).asnumpy()[selection, :, 0]
        cpu_parts = tree_parts(hp[call, 'exponent'])
        np.save(output / f'call{call}-native-parts.npy', parts)
        np.save(output / f'call{call}-cpu-parts.npy', cpu_parts)
        full[selection] = hp[call, 'logits']
        fused = ShiftExpSum()(ms.Tensor(full)).asnumpy()[selection]
        ce = CEStages(0.3 if call == 1 else 1.0)
        labels = ms.Tensor(mf[name, 'targets'].reshape(-1).astype(np.int32))
        mask = ms.Tensor(mf[name, 'mask'].reshape(-1))
        gradient, ce_sum, logarithm, log_probability, maximum = ce(ms.Tensor(full), labels, mask)
        native_sum = ce_sum.asnumpy()[selection]
        native_dx = gradient.asnumpy()[selection]
        np.save(output / f'call{call}-ce-sum.npy', native_sum)
        row = {'name': name, 'ce_sum_vs_training': measure(native_sum, expected),
               'ce_fp32_dx_vs_training': measure(native_dx, mf[name, 'dx_fp32']), 'direct_sum_vs_training': measure(direct, expected),
               'fused_sum_vs_training': measure(fused, expected),
               'native_partial_recombine_vs_training': measure(sequential(parts), expected),
               'cpu_partial_vs_native': measure(cpu_parts, parts),
               'cpu_tree_vs_hp': measure(sequential(cpu_parts), hp[call, 'sum']),
               'cpu_tree_vs_mf': measure(sequential(cpu_parts), expected)}
        records.append(row)
        print(json.dumps(row), flush=True)
        del tensor, full, gradient, ce_sum, logarithm, log_probability, maximum
    report = {'complete': True, 'records': records,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'scope': 'Actual4 selected rows in original4096 shapes; diagnostic isolation, not a training result.'}
    (ROOT / 'evidence/ce226-reduction-replay-r2.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
