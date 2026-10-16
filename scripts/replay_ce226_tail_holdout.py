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

"""Validate candidate tail joins on fresh synthetic CE rows, independent of training values."""
import hashlib
import json
from pathlib import Path

import numpy as np
import mindspore as ms
from mindspore import nn, ops
from mindspore.ops.auto_generate import ArgMaxWithValue, SubExt, Exp, SumExt, Log, OneHotExt, Div

from replay_ce226_reduction import tree_parts, sequential
from replay_ce_sumx_ragged import reduce_tree

ROOT = Path('/alignment/phase12')


class HoldoutCE(nn.Cell):
    """Retain the complete validated CE graph and expose exact exponent inputs."""

    def construct(self, values, labels, mask):
        _, maximum = ArgMaxWithValue(axis=1, keep_dims=True)(values)
        shifted = SubExt()(values, maximum)
        exponent = Exp()(shifted)
        total = SumExt()(exponent, -1, True)
        logarithm = Log()(total)
        log_probability = SubExt()(shifted, logarithm)
        hot = OneHotExt(axis=-1)(labels, 152064, ms.Tensor(1, ms.int32), ms.Tensor(0, ms.int32))
        denominator = SumExt()(mask) + ms.Tensor(1e-8, ms.float32)
        multiplier = Div()(ms.Tensor(1.0, ms.float32), denominator) * mask
        gradient = (Exp()(log_probability) - hot) * ops.expand_dims(multiplier, -1)
        return gradient, total, logarithm, log_probability, maximum, exponent


def main() -> None:
    """Test128 new rows twice each, in ordinary and final row positions."""
    ms.set_context(mode=ms.GRAPH_MODE, device_target='Ascend', device_id=0, deterministic='ON',
                   jit_config={'jit_level': 'O1'})
    model = HoldoutCE()
    labels = ms.Tensor(np.zeros(4096, np.int32))
    mask = ms.Tensor(np.ones(4096, np.float32))
    selected = np.array(list(range(16)) + list(range(4080, 4096)), np.int32)
    indices = ms.Tensor(selected)
    rng = np.random.default_rng(226704)
    full = np.zeros((4096, 152064), np.float32)
    records = []
    for case in range(8):
        deviation = [1., 2., 4., 8., 12., 16., 24., 32.][case]
        samples = rng.standard_normal((16, 152064), dtype=np.float32) * np.float32(deviation)
        full[:16], full[-16:] = samples, samples
        result = model(ms.Tensor(full), labels, mask)
        total = result[1].asnumpy()[selected, 0]
        exponent = ops.gather(result[-1], indices, 0).asnumpy()
        parts = tree_parts(exponent)
        candidates = {'sequential': sequential(parts).reshape(-1)}
        for split in [4, 5, 7, 11]:
            candidates['split-' + str(split)] = reduce_tree(parts[:, :split], 'sequential') + reduce_tree(parts[:, split:], 'sequential')
        candidates['even-odd'] = reduce_tree(parts[:, 0::2], 'sequential') + reduce_tree(parts[:, 1::2], 'sequential')
        checks = {mode: {'ordinary_different': int(np.count_nonzero(values[:16] != total[:16])),
                         'tail_different': int(np.count_nonzero(values[16:] != total[16:])),
                         'tail_max_abs': float(np.max(np.abs(values[16:].astype(np.float64) - total[16:])))}
                  for mode, values in candidates.items()}
        row = {'case': case, 'logit_stddev': deviation, 'input_sha256': hashlib.sha256(samples.tobytes()).hexdigest(),
               'repeated_exponents_exact': bool(np.array_equal(exponent[:16], exponent[16:])), 'candidates': checks}
        records.append(row)
        print(json.dumps(row), flush=True)
        del result, exponent, parts
    report = {'complete': True, 'seed': 226704, 'unique_new_rows': 128, 'training_values_used': False,
              'records': records, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'scope': 'Synthetic operand validation at original full shape; no training acceptance claim.'}
    (ROOT / 'evidence/ce226-tail-holdout.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
