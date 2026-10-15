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

"""Compare documented DVM-style lane trees against the complete native FP32 sum."""
import hashlib
import json
from pathlib import Path

import numpy as np

from compare_head_weight_boundaries import rounded_bf16
from compare_preclip212 import compare_arrays
from compare_weighted212_operands import HP, load
from replay_ce_sumx_trees import reduce_tree

ROOT = Path('/alignment/phase12')
STATE = Path('/state/jt-recipe-20261014')


def main() -> None:
    """Use complete intermediate outputs, rather than matching a selected BF16 coordinate."""
    run = 'weighted212-mf-reduction-graph-r3'
    if (ROOT / 'runs' / (run + '.exit')).read_text().strip() != '0':
        raise ValueError('Native FP32 observation has not completed')
    native = json.loads((ROOT / 'evidence' / (run + '.json')).read_text())
    hp, _ = load(HP, 'hp')
    inverse = np.argsort(hp['order'])
    product = rounded_bf16(rounded_bf16(hp['dy'][inverse]) * hp['outputs'][inverse])
    if hashlib.sha256(product.tobytes()).hexdigest() != native['product_sha256']:
        raise ValueError('Native input does not match actual weighted product')
    expected = np.load(STATE / 'weighted212-mf-fp32sum-r3/2d-fp32sum.npy').reshape(-1)
    second = np.load(STATE / 'weighted212-mf-fp32sum-r3/4d-fp32sum.npy').reshape(-1)
    if not np.array_equal(expected, second):
        raise ValueError('Native shape variants differ')
    records = []
    for width in [16, 32, 64, 128]:
        outputs = []
        for start in range(0, product.shape[0], 256):
            values = product[start:start + 256].reshape(-1, 5120 // width, width).transpose(0, 2, 1)
            lanes = reduce_tree(values, 'halves')
            outputs.append(reduce_tree(lanes, 'adjacent'))
        result = np.concatenate(outputs)
        row = {'width': width, 'vertical': 'zero-padded halves', 'horizontal': 'adjacent',
               'fp32': compare_arrays(result, expected),
               'bf16': compare_arrays(rounded_bf16(result), rounded_bf16(expected)),
               'sha256': hashlib.sha256(result.tobytes()).hexdigest()}
        records.append(row)
        print(json.dumps(row), flush=True)
    report = {'complete': True, 'same_actual_input': True, 'native_shape_variants_exact': True,
              'native_fp32_sha256': hashlib.sha256(expected.tobytes()).hexdigest(),
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'records': records,
              'scope': 'All8208 native FP32 sums including16 padding choices, actual BF16 products.'}
    (ROOT / 'evidence/weighted212-explicit-trees.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
