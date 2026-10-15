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

"""Replay MF8.5 standalone reductions and multiply-reduce graphs on actual212 inputs."""
import argparse
import hashlib
import json
from pathlib import Path

import mindspore as ms
from mindspore import nn, ops
import numpy as np

from compare_head_weight_boundaries import rounded_bf16
from compare_preclip212 import compare_arrays
from compare_router212_scores import load_scores
from compare_weighted212_operands import HP, MF, load

ROOT = Path('/alignment/phase12')


class ProductSum(nn.Cell):
    """Keep the same native BF16 multiplication and final-axis sum."""

    def construct(self, gradient, values):
        return ops.ReduceSum(keep_dims=True)(ops.mul(gradient, values), (-1,))


class SumOnly(nn.Cell):
    """Isolate the final-axis reduction from any producer fusion."""

    def construct(self, product):
        return ops.ReduceSum(keep_dims=True)(product, (-1,))


def main() -> None:
    """Compare all8192 outputs and save bounded profiler-friendly evidence."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['graph', 'eager'], required=True)
    args = parser.parse_args()
    ms.set_context(mode=ms.GRAPH_MODE if args.mode == 'graph' else ms.PYNATIVE_MODE,
                   device_target='Ascend', device_id=0, deterministic='ON', max_device_memory='24GB',
                   memory_optimize_level='O0', jit_config={'jit_level': 'O1'},
                   ascend_config={'parallel_speed_up_json_path':
                                  '/workspace/mindformers/configs/qwen3/parallel_speed_up.json'})
    audit = json.loads((ROOT / 'evidence/weighted212-operands-comparison.json').read_text())
    if not audit['same_inputs'] or not all(audit['fidelity'].values()):
        raise ValueError('Actual operand fidelity required')
    hp, _ = load(HP, 'hp')
    mf, _ = load_scores(MF, 'mf')
    inverse = np.argsort(hp['order'])
    padding = json.loads((ROOT / 'runs' / MF / 'WEIGHTED_PRODUCT-212.json').read_text())['padding']
    token_count = hp['outputs'].shape[0] // 2
    values, gradient = np.asarray(hp['outputs'])[inverse], np.asarray(hp['dy'])[inverse]
    product = rounded_bf16(rounded_bf16(gradient) * values)
    product_hash = hashlib.sha256(product.tobytes()).hexdigest()
    records = []
    dump = Path('/state/jt-recipe-20261014/weighted212-mf-fp32sum-r3')
    dump.mkdir(parents=True, exist_ok=False)
    for shape_name, shape in [('2d', (token_count * 2, 5120)), ('4d', (1, token_count, 2, 5120))]:
        for route in ['sum_only_bf16', 'sum_only_fp32', 'multiply_sum_bf16']:
            if route == 'multiply_sum_bf16':
                result = ProductSum()(ms.Tensor(gradient.reshape(shape), ms.bfloat16),
                                      ms.Tensor(values.reshape(shape), ms.bfloat16))
            else:
                dtype = ms.bfloat16 if route == 'sum_only_bf16' else ms.float32
                summed = SumOnly()(ms.Tensor(product.reshape(shape), dtype))
                if route == 'sum_only_fp32':
                    np.save(dump / (shape_name + '-fp32sum.npy'), summed.asnumpy(), allow_pickle=False)
                result = summed.astype(ms.bfloat16)
            actual = result.astype(ms.float32).asnumpy().reshape(token_count, 2)[padding:]
            row = {'shape': shape_name, 'route': route,
                   'hp': compare_arrays(actual, hp['weights_dy'][inverse].reshape(token_count, 2)[padding:]),
                   'mf': compare_arrays(actual, mf['coefficients_dy']),
                   'sha256': hashlib.sha256(actual.tobytes()).hexdigest()}
            records.append(row)
            print(json.dumps(row), flush=True)
    report = {'complete': True, 'mode': args.mode, 'mindspore': ms.__version__, 'records': records,
              'product_sha256': product_hash,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'loaded_libraries': sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                                          if any(name in line for name in ['libopapi.so', 'libascendcl.so'])}),
              'scope': 'Actual212 identical operands; standalone operator replay only, not training acceptance.'}
    (ROOT / 'evidence' / ('weighted212-mf-reduction-' + args.mode + '-r3.json')).write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
