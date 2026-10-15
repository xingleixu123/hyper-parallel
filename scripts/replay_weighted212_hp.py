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

"""Replay native and explicit legacy reduction paths on the verified actual212 product."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch_npu

from compare_preclip212 import compare_arrays
from compare_router212_scores import load_scores
from compare_weighted212_operands import HP, MF, load
from hyper_parallel.trainer.runtime.random import enable_full_determinism
from weighted_sum_aclop import reduce_last

ROOT = Path('/alignment/phase12')


def main() -> None:
    """Reproduce the original full result before screening independent arithmetic routes."""
    audit = json.loads((ROOT / 'evidence/weighted212-operands-comparison.json').read_text())
    if not audit['complete'] or not audit['same_inputs'] or not all(audit['fidelity'].values()):
        raise ValueError('Actual operand fidelity is required')
    torch.npu.set_device(0)
    enable_full_determinism(42, close_matmul_k_shift=False)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    hp, _ = load(HP, 'hp')
    mf, _ = load_scores(MF, 'mf')
    inverse = np.argsort(hp['order'])
    outputs = torch.from_numpy(np.array(hp['outputs'])).npu().bfloat16()
    gradient = torch.from_numpy(np.array(hp['dy'])).npu()
    product = gradient.to(outputs.dtype) * outputs
    baseline = product.sum(-1, keepdim=True).float().cpu().numpy()
    if not np.array_equal(baseline, hp['weights_dy']):
        raise ValueError('Original HP weighted derivative does not reproduce')
    records = []
    for order_name in ['expert_major', 'token_major']:
        values = product if order_name == 'expert_major' else product[torch.from_numpy(inverse).npu()]
        for shape_name in ['2d', '4d']:
            shaped = values if shape_name == '2d' else values.reshape(1, 4096, 2, 5120)
            for route in ['eager_bf16', 'eager_fp32_sum', 'jit_bf16', 'aclop_bf16', 'aclop_fp32']:
                try:
                    torch_npu.npu.set_compile_mode(jit_compile=route.startswith(('jit', 'aclop')))
                    if route == 'eager_fp32_sum':
                        result = shaped.sum(-1, keepdim=True, dtype=torch.float32).bfloat16()
                    elif route == 'aclop_bf16':
                        result = reduce_last(shaped)
                    elif route == 'aclop_fp32':
                        result = reduce_last(shaped.float()).bfloat16()
                    else:
                        result = shaped.sum(-1, keepdim=True)
                    actual = result.float().cpu().numpy().reshape(-1, 1)
                    if order_name == 'expert_major':
                        actual = actual[inverse]
                    actual = actual.reshape(4096, 2)
                    row = {'order': order_name, 'shape': shape_name, 'route': route,
                           'hp': compare_arrays(actual, hp['weights_dy'][inverse].reshape(4096, 2)),
                           'mf': compare_arrays(actual, mf['coefficients_dy']),
                           'sha256': hashlib.sha256(actual.tobytes()).hexdigest()}
                except (RuntimeError, ValueError) as error:
                    row = {'order': order_name, 'shape': shape_name, 'route': route, 'error': str(error)[:1500]}
                finally:
                    torch_npu.npu.set_compile_mode(jit_compile=False)
                records.append(row)
                print(json.dumps(row), flush=True)
    report = {'complete': True, 'baseline_full_hash_exact': True, 'records': records,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'helper_sha256': hashlib.sha256((Path(__file__).parent / 'weighted_sum_aclop.py').read_bytes()).hexdigest(),
              'scope': 'Same actual BF16 products; no teacher output supplied to any reduction.'}
    (ROOT / 'evidence/weighted212-hp-reduction-replay.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
