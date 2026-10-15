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

"""Verify the independent NPU64-lane tree against every native FP32 output."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch_npu

from compare_preclip212 import compare_arrays
from compare_weighted212_operands import HP, load
from hp_weighted_dvm_factor import sum_fp32
from hyper_parallel.trainer.runtime.random import enable_full_determinism

ROOT = Path('/alignment/phase12')
STATE = Path('/state/jt-recipe-20261014')


def main() -> None:
    torch.npu.set_device(0)
    enable_full_determinism(42, close_matmul_k_shift=False)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    hp, _ = load(HP, 'hp')
    inverse = np.argsort(hp['order'])
    outputs = torch.from_numpy(np.array(hp['outputs'])).npu().bfloat16()
    gradient = torch.from_numpy(np.array(hp['dy'])).npu()
    product = gradient.to(outputs.dtype) * outputs
    actual = sum_fp32(product).float().cpu().numpy()[inverse].reshape(-1)
    expected = np.load(STATE / 'weighted212-mf-fp32sum-r3/2d-fp32sum.npy').reshape(-1)
    comparison = compare_arrays(actual, expected)
    report = {'complete': True, 'native_fp32': comparison, 'teacher_values_used': False,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'factor_sha256': hashlib.sha256((ROOT / 'tools-f637/hp_weighted_dvm_factor.py').read_bytes()).hexdigest(),
              'same_actual_input': True, 'output_sha256': hashlib.sha256(actual.tobytes()).hexdigest()}
    (ROOT / 'evidence/weighted212-npu-tree-verification.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    if comparison['fp32_bitwise_different']:
        raise ValueError('NPU tree differs from actual MF FP32 sum')


if __name__ == '__main__':
    main()
