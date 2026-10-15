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

"""Profile actual head207 routes while requiring their complete prior output hashes."""
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch_npu

from hyper_parallel.trainer.runtime.random import enable_full_determinism
from screen_head207_gemm_routes import bits, matmul, sha

ROOT = Path('/alignment/phase12')


@torch.no_grad()
def main() -> None:
    """Run three distinguishable operator routes on the exact op2 BF16 device operands."""
    native = json.loads((ROOT / 'evidence/head-moments207-branches-comparison.json').read_text())['native']['2']
    expected = json.loads((ROOT / 'evidence/head207-gemm-screen-op2.json').read_text())
    modes = {row['mode']: row for row in expected['results']}
    torch.npu.set_device(0)
    enable_full_determinism(42, close_matmul_k_shift=False)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    torch.npu.config.allow_internal_format = False
    torch.npu.matmul.allow_hf32 = False
    tensors = {}
    inputs = {}
    for name in ['x', 'dy']:
        array = np.load(native[name]['path'], mmap_mode='r')
        tensors[name] = torch.from_numpy(np.array(array, copy=True)).to(device='npu', dtype=torch.bfloat16)
        inputs[name] = sha(bits(tensors[name]))
        if inputs[name] != native[name]['bf16_sha256']:
            raise ValueError('Actual operand differs from native reference')
        del array
    records = []
    for mode in ['nd_transpose_view', 'single_batch_bmm', 'native_bmmv2']:
        torch.npu.synchronize()
        start = time.time_ns()
        output, formats = matmul(tensors['dy'], tensors['x'], mode)
        torch.npu.synchronize()
        end = time.time_ns()
        digest = sha(bits(output))
        if digest != modes[mode]['bf16_sha256']:
            raise ValueError('Profiled route does not reproduce previous complete output')
        row = {'mode': mode, 'formats': formats, 'bf16_sha256': digest,
               'matches_prior_screen': True, 'start_wall_ns': start, 'end_wall_ns': end}
        records.append(row)
        print(json.dumps(row), flush=True)
        del output
        gc.collect()
        torch.npu.empty_cache()
    libraries = sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                        if any(name in line for name in ['libopapi.so', 'libascendcl.so', 'libnnopbase.so'])})
    report = {'complete': True, 'kernel': 2, 'input_hashes': inputs, 'records': records,
              'torch': torch.__version__, 'torch_npu': torch_npu.__version__, 'loaded_libraries': libraries,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'helper_sha256': hashlib.sha256((Path(__file__).parent / 'screen_head207_gemm_routes.py').read_bytes()).hexdigest(),
              'scope': 'Profiler observation only; no new training result or throughput claim.'}
    (ROOT / 'evidence/head207-gemm-hp-profile.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
