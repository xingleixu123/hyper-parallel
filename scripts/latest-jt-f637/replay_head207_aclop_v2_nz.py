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

"""Verify an explicit CANN9.2 legacy MatMulV2 against actual207 full outputs."""
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch_npu

from hyper_parallel.trainer.runtime.random import enable_full_determinism
from head_gemm_aclop_nz import matmul as nz_matmul
from screen_head207_gemm_routes import bits, sha

ROOT = Path('/alignment/phase12')


LAST_METADATA = []


def matmul(gradient, inputs):
    """Limit NZ permission and legacy compilation to this independently profiled product."""
    torch_npu.npu.set_compile_mode(jit_compile=True)
    torch.npu.config.allow_internal_format = True
    try:
        output, metadata = nz_matmul(gradient, inputs)
        LAST_METADATA[:] = metadata
        return output
    finally:
        torch_npu.npu.set_compile_mode(jit_compile=False)
        torch.npu.config.allow_internal_format = False


def compare(actual, reference):
    """Compare every output coordinate with the retained native BF16-representable dump."""
    different, maximum, squared, reference_squared = 0, 0.0, 0.0, 0.0
    for start in range(0, actual.shape[0], 256):
        value = (actual[start:start + 256].astype(np.uint32) << 16).view(np.float32).astype(np.float64)
        target = np.asarray(reference[start:start + 256], dtype=np.float64)
        difference = value - target
        different += int(np.count_nonzero(difference))
        maximum = max(maximum, float(np.abs(difference).max()))
        squared += float(np.sum(difference * difference))
        reference_squared += float(np.sum(target * target))
    return {'different_vs_mf': different, 'max_abs_vs_mf': maximum,
            'relative_l2_vs_mf': (squared / reference_squared) ** .5}


@torch.no_grad()
def main() -> None:
    """First validate transpose/ABI on a small exact product, then both full actual operands."""
    torch.npu.set_device(0)
    enable_full_determinism(42, close_matmul_k_shift=False)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    torch.npu.config.allow_internal_format = False
    torch.npu.matmul.allow_hf32 = False
    left_cpu = ((torch.arange(32 * 48).reshape(32, 48) % 7) - 3).float()
    right_cpu = ((torch.arange(32 * 64).reshape(32, 64) % 5) - 2).float()
    expected_smoke = (left_cpu.T @ right_cpu).numpy()
    left = left_cpu.to(device='npu', dtype=torch.bfloat16)
    right = right_cpu.to(device='npu', dtype=torch.bfloat16)
    smoke = matmul(left, right).float().cpu().numpy()
    if smoke.shape != (48, 64) or not np.array_equal(smoke, expected_smoke):
        raise ValueError('Explicit transpose or storage descriptor smoke failed')
    del left, right, smoke
    native = json.loads((ROOT / 'evidence/head-moments207-branches-comparison.json').read_text())['native']
    records = []
    for kernel in [2, 3]:
        source = native[str(kernel)]
        tensors, hashes = {}, {}
        for name in ['dy', 'x']:
            array = np.load(source[name]['path'], mmap_mode='r')
            tensors[name] = torch.from_numpy(np.array(array, copy=True)).to(device='npu', dtype=torch.bfloat16)
            hashes[name] = sha(bits(tensors[name]))
            if hashes[name] != source[name]['bf16_sha256']:
                raise ValueError('Actual device input differs from native reference')
        baseline = torch.mm(tensors['dy'].T, tensors['x'])
        baseline_sha = sha(bits(baseline))
        expected = json.loads((ROOT / 'evidence' / f'head207-gemm-screen-op{kernel}.json').read_text())
        if baseline_sha != expected['results'][0]['bf16_sha256']:
            raise ValueError('Ordinary mm does not reproduce original HP baseline')
        del baseline
        output = matmul(tensors['dy'], tensors['x'])
        value = bits(output)
        row = {'kernel': kernel, 'input_hashes': hashes, 'ordinary_mm_hash': baseline_sha,
               'bf16_sha256': sha(value), 'storage_metadata': list(LAST_METADATA), 'comparison': compare(value, np.load(source['dw']['path'], mmap_mode='r'))}
        records.append(row)
        print(json.dumps(row), flush=True)
        del tensors, output, value, array
        gc.collect()
        torch.npu.empty_cache()
    report = {'complete': True, 'small_exact_transpose_smoke': True, 'jit_compile': True, 'input_output_storage': 'FRACTAL_NZ', 'records': records,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'helper_sha256': hashlib.sha256((Path(__file__).parent / 'head_gemm_aclop_nz.py').read_bytes()).hexdigest(),
              'loaded_libraries': sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                                         if any(name in line for name in ['libopapi.so', 'libacl_op_compiler.so', 'libascendcl.so'])}),
              'scope': 'Same-input isolated explicit legacy operator with NZ storage replay, not a training acceptance result.'}
    (ROOT / 'evidence/head207-gemm-aclop-v2-replay.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
