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

"""Screen independent full-head GEMM routes on identical actual207 BF16 operands."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch_npu

from hyper_parallel.trainer.runtime.random import enable_full_determinism

ROOT = Path('/alignment/phase12')
STATE = Path('/state/jt-recipe-20261014')
BLOCK = 1_000_000


def bits(value: torch.Tensor) -> np.ndarray:
    """Read exact logical BF16 storage after device synchronization."""
    if value.dtype != torch.bfloat16:
        raise ValueError('GEMM candidate must end at the native BF16 output boundary')
    return value.detach().cpu().contiguous().view(torch.int16).numpy().view(np.uint16)


def sha(array: np.ndarray) -> str:
    """Hash contiguous storage without a second full byte copy."""
    return hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast('B')).hexdigest()


def as_fp32(value: np.ndarray) -> np.ndarray:
    """Interpret stored BF16 bits exactly as FP32 values."""
    return (value.astype(np.uint32) << np.uint32(16)).view(np.float32)


def compare(output: np.ndarray, native: np.ndarray, patch: dict, ideal: dict) -> dict:
    """Measure every output and separately label coverage of prior differing-coordinate FP64 references."""
    flat, reference = output.reshape(-1), native.reshape(-1)
    count_mf = count_hp = 0
    maximum = error_squared = reference_squared = 0.0
    for start in range(0, flat.size, BLOCK):
        stop = min(start + BLOCK, flat.size)
        mf = reference[start:stop]
        hp_bits = (mf.view(np.uint32) >> np.uint32(16)).astype(np.uint16)
        low, high = np.searchsorted(patch['indices'], [start, stop])
        hp_bits[patch['indices'][low:high] - start] = patch['hp'][low:high]
        actual = flat[start:stop]
        count_hp += int(np.count_nonzero(actual != hp_bits))
        value = as_fp32(actual)
        count_mf += int(np.count_nonzero(value != mf))
        difference = value.astype(np.float64) - mf.astype(np.float64)
        maximum = max(maximum, float(np.abs(difference).max()))
        error_squared += float(np.dot(difference, difference))
        reference64 = mf.astype(np.float64)
        reference_squared += float(np.dot(reference64, reference64))
    selected = as_fp32(flat[ideal['indices']]).astype(np.float64)
    error = np.abs(selected - ideal['reference'])
    mf_error = np.abs(ideal['mf'] - ideal['reference'])
    hp_error = np.abs(ideal['hp'] - ideal['reference'])
    return {'size': int(flat.size), 'different_vs_mf': count_mf, 'different_vs_actual_hp': count_hp,
            'max_abs_vs_mf': maximum, 'relative_l2_vs_mf': float(np.sqrt(error_squared / reference_squared)),
            'fp64_prior_difference_coordinates_only': {
                'count': int(selected.size), 'mean_error': float(error.mean()), 'max_error': float(error.max()),
                'closer_than_mf': int(np.count_nonzero(error < mf_error)),
                'farther_than_mf': int(np.count_nonzero(error > mf_error)),
                'closer_than_hp': int(np.count_nonzero(error < hp_error)),
                'farther_than_hp': int(np.count_nonzero(error > hp_error)),
                'does_not_cover_new_different_coordinates': True}}


def matmul(dy: torch.Tensor, inputs: torch.Tensor, mode: str) -> tuple:
    """Execute a shape-general arithmetic/layout route without teacher outputs or coordinate corrections."""
    torch.npu.config.allow_internal_format = mode.startswith('nz_')
    torch.npu.matmul.allow_hf32 = False
    if mode == 'nd_transpose_view':
        left, right = dy.T, inputs
        result = torch.mm(left, right)
    elif mode == 'nd_transpose_contiguous':
        left, right = dy.T.contiguous(), inputs
        result = torch.mm(left, right)
    elif mode == 'single_batch_bmm':
        left, right = dy.T.unsqueeze(0), inputs.unsqueeze(0)
        result = torch.bmm(left, right).squeeze(0)
    elif mode == 'native_bmmv2':
        left, right = dy.T, inputs
        result = torch_npu.npu_bmmV2(left, right, [])
    elif mode == 'fp32_operands':
        left, right = dy.float().T, inputs.float()
        result = torch.mm(left, right).to(torch.bfloat16)
    elif mode in ('nz_left', 'nz_both'):
        left = torch_npu.npu_format_cast(dy.T.contiguous(), 29)
        right = torch_npu.npu_format_cast(inputs, 29) if mode == 'nz_both' else inputs
        result = torch.mm(left, right)
    else:
        raise ValueError(mode)
    formats = {'left': torch_npu.get_npu_format(left), 'right': torch_npu.get_npu_format(right),
               'output': torch_npu.get_npu_format(result), 'left_shape': list(left.shape),
               'left_stride': list(left.stride()), 'right_shape': list(right.shape),
               'right_stride': list(right.stride()), 'operand_dtype': str(left.dtype)}
    return result, formats


@torch.no_grad()
def main() -> None:
    """Require full actual-HP baseline reproduction before screening the remaining routes."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--kernel', type=int, choices=[2, 3], required=True)
    args = parser.parse_args()
    source = json.loads((ROOT / 'evidence/head-moments207-branches-comparison.json').read_text())
    patches = json.loads((ROOT / 'evidence/head-moments207-branch-compact-reconstruction.json').read_text())
    correctness = json.loads((ROOT / 'evidence/head-moments207-fp64-correctness.json').read_text())
    if not all(row['complete'] for row in [source, patches, correctness]):
        raise ValueError('Native operand evidence incomplete')
    pair = next(row for row in source['comparisons'] if row['native_kernel'] == args.kernel)
    call = str(pair['hp_forward_call'])
    native = source['native'][str(args.kernel)]
    patch_record = next(row for row in patches['branches'] if row['native_kernel'] == args.kernel)
    ideal_record = next(row for row in correctness['comparisons'] if row['native_kernel'] == args.kernel)
    loaded = []
    for path, expected in [(patch_record['patch'], patch_record['patch_sha256']),
                           (ideal_record['local_evidence'], ideal_record['local_evidence_sha256'])]:
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError('Compact reference changed')
        with np.load(path) as archive:
            loaded.append({key: archive[key].copy() for key in archive.files})
    patch, ideal = loaded
    torch.npu.set_device(0)
    enable_full_determinism(42, close_matmul_k_shift=False)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    torch.npu.config.allow_internal_format = False
    torch.npu.matmul.allow_hf32 = False
    tensors, input_hashes = {}, {}
    for kind in ['x', 'dy']:
        array = np.load(native[kind]['path'], mmap_mode='r')
        tensors[kind] = torch.from_numpy(np.array(array, copy=True)).to(device='npu', dtype=torch.bfloat16)
        actual = sha(bits(tensors[kind]))
        if actual != native[kind]['bf16_sha256']:
            raise ValueError('Actual device operand differs from native evidence')
        input_hashes[kind] = actual
    reference = np.load(native['dw']['path'], mmap_mode='r')
    directory = ROOT / 'runs' / ('head207-gemm-screen-op' + str(args.kernel))
    directory.mkdir(exist_ok=False)
    results = []
    report = {'complete': False, 'kernel': args.kernel, 'hp_forward_call': int(call), 'input_hashes': input_hashes,
              'torch': torch.__version__, 'torch_npu': torch_npu.__version__,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'baseline_matches_actual_hp': False, 'results': results,
              'scope': 'Full-size same-input replay only. Runtime/transfer times are not training performance claims. '
                       'No teacher state or coordinate values enter candidate arithmetic.'}
    modes = ['nd_transpose_view', 'nd_transpose_contiguous', 'single_batch_bmm',
             'native_bmmv2', 'nz_left', 'nz_both', 'fp32_operands']
    for mode in modes:
        started = time.monotonic()
        try:
            output, formats = matmul(tensors['dy'], tensors['x'], mode)
            torch.npu.synchronize()
            value = bits(output)
            row = {'mode': mode, 'status': 'completed', 'formats': formats, 'bf16_sha256': sha(value),
                   'comparison': compare(value, reference, patch, ideal)}
            if mode == modes[0]:
                expected = source['hp'][call]['dw']['bf16_sha256']
                report['baseline_matches_actual_hp'] = row['bf16_sha256'] == expected
                if not report['baseline_matches_actual_hp']:
                    raise ValueError('Direct full-size GEMM did not reproduce the actual HP output')
            del output, value
        except RuntimeError as error:
            if mode == modes[0]:
                raise
            torch.npu.synchronize()
            row = {'mode': mode, 'status': 'unsupported_or_runtime_error', 'error': str(error)[:3000]}
        finally:
            torch.npu.config.allow_internal_format = False
            torch.npu.matmul.allow_hf32 = False
        row['elapsed_seconds_including_analysis'] = time.monotonic() - started
        results.append(row)
        (directory / 'screen.json').write_text(json.dumps(report, indent=2))
        print('[P12-HEAD207-GEMM-SCREEN]', args.kernel, json.dumps(row), flush=True)
        gc.collect()
        torch.npu.empty_cache()
    report['complete'] = True
    (directory / 'screen.json').write_text(json.dumps(report, indent=2))
    (ROOT / 'evidence' / ('head207-gemm-screen-op' + str(args.kernel) + '.json')).write_text(
        json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
