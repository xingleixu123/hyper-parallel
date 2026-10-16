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

"""Separate native square-sum substitution from independent live reductions."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch_npu

from hyper_parallel.trainer.runtime.random import enable_full_determinism
from muon_q_norm_probe import half_tree

ROOT = Path('/alignment/phase12')
STATE = Path('/state/jt-recipe-20261014')
EPSILON = 1.0011717677116394e-7


def sequential(values: np.ndarray) -> np.float32:
    """Round each addition to FP32 in its specified order."""
    total = np.float32(0)
    for value in values:
        total = np.float32(total + value)
    return total


def main() -> None:
    """Only diagnose a replay; native scalars cannot enter independent training."""
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    enable_full_determinism(42, close_matmul_k_shift=False)
    records = []
    for piece in [0, 1]:
        value = np.load(STATE / 'hp-frontier258-muon-replay' /
                        ('main1_q_up-' + str(piece)) / 'pre_norm.npy')
        current = torch.from_numpy(value).npu()
        paths = list((STATE / 'frontier258-mf-builtin-tensor-dump').rglob(
            '*-op' + str(piece) + '.*.input.0.DefaultFormat.*.npy'))
        expected_paths = list((STATE / 'mf-frontier258-muon-replay').glob(
            'main1_q_up-' + str(piece) + '-normalized_*.npy'))
        if len(paths) != 1 or len(expected_paths) != 1:
            raise ValueError('Expected unique faithful native boundaries')
        native_sum = np.load(paths[0]).reshape(())
        expected = np.load(expected_paths[0])
        native_tensor = torch.from_numpy(native_sum).npu()
        totals = {
            'teacher_native_square_sum': native_sum,
            'independent_npu_square_sum': current.square().sum().cpu().numpy(),
            'independent_fp64_square_sum_rounded': np.float32(np.square(value, dtype=np.float64).sum()),
        }
        tiles = (current.numel() + 9775) // 9776
        padded = current.new_zeros(tiles * 9776)
        padded[:current.numel()] = current.reshape(-1).square()
        lanes = half_tree(padded.reshape(tiles, 611, 16).transpose(-1, -2))
        for name, work in [('half', lanes), ('adjacent', lanes)]:
            if name == 'half':
                partials = half_tree(work).cpu().numpy()
            else:
                while work.shape[-1] > 1:
                    work = work[..., ::2] + work[..., 1::2]
                partials = work[..., 0].cpu().numpy()
            totals['independent_tile9776_' + name + '_sequential'] = sequential(partials)
            for cores in [20, 40]:
                accumulators = np.zeros(cores, dtype=np.float32)
                for index, partial in enumerate(partials):
                    accumulators[index % cores] = np.float32(accumulators[index % cores] + partial)
                totals['independent_tile9776_' + name + '_cores' + str(cores)] = sequential(accumulators)
        candidates = {}
        for name, total in totals.items():
            total = np.asarray(total, dtype=np.float32).reshape(())
            for sqrt_mode in ['host', 'device']:
                scalar = torch.from_numpy(total).npu()
                norm = scalar.sqrt() if sqrt_mode == 'device' else torch.tensor(np.sqrt(total), device=current.device)
                denominator = norm + EPSILON
                output = (current / denominator).bfloat16().float().cpu().numpy()
                candidates[name + '_' + sqrt_mode] = {
                    'square_sum': float(total), 'square_sum_uint32': int(total.view(np.uint32)),
                    'norm': float(norm.cpu()), 'denominator': float(denominator.cpu()),
                    'different': int(np.count_nonzero(output != expected)),
                    'max_abs': float(np.max(np.abs(output - expected))),
                    'output_sha256': hashlib.sha256(output.tobytes()).hexdigest(),
                    'teacher_scalar': name.startswith('teacher_'),
                }
        record = {'piece': piece, 'shape': list(value.shape),
                  'native_square_sum': float(native_sum), 'native_sum_uint32': int(native_sum.view(np.uint32)),
                  'input_sha256': hashlib.sha256(value.tobytes()).hexdigest(), 'candidates': candidates}
        records.append(record)
        print(json.dumps(record), flush=True)
    report = {'complete': True, 'diagnostic_only': True, 'eligible_for_acceptance': False,
              'records': records, 'arrays_written': False,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (ROOT / 'evidence/frontier258-square-sum-screen.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
