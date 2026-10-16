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

"""Screen independent normalization expressions on identical actual NS inputs."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch_npu

from hyper_parallel.trainer.runtime.random import enable_full_determinism

ROOT = Path('/alignment/phase12')
STATE = Path('/state/jt-recipe-20261014')


def main() -> None:
    """Report diagnostics only; no candidate is accepted without independent training."""
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    enable_full_determinism(42, close_matmul_k_shift=False)
    records = []
    for piece in [0, 1]:
        path = STATE / 'hp-frontier229-muon-replay' / ('mtp_q_up-' + str(piece)) / 'pre_norm.npy'
        value = np.load(path)
        actual = torch.from_numpy(value).npu()
        mf_paths = list((STATE / 'mf-frontier229-muon-replay').glob('mtp_q_up-' + str(piece) + '-normalized_*.npy'))
        if len(mf_paths) != 1:
            raise ValueError('Expected one actual MF normalized boundary')
        expected = np.load(mf_paths[0])
        norm = actual.norm(dim=(-2, -1), keepdim=True)
        norm64 = float(np.sqrt(np.square(value, dtype=np.float64).sum()))
        scalar64 = torch.tensor([[norm64]], dtype=torch.float32, device=actual.device)
        denominators = {
            'fp32_native_norm': norm + 1e-7,
            'norm64_rounded_fp32': scalar64 + 1e-7,
            'norm_bf16_then_eps_fp32': norm.bfloat16().float() + 1e-7,
            'norm_plus_eps_bf16': (norm + 1e-7).bfloat16().float(),
            'norm_bf16_eps_bf16': (norm.bfloat16() + 1e-7).float(),
            'native_square_sum_sqrt': actual.square().sum().sqrt().reshape(1, 1) + 1e-7,
        }
        candidates = {}
        for name, denominator in denominators.items():
            output = (actual / denominator).bfloat16().float().cpu().numpy()
            candidates[name] = {'denominator': float(denominator.cpu().reshape(-1)[0]),
                                'different': int(np.count_nonzero(output != expected)),
                                'max_abs': float(np.max(np.abs(output - expected)))}
        record = {'piece': piece, 'shape': list(value.shape), 'norm_fp64': norm64,
                  'input_sha256': hashlib.sha256(value.tobytes()).hexdigest(), 'candidates': candidates}
        records.append(record)
        print(json.dumps(record), flush=True)
    report = {'complete': True, 'diagnostic_only': True, 'eligible_for_acceptance': False, 'records': records,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (ROOT / 'evidence/frontier229-norm-expression-screen.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
