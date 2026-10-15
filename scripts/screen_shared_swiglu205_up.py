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

"""Replay SwiGLU up-gradient arithmetic on an actual affected token row."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional
import torch_npu
from hyper_parallel.trainer.runtime.random import enable_full_determinism
from compare_shared_activation_row205 import load_array
from compare_head_selected_operands import stats

ROOT = Path('/alignment/phase12')
STATE = Path('/state/jt-recipe-20261014')


def main():
    """Compare independent arithmetic variants without changing any training state."""
    evidence = json.loads((ROOT / 'evidence/shared-activation-row205-comparison.json').read_text())
    if not evidence['complete'] or not all(row['exact'] for row in evidence['comparisons'].values()):
        raise ValueError('Same-input activation evidence required')
    hp = 'hp-jtf637-adamw-all-sharedfc1206'
    mf = 'mf-sharedfc1-205206'
    rows = json.loads((ROOT / 'runs' / hp / 'SHARED_FC1_BOUNDARIES.json').read_text())['records']
    mf_rows = next(json.loads(line) for line in (ROOT / 'runs' / mf / 'SHARED_FC1_BOUNDARIES.jsonl').read_text().splitlines() if json.loads(line)['step'] == 205)['records']
    projected = load_array(rows, STATE / hp / 'shared_fc1', 'y').reshape(4096, 6144)
    hp_expected = load_array(rows, STATE / hp / 'shared_fc1', 'dy').reshape(4096, 6144)[1605, 1::2]
    mf_expected = load_array(mf_rows, STATE / mf / 'deep', 'dy').reshape(4096, 6144)[1605, 1::2]
    act = 'hp-jtf637-adamw-all-sharedactrow206'
    act_rows = json.loads((ROOT / 'runs' / act / 'SHARED_ACTIVATION_ROW.json').read_text())['records']
    incoming = load_array(act_rows, STATE / act / 'activation_row', 'dx')
    enable_full_determinism(1, close_matmul_k_shift=False)
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    projection = torch.from_numpy(np.array(projected)).to(device='npu', dtype=torch.bfloat16)
    gate, up = projection[:, 0::2], projection[:, 1::2]
    pair = torch.cat((gate, up), dim=-1).unsqueeze(0)
    gradient = torch.zeros((1, 4096, 3072), device='npu', dtype=torch.bfloat16)
    gradient[0, 1605] = torch.from_numpy(np.array(incoming)).to(device='npu', dtype=torch.bfloat16)
    gate = pair[..., :3072]
    g32, d32 = gate.float(), gradient.float()
    variants = [
        ('native_swiglu', lambda: torch.ops.npu.npu_swiglu_backward(gradient, pair, -1).chunk(2, -1)[1]),
        ('bf16_silu_then_multiply', lambda: gradient * functional.silu(gate)),
        ('fp32_silu_then_multiply', lambda: (d32 * functional.silu(g32)).bfloat16()),
        ('fp32_gate_sigmoid_then_dy', lambda: ((g32 * g32.sigmoid()) * d32).bfloat16()),
        ('fp32_dy_gate_then_sigmoid', lambda: ((d32 * g32) * g32.sigmoid()).bfloat16()),
        ('fp32_dy_sigmoid_then_gate', lambda: ((d32 * g32.sigmoid()) * g32).bfloat16()),
        ('fp32_division_then_dy', lambda: ((g32 / (1 + (-g32).exp())) * d32).bfloat16()),
        ('fp32_dy_gate_then_division', lambda: ((d32 * g32) / (1 + (-g32).exp())).bfloat16()),
    ]
    records = []
    for name, operation in variants:
        value = operation()[0, 1605].float().cpu().numpy()
        row = {'variant': name, 'vs_hp': stats(value, hp_expected), 'vs_mf': stats(value, mf_expected),
               'critical_value': float(value[2309]), 'sha256': hashlib.sha256(value.tobytes()).hexdigest()}
        records.append(row)
        print(json.dumps(row), flush=True)
        if name == 'native_swiglu' and not row['vs_hp']['exact']:
            raise ValueError('Full-shape native replay failed to reproduce the actual row')
    report = {'complete': True, 'records': records, 'scope': 'Full4096x6144 actual gate/up inputs; incoming gradient uses the actual token1605 row, zeros elsewhere. Native replay must reproduce all3072 actual HP up derivatives. Elementwise replay only; no training result.',
              'runtime': {'torch': torch.__version__, 'CLOSE_MATMUL_K_SHIFT': os.environ.get('CLOSE_MATMUL_K_SHIFT')},
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (ROOT / 'evidence/shared-swiglu205-up-screen.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
