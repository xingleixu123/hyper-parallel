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

"""Screen scalar square roots from the same actual gradient-square totals."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch_npu

from hyper_parallel.trainer.runtime.random import enable_full_determinism

ROOT = Path('/alignment/phase12')


def main() -> None:
    """Compare CPU/NPU outputs without changing or replaying training states."""
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    enable_full_determinism(42, close_matmul_k_shift=False)
    path = ROOT / 'runs/hp-jtf637-epsilon400/CLIP_REDUCTION.jsonl'
    records = [json.loads(line) for line in path.read_text().splitlines()][:35]
    if [row['step'] for row in records] != list(range(201, 236)):
        raise ValueError('Expected the actual first 35 scalar inputs')
    output = []
    for row in records:
        total = np.float32(row['square_sum'])
        cpu = np.sqrt(total)
        if float(cpu) != row['norm']:
            raise ValueError('CPU square root does not reproduce its training scalar')
        scalar = torch.tensor([total], dtype=torch.float32, device='npu')
        actual = scalar.sqrt().cpu().numpy()
        reference = math.sqrt(float(total))
        output.append({
            'step': row['step'], 'square_sum': float(total), 'cpu': float(cpu), 'npu': float(actual[0]),
            'fp64_reference': reference, 'cpu_abs_error': abs(float(cpu) - reference),
            'npu_abs_error': abs(float(actual[0]) - reference),
            'cpu_sha1': hashlib.sha1(np.asarray(cpu).tobytes()).hexdigest(),
            'npu_sha1': hashlib.sha1(actual.tobytes()).hexdigest(),
        })
    report = {
        'complete': True, 'diagnostic_only': True, 'eligible_for_acceptance': False,
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'actual_input_source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'records': output,
        'scope': 'Identical HP live total sums; MF equality additionally requires native input/hash evidence. '
                 'FP64 is an accuracy reference, not the native DVM scalar output.',
    }
    (ROOT / 'evidence/frontier235-global-sqrt-replay.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'complete': True, 'different': [row for row in output if row['cpu'] != row['npu']]}))


if __name__ == '__main__':
    main()
