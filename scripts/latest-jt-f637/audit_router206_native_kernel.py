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

"""Audit the native MF206 router NS boundary against exact real-training operands."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch_npu
from hyper_parallel.trainer.runtime.random import enable_full_determinism
from compare_vocab_optimizer_trace import fidelity
from muon_jit_probe import graph_jit_ns
from muon_lr_fp32_scale import fp32_archive_scale
from screen_router206_ns import load, stats

ROOT = Path('/alignment/phase12')
STATE = Path('/state/jt-recipe-20261014')
RUN = 'mf-native-router-norm206'


def read_one(directory, pattern):
    files = list(directory.glob(pattern))
    if len(files) != 1:
        raise ValueError('Expected one kernel operand: ' + pattern)
    value = np.load(files[0])
    return value, {'file': str(files[0].relative_to(STATE)), 'shape': list(value.shape),
                   'dtype': str(value.dtype), 'sha256': hashlib.sha256(value.tobytes()).hexdigest()}


def main():
    checks = fidelity(RUN, 'mf-resume', 6)
    checks['exit_zero'] = (ROOT / 'runs' / (RUN + '.exit')).read_text().strip() == '0'
    if not all(checks.values()):
        raise ValueError(checks)
    source = np.load(STATE / 'router206-ns-screen/input.npy')
    hp_normalized = np.load(STATE / 'router206-ns-screen/normalized_hp.npy')
    directory = STATE / (RUN + '-kernels/rank_0/NativeNS/1/5')
    candidates = []
    for number in (19, 32):
        values, records = {}, {}
        for kind, suffix in [('square_sum', 'input.0'), ('input', 'input.1'), ('normalized', 'output.0')]:
            values[kind], records[kind] = read_one(directory, '*split-op' + str(number) + '.*.' + suffix + '.*.npy')
        candidates.append((number, values, records))
    matches = [(number, values, records) for number, values, records in candidates if np.array_equal(source, values['input'])]
    if len(matches) != 1:
        raise ValueError('Native router input did not match exactly one observed kernel')
    number, values, records = matches[0]
    torch.npu.set_device(0)
    enable_full_determinism(42, close_matmul_k_shift=False)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    tensor = torch.from_numpy(source).npu()
    native_sum = float(values['square_sum'].reshape(-1)[0])
    native_output = values['normalized']
    tensor_sum = (tensor * tensor).sum()
    variants = {
        'hp_tensor_norm': (tensor / (tensor.norm(dim=(-2, -1), keepdim=True) + 1e-7)).bfloat16(),
        'hp_square_sum_sqrt': (tensor / (tensor_sum.sqrt() + 1e-7)).bfloat16(),
        'native_sum_hp_sqrt': (tensor / (torch.tensor(native_sum, device=tensor.device).sqrt() + 1.0011717677116394e-7)).bfloat16(),
    }
    checks['stored_hp_normalized_exact'] = np.array_equal(variants['hp_tensor_norm'].float().cpu().numpy(), hp_normalized)
    parameter = torch.from_numpy(load('hp-jtf637-sharedup-router206-state', 205, 'weight')).npu()
    reference = load('mf-router206-state', 206, 'weight')
    update = graph_jit_ns(tensor, 5, torch.from_numpy(native_output).npu().bfloat16())
    scale = fp32_archive_scale(update, .2)
    scale *= -1e-4
    replay = (parameter * (1.0 - 1e-4 * .1) + update * scale).cpu().numpy()
    checks['native_normalized_replays_all_mf_weights'] = np.array_equal(replay, reference)
    exact_sum = math.fsum(float(value) * float(value) for value in source.reshape(-1))
    ideal_norm = math.sqrt(exact_sum)
    ideal = source.astype(np.float64) / (ideal_norm + 1e-7)
    different = hp_normalized != native_output
    hp_errors = abs(hp_normalized.astype(np.float64)[different] - ideal[different])
    mf_errors = abs(native_output.astype(np.float64)[different] - ideal[different])
    closer = {'different_normalized_points': int(np.count_nonzero(different)),
              'mf_closer': int(np.count_nonzero(mf_errors < hp_errors)),
              'hp_closer': int(np.count_nonzero(hp_errors < mf_errors)),
              'tie': int(np.count_nonzero(hp_errors == mf_errors)),
              'scope': 'Only differing actual206 normalized coordinates versus FP64 ideal of the same FP32 input; no general optimizer or full-trajectory accuracy claim.'}
    if not all(checks.values()):
        raise ValueError(checks)
    report = {'complete': True, 'fidelity': checks, 'native_kernel': number, 'native_files': records,
              'native_square_sum': native_sum, 'hp_square_sum': float(tensor_sum.item()),
              'hp_tensor_norm': float(tensor.norm().item()), 'fp64_reference_square_sum': exact_sum,
              'fp64_reference_norm': ideal_norm, 'normalized': {
                  key: stats(value.float().cpu().numpy(), native_output) for key, value in variants.items()},
              'correctness_at_boundary': closer,
              'scope': 'Actual unmodified full-model MF206 normalization dump; one normalized BF16 coordinate explains12 updated router weights under same-input replay. Independent training square-sum formula was already rejected due earlier205 side effects.'}
    (ROOT / 'evidence/router206-native-kernel-audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
