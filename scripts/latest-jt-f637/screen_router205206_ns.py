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

"""Screen router205 and206 NS normalization using identical actual state and gradient."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch_npu
from hyper_parallel.trainer.runtime.random import enable_full_determinism
from muon_jit_probe import graph_jit_ns
from muon_lr_fp32_scale import fp32_archive_scale
from observe_hp_preclip_norm_adjacent import dvm_square_sum

ROOT = Path('/alignment/phase12')
STATE = Path('/state/jt-recipe-20261014')


def load(run, step, kind):
    report = json.loads((ROOT / 'runs' / run / ('ROUTER_STATE-' + str(step) + '.json')).read_text())
    row = next(row for row in report['records'] if row['kind'] == kind)
    value = np.load(STATE / run / ('state-' + str(step)) / row['file'])
    if hashlib.sha256(value.tobytes()).hexdigest() != row['sha256']:
        raise ValueError('State content changed')
    return value


def stats(actual, expected):
    error = actual.astype(np.float64) - expected.astype(np.float64)
    return {'different': int(np.count_nonzero(error)), 'max_abs': float(np.abs(error).max()),
            'relative_l2': float(np.linalg.norm(error) / np.linalg.norm(expected.astype(np.float64)))}


def screen(step):
    torch.npu.set_device(0)
    enable_full_determinism(42, close_matmul_k_shift=False)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    run = 'hp-jtf637-sharedup-router206-state'
    gradient_row = next(row for row in json.loads((ROOT / 'runs' / run / ('ROUTER_GRADIENT-' + str(step) + '.json')).read_text())['records'] if row['kind'] == 'postclip')
    gradient = np.load(STATE / run / ('gradient-' + str(step)) / gradient_row['file'])
    if hashlib.sha256(gradient.tobytes()).hexdigest() != gradient_row['sha256']:
        raise ValueError('Gradient changed')
    gradient = torch.from_numpy(gradient).npu()
    parameter = torch.from_numpy(load(run, step - 1, 'weight')).npu()
    momentum = torch.from_numpy(load(run, step - 1, 'muon_m')).npu()
    next_m = momentum * .95 + gradient
    if not np.array_equal(next_m.cpu().numpy(), load(run, step, 'muon_m')):
        raise ValueError('Actual next momentum not reproduced')
    value = gradient + next_m * .95
    cpu = value.cpu().numpy()
    expected_hp = load(run, step, 'weight')
    expected_mf = load('mf-router206-state', step, 'weight')
    scale = fp32_archive_scale(value, .2)
    scale *= -1e-4
    def weight(update):
        return (parameter * (1.0 - 1e-4 * .1) + update * scale).cpu().numpy()
    native = weight(graph_jit_ns(value, 5))
    if not np.array_equal(native, expected_hp):
        raise ValueError('Same-input replay did not reproduce all actual HP updated weights')
    native_norm = np.float32(value.norm(dim=(-2, -1), keepdim=True).item())
    norms = {'torch_native': native_norm, 'numpy_fp32': np.float32(np.linalg.norm(cpu)),
             'numpy_fp64_rounded': np.float32(np.linalg.norm(cpu.astype(np.float64))),
             'torch_square_sum_sqrt': np.float32((value * value).sum().sqrt().item()),
             'dvm_clip_sum_cpu_sqrt': np.float32(np.sqrt(dvm_square_sum(value))),
             'native_minus_one_ulp_diagnostic': np.nextafter(native_norm, np.float32(-np.inf)),
             'native_plus_one_ulp_diagnostic': np.nextafter(native_norm, np.float32(np.inf))}
    rows = []
    native_normalized = (value / (value.norm(dim=(-2, -1), keepdim=True) + 1e-7)).bfloat16()
    for name, norm in norms.items():
        normalized = (value / (torch.tensor(float(norm), device=value.device) + 1e-7)).bfloat16()
        result = weight(graph_jit_ns(value, 5, normalized))
        rows.append({'variant': name, 'norm': float(norm), 'normalized_diff_native': int(torch.count_nonzero(normalized != native_normalized).item()),
                     'vs_mf_weight': stats(result, expected_mf), 'vs_hp_weight': stats(result, expected_hp)})
    return {'step': step, 'actual_hp_weight_exact': True, 'actual_hp_momentum_exact': True,
            'records': rows}


def main():
    report = {'complete': True, 'steps': [screen(step) for step in [205, 206]],
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'scope': 'Same-input205/206 screening only; full actual state must reproduce. No arrays written; no continuous-trajectory claim.'}
    (ROOT / 'evidence/router205206-ns-screen.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
