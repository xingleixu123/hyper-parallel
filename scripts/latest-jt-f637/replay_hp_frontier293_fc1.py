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

"""Replay actual293 HP Muon updates and expose boundaries only after full-state fidelity."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch_npu

from hyper_parallel.core.optimizer import muon as muon_module
from hyper_parallel.trainer.runtime.random import enable_full_determinism
from muon_all_casts_probe import AllMeasuredCastMuon
from muon_lr_fp32_scale import fp32_archive_scale
import muon_broadcast_fp32_factor
import muon_dense_adjacent_factor
import muon_jit_probe
import muon_ns_eager_factor
import muon_q_native_tile_factor
import router_ns_dvm_factor
import muon_expert_bmm_eager_factor
import hp_ns_epsilon229_factor
import hp_fc1_norm293_factor

ROOT = Path('/alignment/phase12')
STATE = Path('/state/jt-recipe-20261014')
OUTPUT = STATE / 'hp-frontier293-fc1-replay'
RECORDS = []


def digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def main():
    audit = json.loads((ROOT / 'evidence/frontier293-state-comparison.json').read_text())
    if not audit['complete'] or not all(all(item.values()) for item in audit['fidelity'].values()):
        raise ValueError('Training state fidelity not established')
    OUTPUT.mkdir(exist_ok=False)
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    enable_full_determinism(42, close_matmul_k_shift=False)
    os.environ['CLOSE_MATMUL_K_SHIFT'] = '0'
    os.environ['P12_NS_EAGER_SCOPE'] = 'all'
    dist.init_process_group('hccl')
    epsilon_policy = hp_ns_epsilon229_factor.install()
    router_ns_dvm_factor.install()
    muon_ns_eager_factor.install()
    muon_q_native_tile_factor.install()
    muon_expert_bmm_eager_factor.install()
    muon_dense_adjacent_factor.install()
    fc1_policy = hp_fc1_norm293_factor.install()
    policy = muon_broadcast_fp32_factor.install()
    muon_module.compute_muon_slice_scale = fp32_archive_scale
    state = STATE / 'hp-jtf637-qup-state293-r3'
    targets = {'main0_gate': 'model.layers.0.mlp.gate_proj.weight',
               'main0_up': 'model.layers.0.mlp.up_proj.weight'}
    checks = []
    try:
        for index, (label, name) in enumerate(targets.items()):
            weight = np.load(state / 'state-292' / (label + '-weight.npy'))
            momentum = np.load(state / 'state-292' / (label + '-muon_m.npy'))
            gradient = np.load(state / 'gradient-293/main0_fc1-postclip.npy', mmap_mode='r')[index::2].copy()
            parameter = torch.nn.Parameter(torch.from_numpy(weight.copy()).npu())
            parameter.model_name = name
            parameter.grad = torch.from_numpy(gradient).npu()
            optimizer = AllMeasuredCastMuon([parameter], lr=1e-4, weight_decay=.1, momentum=.95,
                                            nesterov=True, ns_steps=5, ns_variant='legacy', ns_epsilon=1e-7,
                                            matched_adamw_rms=.2, apply_lr_in_update=True, ns_transform_fn=None)
            optimizer.state[parameter]['momentum_buffer'] = torch.from_numpy(momentum).npu()
            for group in optimizer.param_groups:
                group['step'] = 292
            optimizer.step()
            actual = parameter.detach().cpu().numpy()
            actual_m = optimizer.state[parameter]['momentum_buffer'].cpu().numpy()
            expected = np.load(state / 'state-293' / (label + '-weight.npy'))
            expected_m = np.load(state / 'state-293' / (label + '-muon_m.npy'))
            check = {'label': label, 'weight_exact_training': np.array_equal(actual, expected),
                     'momentum_exact_training': np.array_equal(actual_m, expected_m),
                     'weight_sha256': digest(actual), 'expected_weight_sha256': digest(expected),
                     'momentum_sha256': digest(actual_m), 'optimizer_step': optimizer.param_groups[0]['step']}
            mf_weight = np.load(STATE / 'mf-frontier293-state/state-293/main0_fc1-weight.npy', mmap_mode='r')[index::2]
            check['weight_exact_mf_training'] = np.array_equal(actual, mf_weight)
            check['mf_weight_sha256'] = digest(mf_weight)
            checks.append(check)
            print(json.dumps(check), flush=True)
        report = {'complete': True, 'diagnostic_only': True, 'checks': checks, 'records': RECORDS,
                  'broadcast_policy': policy, 'epsilon_policy': epsilon_policy, 'fc1_policy': fc1_policy, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'warning': 'Candidate recomputes every value from HP pre-update states; MF state is used only for output validation.'}
        (ROOT / 'evidence/frontier293-hp-fc1-replay.json').write_text(json.dumps(report, indent=2))
        if not all(c['weight_exact_mf_training'] and c['momentum_exact_training'] for c in checks):
            raise ValueError('Independent normalization did not reproduce the actual MF update')
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
