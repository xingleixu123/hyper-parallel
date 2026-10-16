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

"""Replay actual258 HP Muon updates and expose boundaries only after full-state fidelity."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch_npu

from hyper_parallel.core.optimizer import muon as muon_module
from hyper_parallel.core.optimizer.muon import NSInputTransform
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

ROOT = Path('/alignment/phase12')
STATE = Path('/state/jt-recipe-20261014')
OUTPUT = STATE / 'hp-frontier258-muon-replay'
RECORDS = []


def digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def save(directory, name, value):
    array = value.detach().float().cpu().numpy()
    np.save(directory / (name + '.npy'), array, allow_pickle=False)
    return {'name': name, 'shape': list(array.shape), 'sha256': digest(array)}


def transform(name, value):
    if not name.endswith('.self_attn.q_b_proj.weight') or tuple(value.shape) != (24576, 1536):
        raise ValueError('Expected the selected full Q-up matrix')
    shaped = value.reshape(128, 192, 1536)
    parts = [shaped[:, :128].reshape(-1, 1536).contiguous(),
             shaped[:, 128:].reshape(-1, 1536).contiguous()]

    def restore(updates, output):
        first = updates[0].reshape(128, 128, 1536)
        second = updates[1].reshape(128, 64, 1536)
        output.copy_(torch.cat((first, second), dim=1).reshape_as(output))

    return NSInputTransform(parts, restore)


def install_trace():
    original = muon_jit_probe.graph_jit_ns
    count = 0

    def traced(value, steps, normalized=None):
        nonlocal count
        if normalized is not None or tuple(value.shape) not in [(16384, 1536), (8192, 1536)]:
            raise ValueError('Unexpected Q-up NS boundary')
        index = count
        count += 1
        directory = OUTPUT / ('main1_q_up-' + str(index))
        directory.mkdir(exist_ok=False)
        previous = torch_npu.npu.is_jit_compile_false()
        torch_npu.npu.set_compile_mode(jit_compile=False)
        try:
            current = value.mT
            norm = current.norm(dim=(-2, -1), keepdim=True)
            denominator = norm + hp_ns_epsilon229_factor.EPSILON
            candidate = (current / denominator).bfloat16()
            scalar = {'norm': norm.float().cpu().reshape(-1).tolist(),
                      'denominator': denominator.float().cpu().reshape(-1).tolist()}
            rows = [save(directory, 'pre_norm', current), save(directory, 'normalized', candidate)]
            raw = current.float().cpu().numpy()
            norm64 = float(np.sqrt(np.sum(np.square(raw, dtype=np.float64))))
        finally:
            torch_npu.npu.set_compile_mode(jit_compile=not previous)
        expected = original(value, steps, normalized)
        observed = original(value, steps, candidate)
        exact = torch.equal(expected, observed)
        rows.append(save(directory, 'ns_output', expected))
        RECORDS.append({'label': 'main1_q_up', 'piece': index, 'explicit_normalization_exact': exact,
                        'scalar': scalar, 'norm_fp64': norm64, 'arrays': rows})
        if not exact:
            raise ValueError('Explicit norm changed the actual NS output')
        return expected

    muon_jit_probe.graph_jit_ns = traced


def main():
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
    install_trace()
    policy = muon_broadcast_fp32_factor.install()
    muon_module.compute_muon_slice_scale = fp32_archive_scale
    state = STATE / 'hp-jtf637-weighted-state258'
    targets = {'main1_q_up': 'model.layers.1.self_attn.q_b_proj.weight'}
    checks = []
    try:
        for label, name in targets.items():
            weight = np.load(state / 'state-257' / (label + '-weight.npy'))
            momentum = np.load(state / 'state-257' / (label + '-muon_m.npy'))
            gradient = np.load(state / 'gradient-258' / (label + '-postclip.npy'))
            parameter = torch.nn.Parameter(torch.from_numpy(weight.copy()).npu())
            parameter.model_name = name
            parameter.grad = torch.from_numpy(gradient).npu()
            optimizer = AllMeasuredCastMuon([parameter], lr=1e-4, weight_decay=.1, momentum=.95,
                                            nesterov=True, ns_steps=5, ns_variant='legacy', ns_epsilon=1e-7,
                                            matched_adamw_rms=.2, apply_lr_in_update=True, ns_transform_fn=transform)
            optimizer.state[parameter]['momentum_buffer'] = torch.from_numpy(momentum).npu()
            for group in optimizer.param_groups:
                group['step'] = 257
            optimizer.step()
            actual = parameter.detach().cpu().numpy()
            actual_m = optimizer.state[parameter]['momentum_buffer'].cpu().numpy()
            expected = np.load(state / 'state-258' / (label + '-weight.npy'))
            expected_m = np.load(state / 'state-258' / (label + '-muon_m.npy'))
            check = {'label': label, 'weight_exact_training': np.array_equal(actual, expected),
                     'momentum_exact_training': np.array_equal(actual_m, expected_m),
                     'weight_sha256': digest(actual), 'expected_weight_sha256': digest(expected),
                     'momentum_sha256': digest(actual_m), 'optimizer_step': optimizer.param_groups[0]['step']}
            mf_weight = np.load(STATE / 'mf-frontier258-state/state-258/main1_q_up-weight.npy')
            check['weight_exact_mf_training'] = np.array_equal(actual, mf_weight)
            check['mf_weight_sha256'] = digest(mf_weight)
            checks.append(check)
            print(json.dumps(check), flush=True)
        report = {'complete': True, 'diagnostic_only': True, 'checks': checks, 'records': RECORDS,
                  'broadcast_policy': policy, 'epsilon_policy': epsilon_policy, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'warning': 'Only full weight and momentum fidelity establishes a replay of actual training.'}
        (ROOT / 'evidence/frontier258-hp-muon-replay.json').write_text(json.dumps(report, indent=2))
        if not all(c['weight_exact_training'] and c['momentum_exact_training'] for c in checks):
            raise ValueError('Replay did not reproduce the actual HP update')
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
