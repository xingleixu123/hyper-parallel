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

"""Replay actual329 HP Muon updates and expose boundaries only after full-state fidelity."""
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
import hp_norm219_factor
from mindspore272_stat_hash import native_dump_sha1

ROOT = Path('/alignment/phase12')
STATE = Path('/state/jt-recipe-20261014')
OUTPUT = STATE / 'hp-frontier329-control-replay'
RECORDS = []


def digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def describe(value: torch.Tensor) -> dict:
    """Record complete logical hashes and bounded statistics without writing arrays."""
    array = value.detach().float().cpu().numpy()
    if value.dtype == torch.bfloat16:
        native = value.detach().contiguous().cpu().view(torch.uint16).numpy()
    else:
        native = array
    return {'shape': list(array.shape), 'dtype': str(value.dtype),
            'fp32_sha256': digest(array), 'native_dump_sha1': native_dump_sha1(native)}


def install_trace() -> None:
    """Observe all eight expert normalizations while returning the original control result."""
    original = muon_jit_probe.graph_jit_ns

    def traced(value, steps, normalized=None):
        if normalized is not None or tuple(value.shape) != (8, 5120, 1536):
            raise ValueError('Unexpected expert-down NS boundary')
        current = value.mT
        previous = torch_npu.npu.is_jit_compile_false()
        torch_npu.npu.set_compile_mode(jit_compile=False)
        try:
            norm = current.float().norm(dim=(-2, -1), keepdim=True)
            denominator = (norm + 1e-7).bfloat16()
            explicit = (current / denominator).bfloat16()
        finally:
            torch_npu.npu.set_compile_mode(jit_compile=not previous)
        expected = original(value, steps, normalized)
        reproduced = original(value, steps, explicit)
        if not torch.equal(expected, reproduced):
            raise ValueError('Explicit normalization changed actual control NS output')
        raw = current.float().cpu().numpy()
        norm64 = [float(np.sqrt(np.sum(np.square(item, dtype=np.float64)))) for item in raw]
        RECORDS.append({
            'label': 'main1_expert_down', 'explicit_normalization_exact': True,
            'norm_fp32': norm.cpu().reshape(-1).tolist(), 'norm_fp64': norm64,
            'denominator_bf16': denominator.float().cpu().reshape(-1).tolist(),
            'pre_norm': describe(current), 'control_normalized': describe(explicit),
            'control_ns_hp_layout': describe(expected),
            'control_ns_mf_layout': describe(expected.mT.contiguous().bfloat16()),
            'pre_norm_by_expert': [describe(item) for item in current],
            'normalized_by_expert': [describe(item) for item in explicit],
            'ns_by_expert': [describe(item) for item in expected.mT],
        })
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
    os.environ['P12_NORM219_FACTOR'] = 'both'
    hp_norm219_factor.install()
    router_ns_dvm_factor.install()
    muon_ns_eager_factor.install()
    muon_q_native_tile_factor.install()
    muon_expert_bmm_eager_factor.install()
    muon_dense_adjacent_factor.install()
    install_trace()
    policy = muon_broadcast_fp32_factor.install()
    muon_module.compute_muon_slice_scale = fp32_archive_scale
    state = STATE / 'hp-jte213-fc1norm-qupboth-state329'
    targets = {'main1_expert_down': 'model.layers.1.mlp.experts.down_proj'}
    checks = []
    try:
        for label, name in targets.items():
            weight = np.load(state / 'state-328' / (label + '-weight.npy'))
            momentum = np.load(state / 'state-328' / (label + '-muon_m.npy'))
            gradient = np.load(state / 'gradient-329' / (label + '-postclip.npy'))
            gradient = np.ascontiguousarray(gradient.reshape(8, 1536, 5120).transpose(0, 2, 1))
            parameter = torch.nn.Parameter(torch.from_numpy(weight.copy()).npu())
            parameter.model_name = name
            parameter.grad = torch.from_numpy(gradient).npu()
            optimizer = AllMeasuredCastMuon([parameter], lr=1e-4, weight_decay=.1, momentum=.95,
                                            nesterov=True, ns_steps=5, ns_variant='legacy', ns_epsilon=1e-7,
                                            matched_adamw_rms=.2, apply_lr_in_update=True)
            optimizer.state[parameter]['momentum_buffer'] = torch.from_numpy(momentum).npu()
            for group in optimizer.param_groups:
                group['step'] = 328
            optimizer.step()
            actual = parameter.detach().cpu().numpy()
            actual_m = optimizer.state[parameter]['momentum_buffer'].cpu().numpy()
            expected = np.load(state / 'state-329' / (label + '-weight.npy'))
            expected_m = np.load(state / 'state-329' / (label + '-muon_m.npy'))
            check = {'label': label, 'weight_exact_training': np.array_equal(actual, expected),
                     'momentum_exact_training': np.array_equal(actual_m, expected_m),
                     'weight_sha256': digest(actual), 'expected_weight_sha256': digest(expected),
                     'momentum_sha256': digest(actual_m), 'optimizer_step': optimizer.param_groups[0]['step']}
            mf_weight = np.load(STATE / 'mf-frontier329-state/state-329/main1_expert_down-weight.npy')
            mf_weight = np.ascontiguousarray(mf_weight.reshape(8, 1536, 5120).transpose(0, 2, 1))
            check['weight_exact_mf_training'] = np.array_equal(actual, mf_weight)
            check['mf_weight_sha256'] = digest(mf_weight)
            checks.append(check)
            print(json.dumps(check), flush=True)
        report = {'complete': True, 'diagnostic_only': True, 'checks': checks, 'records': RECORDS,
                  'mode': 'control', 'broadcast_policy': policy, 'epsilon_policy': epsilon_policy, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'warning': 'Only full weight and momentum fidelity establishes a replay of actual training.'}
        (ROOT / 'evidence/frontier329-hp-control-replay.json').write_text(json.dumps(report, indent=2))
        if not all(c['weight_exact_training'] and c['momentum_exact_training'] for c in checks):
            raise ValueError('Replay did not reproduce the actual HP update')
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
