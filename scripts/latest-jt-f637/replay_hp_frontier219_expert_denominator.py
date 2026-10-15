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

"""Replay actual219 HP Muon updates and expose boundaries only after full-state fidelity."""
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

ROOT = Path('/alignment/phase12')
STATE = Path('/state/jt-recipe-20261014')
OUTPUT = STATE / 'hp-frontier219-expert-denominator-replay'
RECORDS = []


def digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def save(directory, name, value):
    array = value.detach().float().cpu().numpy()
    np.save(directory / (name + '.npy'), array, allow_pickle=False)
    return {'name': name, 'shape': list(array.shape), 'sha256': digest(array)}


def transform(name, value):
    if name.endswith('mlp.experts.gate_up_proj'):
        def restore(updates, output):
            output.copy_(torch.cat(updates, dim=1))
        return NSInputTransform(list(value.chunk(2, dim=1)), restore)
    return None


def install_trace():
    original = muon_jit_probe.graph_jit_ns
    piece = 0

    def traced(value, steps, normalized=None):
        nonlocal piece
        if value.ndim != 3:
            return original(value, steps, normalized)
        index = piece
        piece += 1
        previous = torch_npu.npu.is_jit_compile_false()
        torch_npu.npu.set_compile_mode(jit_compile=False)
        try:
            current = value.mT if value.shape[-2] > value.shape[-1] else value
            norm_fp32 = current.float().norm(dim=(-2, -1), keepdim=True)
            denominator = (norm_fp32 + 1e-7).bfloat16()
            candidate = (current / denominator).bfloat16()
            observed = current.float().cpu().numpy()
            raw64 = [float(np.sqrt(np.sum(np.square(row, dtype=np.float64)))) for row in observed]
            records = {'piece': index, 'norm_fp64': raw64, 'norm_fp32': norm_fp32.cpu().reshape(-1).tolist(),
                       'denominator_candidate': denominator.float().cpu().reshape(-1).tolist()}
        finally:
            torch_npu.npu.set_compile_mode(jit_compile=not previous)
        result = original(value, steps, candidate)
        directory = OUTPUT / ('expert-' + str(index))
        directory.mkdir()
        records['arrays'] = [save(directory, 'normalized', candidate), save(directory, 'ns_output', result)]
        RECORDS.append(records)
        return result

    muon_jit_probe.graph_jit_ns = traced


def main():
    OUTPUT.mkdir(exist_ok=False)
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    enable_full_determinism(42, close_matmul_k_shift=False)
    os.environ['CLOSE_MATMUL_K_SHIFT'] = '0'
    os.environ['P12_NS_EAGER_SCOPE'] = 'all'
    dist.init_process_group('hccl')
    router_ns_dvm_factor.install()
    muon_ns_eager_factor.install()
    muon_q_native_tile_factor.install()
    muon_expert_bmm_eager_factor.install()
    muon_dense_adjacent_factor.install()
    install_trace()
    policy = muon_broadcast_fp32_factor.install()
    muon_module.compute_muon_slice_scale = fp32_archive_scale
    state = STATE / 'hp-jtf637-packed217-state219'
    targets = {'expert': 'model.layers.1.mlp.experts.gate_up_proj'}
    checks = []
    try:
        for label, name in targets.items():
            weight = np.load(state / 'state-218' / (label + '-weight.npy'))
            momentum = np.load(state / 'state-218' / (label + '-muon_m.npy'))
            gradient = np.load(state / 'gradient-219' / (label + '-postclip.npy'))
            if label == 'expert':
                gradient = np.ascontiguousarray(gradient.reshape(8, 5120, 3072).transpose(0, 2, 1))
            parameter = torch.nn.Parameter(torch.from_numpy(weight.copy()).npu())
            parameter.model_name = name
            parameter.grad = torch.from_numpy(gradient).npu()
            optimizer = AllMeasuredCastMuon([parameter], lr=1e-4, weight_decay=.1, momentum=.95,
                                            nesterov=True, ns_steps=5, ns_variant='legacy', ns_epsilon=1e-7,
                                            matched_adamw_rms=.2, apply_lr_in_update=True, ns_transform_fn=transform)
            optimizer.state[parameter]['momentum_buffer'] = torch.from_numpy(momentum).npu()
            for group in optimizer.param_groups:
                group['step'] = 218
            optimizer.step()
            actual = parameter.detach().cpu().numpy()
            actual_m = optimizer.state[parameter]['momentum_buffer'].cpu().numpy()
            expected = np.ascontiguousarray(np.load(STATE / 'mf-frontier219-state/state-219/expert-weight.npy').reshape(8, 5120, 3072).transpose(0, 2, 1))
            expected_m = np.load(state / 'state-219' / (label + '-muon_m.npy'))
            check = {'label': label, 'weight_exact_mf_training': np.array_equal(actual, expected),
                     'momentum_exact_training': np.array_equal(actual_m, expected_m),
                     'weight_sha256': digest(actual), 'expected_weight_sha256': digest(expected),
                     'momentum_sha256': digest(actual_m), 'optimizer_step': optimizer.param_groups[0]['step']}
            checks.append(check)
            print(json.dumps(check), flush=True)
        report = {'complete': True, 'diagnostic_only': True, 'checks': checks, 'records': RECORDS,
                  'broadcast_policy': policy, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'warning': 'Only full weight and momentum fidelity establishes a replay of actual training.'}
        (ROOT / 'evidence/frontier219-expert-denominator-replay.json').write_text(json.dumps(report, indent=2))
        assert all(c['weight_exact_mf_training'] and c['momentum_exact_training'] for c in checks)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
