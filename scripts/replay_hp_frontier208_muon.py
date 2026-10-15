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

"""Replay actual208 HP Muon updates and expose boundaries only after full-state fidelity."""
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

ROOT = Path('/alignment/phase12')
STATE = Path('/state/jt-recipe-20261014')
OUTPUT = STATE / 'hp-frontier208-muon-replay'
RECORDS = []


def digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def save(directory, name, value):
    array = value.detach().float().cpu().numpy()
    np.save(directory / (name + '.npy'), array, allow_pickle=False)
    return {'name': name, 'shape': list(array.shape), 'sha256': digest(array)}


def transform(name, value):
    if name.endswith('linear_qkv.weight'):
        def restore(updates, output):
            output.copy_(torch.cat(updates, dim=0))
        return NSInputTransform(list(value.split((1536, 512, 64), dim=0)), restore)
    return None


def install_trace():
    original = muon_jit_probe.graph_jit_ns

    def traced(value, steps, normalized=None):
        label = {(1536, 5120): 'q', (8, 5120): 'router'}.get(tuple(value.shape))
        if label is None:
            return original(value, steps, normalized)
        directory = OUTPUT / label
        directory.mkdir(exist_ok=False)
        if steps != 5 or normalized is None:
            raise ValueError('Expected actual five-step normalized control inputs')
        rows = [save(directory, 'pre_ns', value), save(directory, 'normalized', normalized)]
        current = normalized
        for index in range(1, steps + 1):
            gram = current @ current.mT.contiguous()
            square = (gram * 2.03125) @ gram
            polynomial = (gram.float() * -4.78125 + square.float()).bfloat16()
            product = polynomial @ current
            current = (current.float() * 3.4375 + product.float()).bfloat16()
            for name, tensor in [('gram', gram), ('poly', polynomial), ('product', product), ('stage', current)]:
                rows.append(save(directory, name + '-' + str(index), tensor))
        result = current.float()
        expected = original(value, steps, normalized)
        exact = torch.equal(result, expected)
        if not exact:
            raise ValueError('Observer changed actual NS output')
        RECORDS.append({'label': label, 'trace_exact_original_ns': exact, 'arrays': rows})
        return expected

    muon_jit_probe.graph_jit_ns = traced


def main():
    OUTPUT.mkdir(exist_ok=False)
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    enable_full_determinism(42)
    os.environ['CLOSE_MATMUL_K_SHIFT'] = '0'
    os.environ['P12_NS_EAGER_SCOPE'] = 'all'
    dist.init_process_group('hccl')
    install_trace()
    router_ns_dvm_factor.install()
    muon_ns_eager_factor.install()
    muon_q_native_tile_factor.install()
    muon_dense_adjacent_factor.install()
    policy = muon_broadcast_fp32_factor.install()
    muon_module.compute_muon_slice_scale = fp32_archive_scale
    state = STATE / 'hp-jtf637-headrouter-state208'
    targets = {'qkv': 'model.layers.0.self_attn.linear_qkv.weight', 'router': 'model.layers.1.mlp.gate.weight'}
    checks = []
    try:
        for label, name in targets.items():
            weight = np.load(state / 'state-207' / (label + '-weight.npy'))
            momentum = np.load(state / 'state-207' / (label + '-muon_m.npy'))
            if label == 'qkv':
                gradient = np.concatenate([np.load(state / 'gradient-208' / (part + '-postclip.npy'))
                                           for part in ['q', 'kv']], axis=0)
            else:
                gradient = np.load(state / 'gradient-208/router-postclip.npy')
            parameter = torch.nn.Parameter(torch.from_numpy(weight.copy()).npu())
            parameter.model_name = name
            parameter.grad = torch.from_numpy(gradient).npu()
            optimizer = AllMeasuredCastMuon([parameter], lr=1e-4, weight_decay=.1, momentum=.95,
                                            nesterov=True, ns_steps=5, ns_variant='legacy', ns_epsilon=1e-7,
                                            matched_adamw_rms=.2, apply_lr_in_update=True, ns_transform_fn=transform)
            optimizer.state[parameter]['momentum_buffer'] = torch.from_numpy(momentum).npu()
            for group in optimizer.param_groups:
                group['step'] = 207
            optimizer.step()
            actual = parameter.detach().cpu().numpy()
            actual_m = optimizer.state[parameter]['momentum_buffer'].cpu().numpy()
            expected = np.load(state / 'state-208' / (label + '-weight.npy'))
            expected_m = np.load(state / 'state-208' / (label + '-muon_m.npy'))
            check = {'label': label, 'weight_exact_training': np.array_equal(actual, expected),
                     'momentum_exact_training': np.array_equal(actual_m, expected_m),
                     'weight_sha256': digest(actual), 'expected_weight_sha256': digest(expected),
                     'momentum_sha256': digest(actual_m), 'optimizer_step': optimizer.param_groups[0]['step']}
            checks.append(check)
            print(json.dumps(check), flush=True)
        report = {'complete': True, 'diagnostic_only': True, 'checks': checks, 'records': RECORDS,
                  'broadcast_policy': policy, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'warning': 'Only full weight and momentum fidelity establishes a replay of actual training.'}
        (ROOT / 'evidence/frontier208-hp-muon-replay.json').write_text(json.dumps(report, indent=2))
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
