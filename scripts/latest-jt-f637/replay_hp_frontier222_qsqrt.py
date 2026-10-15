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

"""Replay actual222 HP Muon updates and expose boundaries only after full-state fidelity."""
import hashlib
import inspect
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
import muon_q_adjacent_factor

ROOT = Path('/alignment/phase12')
STATE = Path('/state/jt-recipe-20261014')
VARIANT = os.environ.get('P12_Q222_VARIANT', 'control')
OUTPUT = STATE / ('hp-frontier222-q-' + VARIANT + '-replay')
CURRENT = None
POLICY = None
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
        if tuple(value.shape) != (1536, 5120):
            return original(value, steps, normalized)
        assert normalized is not None and steps == 5
        directory = OUTPUT / CURRENT
        directory.mkdir(exist_ok=False)
        arrays = [save(directory, 'pre_ns', value), save(directory, 'normalized', normalized)]
        result = original(value, steps, normalized)
        arrays.append(save(directory, 'ns_output', result))
        raw = value.float().cpu().numpy()
        norm64 = float(np.sqrt(np.sum(np.square(raw, dtype=np.float64))))
        RECORDS.append({'label': CURRENT, 'arrays': arrays, 'norm_fp64': norm64,
                        'scalar': muon_q_adjacent_factor.P12_Q222_SCALARS[-1]})
        return result

    muon_jit_probe.graph_jit_ns = traced


def install_q_factor():
    global POLICY
    assert VARIANT in ['control', 'npu']
    original = inspect.getsource(muon_q_adjacent_factor.normalized_q)
    source = original.replace('def normalized_q(', 'def normalized_q_observed222(')
    if VARIANT == 'npu':
        old = 'np.float32(np.sqrt(total) + np.float32(1.0011717677116394e-7))'
        assert source.count(old) == 1
        source = source.replace(old, 'torch.tensor(total, device=value.device).sqrt() + 1.0011717677116394e-7')
        source = source.replace('torch.tensor(denominator, device=value.device)', 'denominator')
    source = source.replace('    CALLS += 1',
                            '    P12_Q222_SCALARS.append({"square_sum": float(total), "denominator": float(denominator)})\n'
                            '    CALLS += 1')
    muon_q_adjacent_factor.P12_Q222_SCALARS = []
    exec(compile(source, __file__ + ':observed-q-factor', 'exec'), muon_q_adjacent_factor.__dict__)
    muon_q_adjacent_factor.normalized_q = muon_q_adjacent_factor.normalized_q_observed222
    muon_q_adjacent_factor.install()
    POLICY = {'variant': VARIANT, 'original_sha256': hashlib.sha256(original.encode()).hexdigest(),
              'candidate_sha256': hashlib.sha256(source.encode()).hexdigest()}


def main():
    global CURRENT
    OUTPUT.mkdir(exist_ok=False)
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    enable_full_determinism(42, close_matmul_k_shift=False)
    os.environ['CLOSE_MATMUL_K_SHIFT'] = '0'
    os.environ['P12_NS_EAGER_SCOPE'] = 'all'
    dist.init_process_group('hccl')
    install_trace()
    install_q_factor()
    router_ns_dvm_factor.install()
    muon_ns_eager_factor.install()
    muon_q_native_tile_factor.install()
    muon_dense_adjacent_factor.install()
    policy = muon_broadcast_fp32_factor.install()
    muon_module.compute_muon_slice_scale = fp32_archive_scale
    state = STATE / 'hp-jtf637-norm219-state222'
    targets = {'main': 'model.layers.0.self_attn.linear_qkv.weight', 'mtp': 'mtp.layers.0.transformer_layer.self_attn.linear_qkv.weight'}
    checks = []
    try:
        for label, name in targets.items():
            CURRENT = label
            weight = np.load(state / 'state-221' / (label + '-weight.npy'))
            momentum = np.load(state / 'state-221' / (label + '-muon_m.npy'))
            gradient = np.concatenate([np.load(state / 'gradient-222' / (part + '_' + label + '-postclip.npy'))
                                       for part in ['q', 'kv']], axis=0)
            parameter = torch.nn.Parameter(torch.from_numpy(weight.copy()).npu())
            parameter.model_name = name
            parameter.grad = torch.from_numpy(gradient).npu()
            optimizer = AllMeasuredCastMuon([parameter], lr=1e-4, weight_decay=.1, momentum=.95,
                                            nesterov=True, ns_steps=5, ns_variant='legacy', ns_epsilon=1e-7,
                                            matched_adamw_rms=.2, apply_lr_in_update=True, ns_transform_fn=transform)
            optimizer.state[parameter]['momentum_buffer'] = torch.from_numpy(momentum).npu()
            for group in optimizer.param_groups:
                group['step'] = 221
            optimizer.step()
            actual = parameter.detach().cpu().numpy()
            actual_m = optimizer.state[parameter]['momentum_buffer'].cpu().numpy()
            expected = np.load(state / 'state-222' / (label + '-weight.npy'))
            expected_m = np.load(state / 'state-222' / (label + '-muon_m.npy'))
            mf_weight = np.concatenate([np.load(STATE / 'mf-frontier222-state/state-222' / (part + '_' + label + '-weight.npy')) for part in ['q', 'kv']], axis=0)
            check = {'label': label, 'weight_exact_training': np.array_equal(actual, expected),
                     'momentum_exact_training': np.array_equal(actual_m, expected_m),
                     'weight_exact_mf_training': np.array_equal(actual, mf_weight),
                     'mf_weight_sha256': digest(mf_weight),
                     'weight_sha256': digest(actual), 'expected_weight_sha256': digest(expected),
                     'momentum_sha256': digest(actual_m), 'optimizer_step': optimizer.param_groups[0]['step']}
            checks.append(check)
            print(json.dumps(check), flush=True)
        report = {'complete': True, 'diagnostic_only': True, 'checks': checks, 'records': RECORDS,
                  'broadcast_policy': policy, 'q_factor': POLICY, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'warning': 'Only full weight and momentum fidelity establishes a replay of actual training.'}
        (ROOT / 'evidence' / ('frontier222-q-' + VARIANT + '-replay.json')).write_text(json.dumps(report, indent=2))
        assert all(c['momentum_exact_training'] and c['weight_exact_training' if VARIANT == 'control' else 'weight_exact_mf_training'] for c in checks)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
