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

"""Replay actual306 HP Muon updates and expose boundaries only after full-state fidelity."""
import hashlib
import json
import math
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
import hp_q_up258_factor
from mindspore272_stat_hash import native_dump_sha1

ROOT = Path('/alignment/phase12')
STATE = Path('/state/jt-recipe-20261014')
MODE = os.environ.get('P12_FRONTIER306_REPLAY_SCOPE', 'control')
OUTPUT = STATE / ('hp-frontier306-' + MODE + '-replay')
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


def precision(raw: np.ndarray, hp: torch.Tensor, mf: torch.Tensor, norm64: float) -> dict:
    """Compare all represented outputs with the same formula evaluated in FP64."""
    first = hp.float().cpu().numpy().reshape(-1)
    second = mf.float().cpu().numpy().reshape(-1)
    values = raw.reshape(-1)
    counts = {'different': 0, 'hp_closer': 0, 'mf_closer': 0, 'equal_distance': 0}
    sums = {'hp': [], 'mf': []}
    maxima = {'hp': 0., 'mf': 0.}
    for start in range(0, values.size, 1_048_576):
        left = first[start:start + 1_048_576]
        right = second[start:start + 1_048_576]
        exact = values[start:start + 1_048_576].astype(np.float64) / (norm64 + hp_ns_epsilon229_factor.EPSILON)
        delta_hp, delta_mf = np.abs(left.astype(np.float64) - exact), np.abs(right.astype(np.float64) - exact)
        different = left.view(np.uint32) != right.view(np.uint32)
        counts['different'] += int(np.count_nonzero(different))
        counts['hp_closer'] += int(np.count_nonzero(different & (delta_hp < delta_mf)))
        counts['mf_closer'] += int(np.count_nonzero(different & (delta_mf < delta_hp)))
        counts['equal_distance'] += int(np.count_nonzero(different & (delta_mf == delta_hp)))
        for name, error in [('hp', delta_hp), ('mf', delta_mf)]:
            sums[name].append(float(error.sum()))
            maxima[name] = max(maxima[name], float(error.max()))
    return {'elements': int(values.size), **counts, 'norm_fp64': norm64,
            'epsilon_same_archive_value': hp_ns_epsilon229_factor.EPSILON,
            'all_element_mean_abs': {name: math.fsum(items) / values.size for name, items in sums.items()},
            'all_element_max_abs': maxima}


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


def install_trace() -> None:
    """Compare complete live partitions without saving raw arrays or injecting reference values."""
    original = muon_jit_probe.graph_jit_ns
    count = 0

    def traced(value, steps, normalized=None):
        nonlocal count
        if normalized is not None or tuple(value.shape) not in [(16384, 1536), (8192, 1536)]:
            raise ValueError('Unexpected Q-up NS boundary')
        piece = count
        count += 1
        current = value.mT
        norm = current.norm(dim=(-2, -1), keepdim=True)
        denominator = norm + hp_ns_epsilon229_factor.EPSILON
        ordinary = (current / denominator).bfloat16()
        independent, scalars = hp_q_up258_factor.normalized_q_up(value)
        control_input = independent if piece == 0 else ordinary
        expected = original(value, steps, normalized)
        reproduced = original(value, steps, control_input)
        exact = torch.equal(expected, reproduced)
        if not exact:
            raise ValueError('Explicit normalization changed the actual control NS output')
        candidate = original(value, steps, independent)
        raw = current.float().cpu().numpy()
        norm64 = float(np.sqrt(np.sum(np.square(raw, dtype=np.float64))))
        RECORDS.append({
            'label': 'main0_q_up', 'piece': piece, 'explicit_normalization_exact': exact,
            'ordinary_scalar': {'norm': float(norm.cpu()), 'denominator': float(denominator.cpu())},
            'pre_norm': describe(current), 'control_normalized': describe(control_input),
            'candidate_normalized': describe(independent),
            'candidate_square_sum_sha1': hashlib.sha1(np.asarray(scalars['square_sum'], dtype='<f4').tobytes()).hexdigest(),
            'candidate_scalar': scalars, 'control_ns': describe(expected), 'candidate_ns': describe(candidate),
            'precision_control_vs_candidate': precision(raw, control_input, independent, norm64),
        })
        return candidate if MODE == 'both' else expected

    muon_jit_probe.graph_jit_ns = traced


def main():
    if MODE not in ['control', 'both']:
        raise ValueError('Unknown replay scope')
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
    os.environ['P12_QUP258_SCOPE'] = 'nope'
    hp_q_up258_factor.install()
    install_trace()
    policy = muon_broadcast_fp32_factor.install()
    muon_module.compute_muon_slice_scale = fp32_archive_scale
    state = STATE / 'hp-jte213-fc1norm-state306'
    targets = {'main0_q_up': 'model.layers.0.self_attn.q_b_proj.weight'}
    checks = []
    try:
        for label, name in targets.items():
            weight = np.load(state / 'state-305' / (label + '-weight.npy'))
            momentum = np.load(state / 'state-305' / (label + '-muon_m.npy'))
            gradient = np.load(state / 'gradient-306' / (label + '-postclip.npy'))
            parameter = torch.nn.Parameter(torch.from_numpy(weight.copy()).npu())
            parameter.model_name = name
            parameter.grad = torch.from_numpy(gradient).npu()
            optimizer = AllMeasuredCastMuon([parameter], lr=1e-4, weight_decay=.1, momentum=.95,
                                            nesterov=True, ns_steps=5, ns_variant='legacy', ns_epsilon=1e-7,
                                            matched_adamw_rms=.2, apply_lr_in_update=True, ns_transform_fn=transform)
            optimizer.state[parameter]['momentum_buffer'] = torch.from_numpy(momentum).npu()
            for group in optimizer.param_groups:
                group['step'] = 305
            optimizer.step()
            actual = parameter.detach().cpu().numpy()
            actual_m = optimizer.state[parameter]['momentum_buffer'].cpu().numpy()
            expected = np.load(state / 'state-306' / (label + '-weight.npy'))
            expected_m = np.load(state / 'state-306' / (label + '-muon_m.npy'))
            check = {'label': label, 'weight_exact_training': np.array_equal(actual, expected),
                     'momentum_exact_training': np.array_equal(actual_m, expected_m),
                     'weight_sha256': digest(actual), 'expected_weight_sha256': digest(expected),
                     'momentum_sha256': digest(actual_m), 'optimizer_step': optimizer.param_groups[0]['step']}
            mf_weight = np.load(STATE / 'mf-frontier306-state/state-306/main0_q_up-weight.npy')
            check['weight_exact_mf_training'] = np.array_equal(actual, mf_weight)
            check['mf_weight_sha256'] = digest(mf_weight)
            checks.append(check)
            print(json.dumps(check), flush=True)
        report = {'complete': True, 'diagnostic_only': True, 'checks': checks, 'records': RECORDS,
                  'mode': MODE, 'broadcast_policy': policy, 'epsilon_policy': epsilon_policy, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'warning': 'Only full weight and momentum fidelity establishes a replay of actual training.'}
        (ROOT / ('evidence/frontier306-hp-' + MODE + '-replay.json')).write_text(json.dumps(report, indent=2))
        field = 'weight_exact_mf_training' if MODE == 'both' else 'weight_exact_training'
        if not all(c[field] and c['momentum_exact_training'] for c in checks):
            raise ValueError('Replay did not reproduce the actual HP update')
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
