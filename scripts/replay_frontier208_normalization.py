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

"""Isolate live reduction and sqrt routes against fully faithful native208 boundaries."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch_npu

from hyper_parallel.trainer.runtime.random import enable_full_determinism
from muon_jit_probe import graph_jit_ns
from muon_q_norm_probe import half_tree
from muon_lr_fp32_scale import fp32_archive_scale
from muon_q_native_tile_factor import normalized_q
from observe_hp_preclip_norm_adjacent import dvm_square_sum
from compare_frontier208_state import stats

ROOT = Path('/alignment/phase12')
STATE = Path('/state/jt-recipe-20261014')


def q_sum(value, adjacent):
    flat = value.reshape(-1)
    padded = value.new_zeros(1127 * 6984)
    padded[:flat.numel()] = flat * flat
    lanes = half_tree(padded.reshape(1127, 291, 24).transpose(-1, -2))
    if adjacent:
        buffer = value.new_zeros((1127, 32))
        buffer[:, :24] = lanes
        while buffer.shape[-1] > 1:
            buffer = buffer[..., ::2] + buffer[..., 1::2]
        partials = buffer[..., 0].cpu().numpy()
    else:
        partials = half_tree(lanes).cpu().numpy()
    accumulators = np.zeros(40, dtype=np.float32)
    for index, partial in enumerate(partials):
        core = index % 40
        accumulators[core] = np.float32(accumulators[core] + partial)
    total = np.float32(0)
    for partial in accumulators:
        total = np.float32(total + partial)
    return total


def main():
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    enable_full_determinism(42)
    rows = json.loads((ROOT / 'runs/mf-frontier208-ns-train/NS_BOUNDARIES-208.json').read_text())
    if not json.loads((ROOT / 'evidence/frontier208-native-ns-audit.json').read_text())['usable_as_native_training']:
        raise ValueError('Native observations are not faithful')
    native_rows = {row['prefix']: row for row in rows}
    records, correctness = [], []
    for label in ['q', 'router']:
        source = np.load(STATE / 'hp-frontier208-muon-replay' / label / 'pre_ns.npy')
        native_file = STATE / 'mf-frontier208-ns-operators' / native_rows[label + '-normalized']['file']
        native = np.load(native_file)
        if hashlib.sha256(native.tobytes()).hexdigest() != native_rows[label + '-normalized']['sha256']:
            raise ValueError('Native normalized array changed')
        baseline = np.load(STATE / 'hp-frontier208-muon-replay' / label / 'normalized.npy')
        before = np.load(STATE / 'mf-frontier208-state/state-207' / (label + '-weight.npy'))
        after = np.load(STATE / 'mf-frontier208-state/state-208' / (label + '-weight.npy'))
        hp_after = np.load(STATE / 'hp-jtf637-headrouter-state208/state-208' /
                           ('qkv-weight.npy' if label == 'q' else 'router-weight.npy'))
        if label == 'q':
            hp_after = hp_after[:1536]
        value = torch.from_numpy(source).npu()
        original_weight = torch.from_numpy(before).npu()
        variants = {}
        for reduction in (['half', 'adjacent'] if label == 'q' else ['dvm']):
            total = q_sum(value, reduction == 'adjacent') if label == 'q' else dvm_square_sum(value)
            for sqrt_route in ['cpu', 'npu']:
                norm = (torch.tensor(float(np.float32(np.sqrt(total))), dtype=torch.float32, device=value.device)
                        if sqrt_route == 'cpu' else torch.sqrt(torch.tensor(float(total), device=value.device)))
                epsilon = 1.0011717677116394e-7 if label == 'q' else 1e-7
                denominator = norm + epsilon
                normalized = (value / denominator).bfloat16()
                variants[reduction + '-' + sqrt_route] = (normalized, float(total), float(norm.cpu()), float(denominator.cpu()))
        control_name = 'half-cpu' if label == 'q' else 'dvm-cpu'
        if not np.array_equal(variants[control_name][0].float().cpu().numpy(), baseline):
            raise ValueError('Live baseline normalization does not reproduce')
        if label == 'q' and not torch.equal(normalized_q(value), variants[control_name][0]):
            raise ValueError('Original Q helper changed')
        variants['teacher-normalized-diagnostic'] = (torch.from_numpy(native).npu().bfloat16(), None, None, None)
        for name, (normalized, total, norm, denominator) in variants.items():
            ns_output = graph_jit_ns(value, 5, normalized)
            scale = fp32_archive_scale(ns_output, .2)
            scale *= 1e-4
            result = original_weight * (1 - 1e-4 * .1) - ns_output * scale
            actual = result.cpu().numpy()
            if name == control_name and not np.array_equal(actual, hp_after):
                raise ValueError('Explicit update replay does not reproduce actual HP weight')
            records.append({'label': label, 'variant': name, 'square_sum': total, 'norm': norm,
                            'denominator': denominator, 'teacher_values_used': name.startswith('teacher'),
                            'normalized': stats(normalized.float().cpu().numpy(), native), 'weight': stats(actual, after)})
        reference = source.astype(np.float64)
        true_norm = np.linalg.norm(reference)
        reference /= true_norm + 1e-7
        mask = baseline != native
        hp_error, mf_error = np.abs(baseline[mask].astype(np.float64) - reference[mask]), np.abs(native[mask].astype(np.float64) - reference[mask])
        correctness.append({'label': label, 'scope': 'Only normalized BF16 coordinates where actual HP and MF differ; FP64 unrounded NS input reference.',
                            'different': int(mask.sum()), 'mf_closer': int((mf_error < hp_error).sum()),
                            'hp_closer': int((hp_error < mf_error).sum()), 'tied': int((hp_error == mf_error).sum()),
                            'fp64_norm': float(true_norm), 'mf_max_abs_error': float(mf_error.max()),
                            'hp_max_abs_error': float(hp_error.max())})
    report = {'complete': True, 'diagnostic_only': True, 'records': records, 'fp64_correctness': correctness,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (ROOT / 'evidence/frontier208-normalization-replay.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'records': [{k: v for k, v in row.items() if k not in ['normalized', 'weight']} | {
        'normalized_different': row['normalized']['different'], 'weight_different': row['weight']['different']}
        for row in records], 'fp64_correctness': correctness}))


if __name__ == '__main__':
    main()
