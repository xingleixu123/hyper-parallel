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

"""Test archived shared-expert down square-sum scheduling using only current optimizer values."""
import functools
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch_npu

import capture_hp_native
import muon_jit_probe
from muon_q_norm_probe import half_tree

EPSILON = 1.0011717677116394e-7


def normalized_shared_down(value: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """Compute the measured tile and vector-core reductions from live inputs."""
    if tuple(value.shape) not in [(5120, 3072)] or value.dtype != torch.float32:
        raise ValueError('Expected a measured 2D shared-expert down input')
    previous = torch_npu.npu.is_jit_compile_false()
    torch_npu.npu.set_compile_mode(jit_compile=False)
    try:
        current = value.mT
        tile_count = (current.numel() + 9775) // 9776
        padded = current.new_zeros(tile_count * 9776)
        padded[:current.numel()] = current.reshape(-1).square()
        lanes = half_tree(padded.reshape(tile_count, 611, 16).transpose(-1, -2))
        while lanes.shape[-1] > 1:
            lanes = lanes[..., ::2] + lanes[..., 1::2]
        partials = lanes[..., 0].cpu().numpy()
        accumulators = np.zeros(40, dtype=np.float32)
        for index, partial in enumerate(partials):
            core = index % 40
            accumulators[core] = np.float32(accumulators[core] + partial)
        total = np.float32(0)
        for partial in accumulators:
            total = np.float32(total + partial)
        norm = torch.tensor(total, device=value.device).sqrt()
        denominator = norm + EPSILON
        result = (current / denominator).bfloat16()
        scalars = {'square_sum': float(total), 'norm': float(norm.cpu()),
                   'denominator': float(denominator.cpu())}
        return result, scalars
    finally:
        torch_npu.npu.set_compile_mode(jit_compile=not previous)


def install() -> dict:
    """Install a shape-scoped experimental factor with no recorded reference values."""
    scope = os.environ.get('P12_SHARED_DOWN396_SCOPE', 'both')
    shapes = {'both': [(5120, 3072)]}
    if scope not in shapes:
        raise ValueError('Unknown shared-expert down normalization scope')
    original = muon_jit_probe.graph_jit_ns
    counts = {}
    policy = {
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope': scope, 'shape': [5120, 3072], 'tiles': 1609, 'tail_elements': 8832, 'tile_elements': 9776, 'tile_rows': 611, 'lanes': 16, 'vector_cores': 40,
        'reduction': 'half-tree rows with odd-tail fold, adjacent lanes, strided cores, sequential core join',
        'sqrt': 'device FP32', 'epsilon': EPSILON,
        'teacher_values_used': False, 'step_specific_behavior': False,
        'classification': 'Archived compiled-kernel compatibility experiment; not a product correctness claim.',
    }

    @functools.wraps(original)
    def changed(value, steps, normalized=None):
        if tuple(value.shape) in shapes[scope] and normalized is None:
            normalized, _ = normalized_shared_down(value)
            key = str(tuple(value.shape))
            counts[key] = counts.get(key, 0) + 1
        return original(value, steps, normalized)

    muon_jit_probe.graph_jit_ns = changed
    old_begin = capture_hp_native.Observer.on_train_begin
    old_end = capture_hp_native.Observer.on_step_end

    def begin(self, state, **kwargs):
        old_begin(self, state, **kwargs)
        (self.output / 'SHARED_DOWN396_FACTOR.json').write_text(json.dumps(policy, indent=2))

    def end(self, state, **kwargs):
        old_end(self, state, **kwargs)
        with (self.output / 'SHARED_DOWN396_CALLS.jsonl').open('a') as stream:
            stream.write(json.dumps({'step': int(state.global_step), 'cumulative_calls': dict(counts)}) + '\n')

    capture_hp_native.Observer.on_train_begin = begin
    capture_hp_native.Observer.on_step_end = end
    return policy
