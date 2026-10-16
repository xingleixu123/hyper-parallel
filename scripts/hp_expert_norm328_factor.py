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

"""Reproduce the measured archived expert-down normalization from live optimizer inputs."""
import functools
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch_npu

import capture_hp_native
import muon_jit_probe
from muon_q_norm_probe import half_tree


def normalized_expert_down(value: torch.Tensor, epsilon: float = 1e-7) -> tuple[torch.Tensor, dict]:
    """Apply the validated 10240-element tiles and five-vector-core expert reduction."""
    if tuple(value.shape) != (8, 5120, 1536) or value.dtype != torch.bfloat16:
        raise ValueError('Expected the measured BF16 expert-down NS input')
    previous = torch_npu.npu.is_jit_compile_false()
    torch_npu.npu.set_compile_mode(jit_compile=False)
    try:
        current = value.mT
        tiled = current.float().square().reshape(8, 768, 160, 64).transpose(-1, -2)
        lanes = half_tree(tiled)
        while lanes.shape[-1] > 1:
            lanes = lanes[..., ::2] + lanes[..., 1::2]
        partials = lanes[..., 0].cpu().numpy()
        accumulators = np.zeros((8, 5), dtype=np.float32)
        for index in range(768):
            core = index % 5
            accumulators[:, core] = accumulators[:, core] + partials[:, index]
        total = np.zeros(8, dtype=np.float32)
        for core in range(5):
            total = total + accumulators[:, core]
        norm = torch.from_numpy(total).to(value.device).sqrt().reshape(8, 1, 1)
        denominator = (norm + epsilon).bfloat16()
        result = (current / denominator).bfloat16()
        return result, {
            'squared_sum': total.tolist(), 'norm': norm.cpu().reshape(-1).tolist(),
            'denominator_bf16': denominator.float().cpu().reshape(-1).tolist(), 'epsilon': epsilon,
        }
    finally:
        torch_npu.npu.set_compile_mode(jit_compile=not previous)


def install() -> dict:
    """Install only the measured shape; retain the existing epsilon for a single-factor ablation."""
    original = muon_jit_probe.graph_jit_ns
    counts = {}
    policy = {
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'shape': [8, 5120, 1536], 'input_dtype': 'bfloat16',
        'tile_elements': 10240, 'tile_rows': 160, 'lanes': 64,
        'tiles_per_expert': 768, 'vector_cores': 40, 'cores_per_expert': 5,
        'reduction': 'half-tree rows, adjacent lanes, five strided cores per expert, sequential core join',
        'sqrt': 'device FP32', 'epsilon': 1e-7,
        'teacher_values_used': False, 'step_specific_behavior': False,
        'classification': 'Archived compiled-kernel compatibility experiment, not a default correctness claim.',
    }

    @functools.wraps(original)
    def changed(value, steps, normalized=None):
        if tuple(value.shape) == (8, 5120, 1536) and normalized is None:
            normalized, _ = normalized_expert_down(value)
            key = str(tuple(value.shape))
            counts[key] = counts.get(key, 0) + 1
        return original(value, steps, normalized)

    muon_jit_probe.graph_jit_ns = changed
    old_begin = capture_hp_native.Observer.on_train_begin
    old_end = capture_hp_native.Observer.on_step_end

    def begin(self, state, **kwargs):
        old_begin(self, state, **kwargs)
        (self.output / 'EXPERT328_FACTOR.json').write_text(json.dumps(policy, indent=2))

    def end(self, state, **kwargs):
        old_end(self, state, **kwargs)
        with (self.output / 'EXPERT328_CALLS.jsonl').open('a') as stream:
            stream.write(json.dumps({'step': int(state.global_step), 'cumulative_calls': dict(counts)}) + '\n')

    capture_hp_native.Observer.on_train_begin = begin
    capture_hp_native.Observer.on_step_end = end
    return policy
