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

"""Test native attention-output / MTP projection NS reduction using actual values and static tiling only."""
import functools

import numpy as np
import torch
import torch_npu

import muon_jit_probe
from muon_q_norm_probe import half_tree


def _normalized_three(value: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Test the measured 291-lane reduction followed by adjacent pairing of 24 partials."""
    if tuple(value.shape) not in ((5120, 16384), (5120, 10240)) or value.dtype != torch.float32:
        raise ValueError('Expected the measured attention-output / MTP projection shape and dtype')
    flat = value.reshape(-1)
    tiles = (flat.numel() + 6983) // 6984
    padded = value.new_zeros(tiles * 6984)
    padded[:flat.numel()] = flat * flat
    lanes = half_tree(padded.reshape(tiles, 291, 24).transpose(-1, -2))
    adjacent = value.new_zeros((tiles, 32))
    adjacent[:, :24] = lanes
    while adjacent.shape[-1] > 1:
        adjacent = adjacent[..., ::2] + adjacent[..., 1::2]
    partials = adjacent[..., 0].cpu().numpy()
    accumulators = np.zeros(40, dtype=np.float32)
    for index, partial in enumerate(partials):
        core = index % 40
        accumulators[core] = np.float32(accumulators[core] + partial)
    total = np.float32(0)
    for partial in accumulators:
        total = np.float32(total + partial)
    denominator = torch.tensor(total, device=value.device).sqrt() + 1.0011717677116394e-7
    return (value / denominator).bfloat16(), float(total)


def normalized_three(value: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Use the eager execution mode validated by the native-kernel audit."""
    previous = torch_npu.npu.is_jit_compile_false()
    torch_npu.npu.set_compile_mode(jit_compile=False)
    try:
        return _normalized_three(value)
    finally:
        torch_npu.npu.set_compile_mode(jit_compile=not previous)


def install() -> None:
    """Change only the attention-output / MTP projection normalization within the compatibility path."""
    original = muon_jit_probe.graph_jit_ns

    @functools.wraps(original)
    def dense_norm(value, steps, normalized=None):
        if tuple(value.shape) in ((5120, 16384), (5120, 10240)) and normalized is None:
            normalized, _ = normalized_three(value)
        return original(value, steps, normalized)

    muon_jit_probe.graph_jit_ns = dense_norm
    print('[P12-THREE-NATIVE-ADJACENT-NPU-SQRT] actual live values; no recorded norm used', flush=True)
