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

"""Test the measured Q-down reduction schedule using live optimizer values."""
import numpy as np
import torch
import torch_npu

from muon_jit_probe import GraphJitRoundedKvMuon, graph_jit_ns


def half_tree(values: torch.Tensor) -> torch.Tensor:
    """Use explicit FP32 additions to preserve the candidate reduction order."""
    work = values
    while work.shape[-1] > 1:
        size = work.shape[-1]
        half = size // 2
        reduced = work[..., :half] + work[..., half:2 * half]
        if size % 2:
            reduced[..., 0] = reduced[..., 0] + work[..., -1]
        work = reduced
    return work[..., 0]


def q_normalized(value: torch.Tensor) -> torch.Tensor:
    """Transfer only 1450 live partial sums; no saved reference values enter this path."""
    if tuple(value.shape) != (1536, 5120) or value.dtype != torch.float32:
        raise ValueError("This measured reduction applies only to FP32 Q-down")
    flat = value.reshape(-1)
    padded = value.new_zeros(1450 * 5424)
    padded[:flat.numel()] = flat * flat
    tiled = padded.reshape(1450, 113, 48).transpose(-1, -2)
    partials = half_tree(half_tree(tiled)).cpu().numpy()
    total = np.float32(0)
    for partial in partials:
        total = np.float32(total + partial)
    norm = torch.tensor(np.sqrt(total), device=value.device)
    denominator = norm + 1.0011717677116394e-7
    return (value / denominator).bfloat16()


class GraphQNormMuon(GraphJitRoundedKvMuon):
    """Add the Q-down reduction schedule to the verified KV cast/JIT experiment."""
    def _compute_batched_ns_outputs_for_tensors(self, tensor_list, ns_steps, ns_variant="asym5",
                                                ns_coefficients=None, ns_epsilon=1e-10):
        if ns_variant != "legacy" or ns_coefficients is not None or ns_epsilon != 1e-7:
            raise ValueError("Unexpected NS schedule")
        previous = torch_npu.npu.is_jit_compile_false()
        torch_npu.npu.set_compile_mode(jit_compile=True)
        try:
            outputs = []
            for value in tensor_list:
                if tuple(value.shape) == (1536, 5120):
                    outputs.append(graph_jit_ns(value, ns_steps, normalized=q_normalized(value)))
                else:
                    outputs.extend(super()._compute_batched_ns_outputs_for_tensors(
                        [value], ns_steps, ns_variant, ns_coefficients, ns_epsilon))
            return outputs
        finally:
            torch_npu.npu.set_compile_mode(jit_compile=not previous)
