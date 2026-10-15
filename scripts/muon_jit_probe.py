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

"""Replay measured archived MF NS kernels under HP's CANN9.2 runtime."""
import torch
import torch_npu

from muon_graph_probe import GraphArithmeticMuon, graph_ns, leaf_optimizers
from muon_legacy_composition import build_optimizer as legacy_builder


def graph_jit_ns(value: torch.Tensor, steps: int, normalized: torch.Tensor | None = None) -> torch.Tensor:
    """Use a contiguous Gram RHS; keep all observed scalar and cast boundaries."""
    transposed = value.shape[-2] > value.shape[-1]
    current = value.mT if transposed else value
    current = ((current / (current.norm(dim=(-2, -1), keepdim=True) + 1e-7)).bfloat16()
               if normalized is None else normalized)
    for _ in range(steps):
        gram = current @ current.mT.contiguous()
        if current.ndim == 2:
            square = (gram * 2.03125) @ gram
            polynomial = (gram.float() * -4.78125 + square.float()).bfloat16()
        else:
            square = gram @ gram
            polynomial = (gram.float() * -4.78125 + square.float() * 2.03125).bfloat16()
        product = polynomial @ current
        current = (current.float() * 3.4375 + product.float()).bfloat16()
    return (current.mT if transposed else current).float()


class GraphJitMuon(GraphArithmeticMuon):
    """Confine the compiler experiment to NS, preserving native forward kernels."""
    round_kv = False

    def _compute_batched_ns_outputs_for_tensors(self, tensor_list, ns_steps, ns_variant="asym5",
                                                ns_coefficients=None, ns_epsilon=1e-10):
        if ns_variant != "legacy" or ns_coefficients is not None or ns_epsilon != 1e-7:
            raise ValueError("Unexpected NS schedule")
        previous = torch_npu.npu.is_jit_compile_false()
        torch_npu.npu.set_compile_mode(jit_compile=True)
        try:
            outputs = []
            for value in tensor_list:
                if self.round_kv and tuple(value.shape) in ((512, 5120), (64, 5120)):
                    value = value.bfloat16().float()
                outputs.append(graph_jit_ns(value, ns_steps))
            return outputs
        finally:
            torch_npu.npu.set_compile_mode(jit_compile=not previous)


class GraphJitRoundedKvMuon(GraphJitMuon):
    """Add only the independently demonstrated KV/RoPE pre-normalization cast."""
    round_kv = True


class GraphRoundedKvMuon(GraphArithmeticMuon):
    """Keep the eager operator path while isolating the KV/RoPE cast boundary."""
    def _compute_batched_ns_outputs_for_tensors(self, tensor_list, ns_steps, ns_variant="asym5",
                                                ns_coefficients=None, ns_epsilon=1e-10):
        if ns_variant != "legacy" or ns_coefficients is not None or ns_epsilon != 1e-7:
            raise ValueError("Unexpected NS schedule")
        return [graph_ns(value.bfloat16().float()
                         if tuple(value.shape) in ((512, 5120), (64, 5120)) else value, ns_steps)
                for value in tensor_list]


def build_optimizer(*, model: torch.nn.Module, use_jit: bool, **kwargs):
    """Add one measured NS variant to the archived-scale layout/graph baseline."""
    optimizer = legacy_builder(model=model, graph_arithmetic=True, **kwargs)
    count = 0
    for leaf in leaf_optimizers(optimizer):
        if type(leaf) is GraphArithmeticMuon:
            leaf.__class__ = GraphJitRoundedKvMuon if use_jit else GraphRoundedKvMuon
            count += 1
    if count != 1:
        raise ValueError(f"Expected one GraphArithmeticMuon, found {count}")
    print("[P12-NS-KERNEL] round_kv=True use_jit=", use_jit, flush=True)
    return optimizer
