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

"""Opt-in single-card replay of archived MF graph Muon arithmetic boundaries."""
from typing import Any

import torch
import torch.distributed as dist

from hyper_parallel.core.optimizer.muon import Muon
from hyper_parallel.models.jt_deepseek_v3.adapter.runtime.jt_optimizer import build_optimizer as native_builder
from muon_qkv_probe import build_optimizer as qkv_builder


def local(value):
    """Read the local tensor owned by a single-card optimizer parameter."""
    return value.to_local() if hasattr(value, "to_local") else value


def graph_ns(value: torch.Tensor, steps: int) -> torch.Tensor:
    """Match observed cast boundaries; use no saved gradients or reference outputs."""
    transposed = value.shape[-2] > value.shape[-1]
    current = value.mT if transposed else value
    current = (current / (current.norm(dim=(-2, -1), keepdim=True) + 1e-7)).bfloat16()
    coeff_a, coeff_b, coeff_c = 3.4375, -4.78125, 2.03125
    for _ in range(steps):
        gram = current @ current.mT
        if current.ndim == 2:
            square = (gram * coeff_c) @ gram
            polynomial = (gram.float() * coeff_b + square.float()).bfloat16()
        else:
            square = gram @ gram
            polynomial = (gram.float() * coeff_b + square.float() * coeff_c).bfloat16()
        product = polynomial @ current
        current = (current.float() * coeff_a + product.float()).bfloat16()
    return (current.mT if transposed else current).float()


class GraphArithmeticMuon(Muon):
    """Diagnostic subclass preserving FP32 values until NS normalization completes."""
    def _update_muon_momentum(self, group, params):
        if not dist.is_initialized() or dist.get_world_size() != 1:
            raise ValueError("Graph arithmetic diagnostic supports only one rank")
        beta1, beta2 = group["momentum"]
        if beta1 != beta2 or group["ns_variant"] != "legacy" or group["ns_epsilon"] != 1e-7:
            raise ValueError("Unexpected Muon arithmetic configuration")
        updates = {}
        for parameter in params:
            if parameter.grad is None:
                continue
            gradient = local(parameter.grad)
            if "momentum_buffer" not in self.state[parameter]:
                self.state[parameter]["momentum_buffer"] = torch.zeros_like(parameter.grad)
            buffer = local(self.state[parameter]["momentum_buffer"])
            buffer.mul_(beta2).add_(gradient)
            updates[parameter] = gradient + buffer * beta1 if group["nesterov"] else buffer.clone()
        return updates

    def _compute_batched_ns_outputs_for_tensors(self, tensor_list, ns_steps, ns_variant="asym5",
                                                ns_coefficients=None, ns_epsilon=1e-10):
        if ns_variant != "legacy" or ns_coefficients is not None or ns_epsilon != 1e-7:
            raise ValueError("Unexpected NS schedule")
        # Source 2D and expert 3D expressions have distinct scalar placement.
        return [graph_ns(value, ns_steps) for value in tensor_list]


def leaf_optimizers(optimizer):
    """Visit native optimizer leaves without changing parameter ownership."""
    if hasattr(optimizer, "optimizers_dict"):
        return [leaf for item in optimizer.optimizers_dict.values() for leaf in leaf_optimizers(item)]
    if hasattr(optimizer, "chained_optimizers"):
        return [leaf for item in optimizer.chained_optimizers for leaf in leaf_optimizers(item)]
    if hasattr(optimizer, "optimizer"):
        return leaf_optimizers(optimizer.optimizer)
    return [optimizer]


def build_optimizer(*, model: torch.nn.Module, qkv_split: bool = False, **kwargs: Any) -> Any:
    """Retain native parameter groups, state, AdamW and post-update QK clipping."""
    optimizer = (qkv_builder if qkv_split else native_builder)(model=model, **kwargs)
    count = 0
    for leaf in leaf_optimizers(optimizer):
        if type(leaf) is Muon:
            leaf.__class__ = GraphArithmeticMuon
            count += 1
    if count != 1:
        raise ValueError(f"Expected exactly one native Muon leaf, found {count}")
    print("[P12-GRAPH-MUON] enabled qkv_split=", qkv_split, flush=True)
    return optimizer
