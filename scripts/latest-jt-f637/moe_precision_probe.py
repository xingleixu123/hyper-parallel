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

"""Separate the two independently measured MoE forward rounding boundaries."""
import torch
from torch.nn import functional as functional
import torch_npu

from hyper_parallel.models.jt_deepseek_v3 import modeling_jt_deepseek_v3 as jt
from mlp_precision_probe import FusedActivationMLP, PrecisionFSDP, GlobalMTPModel, configure_factor as base_configure
from one_factor_probe import configure_factor as primitive_configure


class CombinedProjectionMLP(FusedActivationMLP):
    """Combine gate/up GEMMs while keeping parameter ownership and activation fixed."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run one combined projection and the already tested fused activation."""
        if self.gate_proj.bias is not None or self.up_proj.bias is not None:
            raise ValueError("This projection experiment requires bias-free layers")
        pair = functional.linear(inputs, torch.cat((self.gate_proj.weight, self.up_proj.weight), dim=0))
        return self.down_proj(torch_npu.npu_swiglu(pair, dim=-1))


class FP32CombineMoE(jt.JTDeepseekV3MoE):
    """Keep live probabilities and weighted expert accumulation in FP32."""

    def local_routed_forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Keep expert execution and ordering identical to the native path."""
        indices, probabilities = self.route(hidden)
        flat = hidden.reshape(-1, hidden.shape[-1])
        expert_ids = indices.reshape(-1)
        order = expert_ids.argsort()
        sources = torch.arange(flat.shape[0], device=flat.device).repeat_interleave(indices.shape[1])[order]
        counts = expert_ids.new_zeros(self.experts.num_experts).scatter_add_(
            0, expert_ids, torch.ones_like(expert_ids))
        outputs = self.experts.forward_expert_major(flat[sources], counts)
        weights = probabilities.reshape(-1)[order].unsqueeze(-1)
        combined = torch.zeros_like(flat, dtype=torch.float32).index_add_(
            0, sources, outputs.float() * weights.float())
        return combined.to(hidden.dtype).reshape(hidden.shape)


class FP32ResidualMoE(jt.JTDeepseekV3MoE):
    """Defer rounding of routed plus shared output until after the residual add."""

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Preserve native routing, combine and shared expert operators."""
        hidden = hidden_states
        if self.padding:
            hidden = torch.cat((hidden.new_zeros(1, self.padding, hidden.shape[-1]), hidden), dim=1)
        routed = self.ep_compute(hidden)[:, self.padding:]
        shared = self.shared_experts(hidden_states)
        return routed.float() + shared.float()


class FP32BothMoE(FP32CombineMoE, FP32ResidualMoE):
    """Compose the two independently tested precision boundaries."""


class RoundResidualDecoder(jt.JTDeepseekV3Decoder):
    """Restore the activation dtype after the three-term FP32 sum."""

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Use the native decoder ordering and round its final sum once."""
        return super().forward(hidden_states, *args, **kwargs).to(hidden_states.dtype)


def configure_factor(model: torch.nn.Module, factor: str) -> None:
    """Independently compare either MoE factor against norm plus fused SwiGLU."""
    if factor not in {"norm-swiglu-combine", "norm-swiglu-residual", "norm-swiglu-both", "norm-swiglu-both-router", "norm-swiglu-both-router-dense"}:
        return base_configure(model, factor)
    base_configure(model, "norm-swiglu")
    if "router" in factor:
        primitive_configure(model, "router-fp32")
    changed = []
    for name, module in model.named_modules():
        if factor.endswith("dense") and type(module) is FusedActivationMLP:
            module.__class__ = CombinedProjectionMLP
        if type(module) is jt.JTDeepseekV3MoE:
            if "both" in factor:
                module.__class__ = FP32BothMoE
                module.ep_compute = module.local_routed_forward
            elif factor.endswith("combine"):
                module.__class__ = FP32CombineMoE
                module.ep_compute = module.local_routed_forward
            else:
                module.__class__ = FP32ResidualMoE
            changed.append(name)
        if (factor.endswith("residual") or "both" in factor) and type(module) is jt.JTDeepseekV3Decoder:
            if isinstance(module.mlp, jt.JTDeepseekV3MoE):
                module.__class__ = RoundResidualDecoder
    if len(changed) != 2:
        raise ValueError(f"Expected two MoE modules: {changed}")
    print("[P12-MOE-PRECISION]", factor, changed, flush=True)
