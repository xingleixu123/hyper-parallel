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

"""Isolate dense/shared SwiGLU rounding on the measured norm baseline."""
import torch
import torch_npu

from hyper_parallel.models.jt_deepseek_v3 import modeling_jt_deepseek_v3 as jt
from one_factor_probe import PrecisionFSDP, GlobalMTPModel, configure_factor as base_configure


class FusedActivationMLP(jt.DeepseekV32MLP):
    """Preserve both projections and fuse only SiLU plus multiplication."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Round once after the fused activation, using ordinary operator autograd."""
        gate = self.gate_proj(inputs)
        up = self.up_proj(inputs)
        hidden = torch_npu.npu_swiglu(torch.cat((gate, up), dim=-1), dim=-1)
        return self.down_proj(hidden)


def configure_factor(model: torch.nn.Module, factor: str) -> None:
    """Add a single activation factor to the norm-only reference."""
    if factor != "norm-swiglu":
        return base_configure(model, factor)
    base_configure(model, "norm-fp32")
    changed = []
    for name, module in model.named_modules():
        if type(module) is jt.DeepseekV32MLP:
            module.__class__ = FusedActivationMLP
            changed.append(name)
    if len(changed) != 3:
        raise ValueError(f"Expected dense and two shared MLPs: {changed}")
    print("[P12-ACTIVATION-FACTOR]", factor, changed, flush=True)
