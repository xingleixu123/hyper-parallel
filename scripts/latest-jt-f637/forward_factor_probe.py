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

"""Incremental first-forward diagnostics with source state and optimizer unchanged."""
import torch
from torch import nn
from torch.nn import functional as functional

from one_factor_probe import PrecisionFSDP, GlobalMTPModel, configure_factor as base_configure


class SplitQKVLinear(nn.Linear):
    """Keep the fused parameter but evaluate Q and KV projections independently."""
    def forward(self, inputs):
        if self.bias is not None or self.weight.shape[0] != 2112:
            raise ValueError("Unexpected diagnostic projection shape")
        query, key_value = self.weight.split((1536, 576), dim=0)
        return torch.cat((functional.linear(inputs, query), functional.linear(inputs, key_value)), dim=-1)


def configure_factor(model, factor):
    """Add one forward projection factor to the previously measured norm baseline."""
    if factor not in ("norm-qkv-forward", "qkv-forward"):
        return base_configure(model, factor)
    if factor == "norm-qkv-forward":
        base_configure(model, "norm-fp32")
    changed = []
    for name, module in model.named_modules():
        if name.endswith(".linear_qkv"):
            if type(module) is not nn.Linear:
                raise TypeError(f"Unexpected module {name}: {type(module)}")
            module.__class__ = SplitQKVLinear
            changed.append(name)
    if len(changed) != 3:
        raise ValueError(f"Expected three projection units, found {changed}")
    print("[P12-FORWARD-FACTOR]", factor, changed, flush=True)
