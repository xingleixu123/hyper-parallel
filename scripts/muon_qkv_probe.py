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

"""Isolate Muon orthogonalization of the fused Q/latent-KV/RoPE projection."""
from typing import Any

import torch

from hyper_parallel.models.jt_deepseek_v3.adapter.runtime.jt_optimizer import build_optimizer as native_builder


def build_optimizer(*, model: torch.nn.Module, **kwargs: Any) -> Any:
    """Change only the logical matrices sent to Newton-Schulz for linear_qkv."""
    config = dict(kwargs.pop("muon_config"))
    sizes = (model.config.q_lora_rank, model.config.kv_lora_rank, model.config.qk_rope_head_dim)

    def split_qkv(name: str, value: torch.Tensor) -> list[torch.Tensor]:
        if name.endswith(".self_attn.linear_qkv.weight"):
            if value.ndim != 2 or value.shape[0] != sum(sizes):
                raise ValueError(f"Unexpected Q/KV projection layout: {name}, {tuple(value.shape)}")
            return list(value.split(sizes, dim=0))
        return [value]

    config["reshape_fn"] = split_qkv
    print("[P12-FACTOR] muon-qkv-split", sizes, flush=True)
    return native_builder(model=model, muon_config=config, **kwargs)
