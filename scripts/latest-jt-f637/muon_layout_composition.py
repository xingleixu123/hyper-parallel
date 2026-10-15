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

"""Compose separately measured logical Muon matrices without changing arithmetic."""
from typing import Any

import torch

from hyper_parallel.core.optimizer.muon import NSInputTransform
from hyper_parallel.models.jt_deepseek_v3.adapter.runtime.jt_optimizer import build_optimizer as native_builder


def build_optimizer(*, model: torch.nn.Module, **kwargs: Any) -> Any:
    """Combine attention and expert matrix layouts; retain native per-piece scaling."""
    config = dict(kwargs.pop("muon_config"))
    cfg = model.config

    def transform(name: str, value: torch.Tensor) -> NSInputTransform | None:
        if name.endswith(".self_attn.linear_qkv.weight"):
            parts = list(value.split((cfg.q_lora_rank, cfg.kv_lora_rank, cfg.qk_rope_head_dim), dim=0))
            def merge_qkv(updates, output):
                output.copy_(torch.cat(updates, dim=0))
            return NSInputTransform(parts, merge_qkv)
        if name.endswith((".self_attn.q_b_proj.weight", ".self_attn.kv_b_proj.weight")):
            part_a = cfg.qk_nope_head_dim
            part_b = cfg.qk_rope_head_dim if name.endswith("q_b_proj.weight") else cfg.v_head_dim
            columns = value.shape[1]
            shaped = value.reshape(cfg.num_attention_heads, part_a + part_b, columns)
            parts = [shaped[:, :part_a].reshape(-1, columns).contiguous(),
                     shaped[:, part_a:].reshape(-1, columns).contiguous()]
            def merge_heads(updates, output):
                first = updates[0].reshape(cfg.num_attention_heads, part_a, columns)
                second = updates[1].reshape(cfg.num_attention_heads, part_b, columns)
                output.copy_(torch.cat((first, second), dim=1).reshape_as(output))
            return NSInputTransform(parts, merge_heads)
        if name.endswith(".mlp.experts.gate_up_proj"):
            parts = list(value.chunk(2, dim=1))
            def merge_experts(updates, output):
                output.copy_(torch.cat(updates, dim=1))
            return NSInputTransform(parts, merge_experts)
        return None

    config["ns_transform_fn"] = transform
    print("[P12-MUON-LAYOUT-COMPOSITION] qkv q-up kv-up expert-gate-up", flush=True)
    return native_builder(model=model, muon_config=config, **kwargs)
