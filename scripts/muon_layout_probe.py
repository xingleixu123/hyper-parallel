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

"""Independent logical-matrix Muon interventions for attention and routed experts."""
from typing import Any

import torch

from hyper_parallel.core.optimizer.muon import NSInputTransform
from hyper_parallel.models.jt_deepseek_v3.adapter.runtime.jt_optimizer import build_optimizer as native_builder


def build_optimizer(*, model: torch.nn.Module, layout_factor: str, **kwargs: Any) -> Any:
    """Select one matrix boundary while retaining native HP scale and arithmetic."""
    if layout_factor not in {"q-up", "kv-up", "expert-gate-up"}:
        raise ValueError(f"Unknown factor: {layout_factor}")
    config = dict(kwargs.pop("muon_config"))
    head_count = model.config.num_attention_heads
    part_a = model.config.qk_nope_head_dim
    part_b = model.config.qk_rope_head_dim if layout_factor == "q-up" else model.config.v_head_dim

    def transform(name: str, value: torch.Tensor) -> NSInputTransform | None:
        projection = "q_b_proj" if layout_factor == "q-up" else "kv_b_proj"
        if layout_factor in {"q-up", "kv-up"} and name.endswith(f".self_attn.{projection}.weight"):
            columns = value.shape[1]
            shaped = value.reshape(head_count, part_a + part_b, columns)
            parts = [shaped[:, :part_a].reshape(-1, columns).contiguous(),
                     shaped[:, part_a:].reshape(-1, columns).contiguous()]

            def merge(updates: list[torch.Tensor], output: torch.Tensor) -> None:
                first = updates[0].reshape(head_count, part_a, columns)
                second = updates[1].reshape(head_count, part_b, columns)
                output.copy_(torch.cat((first, second), dim=1).reshape_as(output))

            return NSInputTransform(parts, merge)
        if layout_factor == "expert-gate-up" and name.endswith(".mlp.experts.gate_up_proj"):
            if value.ndim != 3:
                raise ValueError(f"Unexpected expert layout: {name}, {tuple(value.shape)}")
            parts = list(value.chunk(2, dim=1))

            def merge(updates: list[torch.Tensor], output: torch.Tensor) -> None:
                output.copy_(torch.cat(updates, dim=1))

            return NSInputTransform(parts, merge)
        return None

    config["ns_transform_fn"] = transform
    print("[P12-FACTOR] muon-layout", layout_factor, flush=True)
    return native_builder(model=model, muon_config=config, **kwargs)
