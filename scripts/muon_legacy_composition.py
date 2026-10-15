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

"""Diagnose archived last-piece scaling on separately measured layout combinations."""
from typing import Any

import torch

from hyper_parallel.core.optimizer import muon as muon_module
from muon_graph_composition import build_optimizer as graph_builder
from muon_layout_composition import build_optimizer as layout_builder
from muon_q_up_legacy_scale_probe import archive_scale


def build_optimizer(*, model: torch.nn.Module, graph_arithmetic: bool = False, **kwargs: Any) -> Any:
    """Change only Q-up scaling; the old behavior is compatibility, not a default recommendation."""
    if model.config.num_attention_heads != 128 or model.config.q_lora_rank != 1536:
        raise ValueError("Unexpected source configuration for archived scale")
    optimizer = (graph_builder if graph_arithmetic else layout_builder)(model=model, **kwargs)
    muon_module.compute_muon_slice_scale = archive_scale
    print("[P12-LEGACY-COMPOSITION] last-piece scale; graph_arithmetic=", graph_arithmetic, flush=True)
    return optimizer
