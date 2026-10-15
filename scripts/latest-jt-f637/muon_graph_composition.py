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

"""Combine measured Muon layouts with the independently tested graph arithmetic."""
from typing import Any

import torch

from hyper_parallel.core.optimizer.muon import Muon
from muon_graph_probe import GraphArithmeticMuon, leaf_optimizers
from muon_layout_composition import build_optimizer as layout_builder


def build_optimizer(*, model: torch.nn.Module, **kwargs: Any) -> Any:
    """Change arithmetic only relative to the complete logical-layout composition."""
    optimizer = layout_builder(model=model, **kwargs)
    count = 0
    for leaf in leaf_optimizers(optimizer):
        if type(leaf) is Muon:
            leaf.__class__ = GraphArithmeticMuon
            count += 1
    if count != 1:
        raise ValueError(f"Expected exactly one Muon leaf: {count}")
    print("[P12-MUON-COMPOSITION] layouts plus graph arithmetic; modern per-piece scale", flush=True)
    return optimizer
