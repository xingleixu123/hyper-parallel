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

"""Add the independently measured 3D norm boundary to split and transpose casts."""
from typing import Any

import torch

from muon_graph_probe import GraphArithmeticMuon, leaf_optimizers
from muon_legacy_composition import build_optimizer as legacy_builder
from muon_transpose_cast_probe import SplitTransposeCastMuon


class AllMeasuredCastMuon(SplitTransposeCastMuon):
    """Retain BF16 expert normalization while 2D normalization remains FP32."""

    def _compute_batched_ns_outputs_for_tensors(self, tensor_list, ns_steps, ns_variant="asym5",
                                                ns_coefficients=None, ns_epsilon=1e-10):
        prepared = [value.bfloat16() if value.ndim == 3 else value for value in tensor_list]
        return super()._compute_batched_ns_outputs_for_tensors(
            prepared, ns_steps, ns_variant, ns_coefficients, ns_epsilon)


def build_optimizer(*, model: torch.nn.Module, **kwargs: Any) -> Any:
    """Add one measured factor to the split/transpose experiment."""
    optimizer = legacy_builder(model=model, graph_arithmetic=True, **kwargs)
    count = 0
    for leaf in leaf_optimizers(optimizer):
        if type(leaf) is GraphArithmeticMuon:
            leaf.__class__ = AllMeasuredCastMuon
            count += 1
    if count != 1:
        raise ValueError(f"Expected one native Muon leaf, found {count}")
    print("[P12-ALL-CASTS] split/transpose casts plus expert BF16 norm", flush=True)
    return optimizer
