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

"""Add the independently measured shared LM/MTP join to the residual-join control."""
from typing import Any

import torch

from backward_precision_builder import ResidualGradientDecoder
import norm_join_builder
from one_factor_probe import MFNorm


class SharedResidualGradientDecoder(ResidualGradientDecoder):
    """Preserve residual joins and expose the final branch join in FP32."""

    def forward(self, hidden_states: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Return identical BF16-representable values with an FP32 gradient boundary."""
        return super().forward(hidden_states, *args, **kwargs).float()


def build_jt_model(**kwargs: Any) -> torch.nn.Module:
    """Change only the shared join relative to the selected residual-join control."""
    original = norm_join_builder.configure_residual_joins

    def configure(model: torch.nn.Module) -> None:
        original(model)
        if len(model.mtp.layers) != 1 or type(model.model.layers[-1]) is not ResidualGradientDecoder:
            raise ValueError("Expected the measured residual-join configuration")
        norms = (model.model.norm, model.mtp.layers[0].hnorm)
        if any(type(norm) is not MFNorm for norm in norms):
            raise ValueError("Both shared consumers must retain the verified FP32 norm")
        model.model.layers[-1].__class__ = SharedResidualGradientDecoder
        for norm in norms:
            norm.alignment_output_dtype = torch.bfloat16
        print("[P12-COMBINED-NORM-JOIN] Added shared LM/MTP join to residual-join control", flush=True)

    norm_join_builder.configure_residual_joins = configure
    try:
        return norm_join_builder.build_jt_model(**kwargs)
    finally:
        norm_join_builder.configure_residual_joins = original
