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

"""Isolate residual RMSNorm-gradient joining on the current forward-aligned baseline."""
from typing import Any

import torch

from hyper_parallel.models.jt_deepseek_v3 import modeling_jt_deepseek_v3 as jt
from backward_precision_builder import DualNorm, ResidualGradientDecoder
from forward_factor_probe import configure_factor as configure_qkv
import moe_backward_builder
import moe_precision_builder
from moe_precision_probe import RoundResidualDecoder
from one_factor_probe import MFNorm


def configure_residual_joins(model: torch.nn.Module) -> None:
    """Replace six norm/residual joins before parameter loading and FSDP wrapping."""
    changed = []
    for name, module in model.named_modules():
        if type(module) in {jt.JTDeepseekV3Decoder, RoundResidualDecoder}:
            if any(type(norm) is not MFNorm for norm in
                   (module.input_layernorm, module.post_attention_layernorm)):
                raise ValueError('Expected the established FP32 norm baseline: ' + name)
            module.__class__ = ResidualGradientDecoder
            module.input_layernorm.__class__ = DualNorm
            module.post_attention_layernorm.__class__ = DualNorm
            changed.append(name)
    if len(changed) != 3:
        raise ValueError(f'Expected three decoder blocks, got {changed}')
    print('[P12-NORM-RESIDUAL-JOIN]', changed, flush=True)


def build_jt_model(*, moe_backward_factor: str = 'none', **kwargs: Any) -> torch.nn.Module:
    """Add only residual joining to either the baseline or measured MoE composition."""
    if moe_backward_factor not in {'none', 'archive-measured-join-backward'}:
        raise ValueError('Unsupported independently measured MoE baseline: ' + moe_backward_factor)
    original = moe_precision_builder.configure_factor

    def configure(model, factor):
        if factor != 'norm-swiglu-both-router':
            raise ValueError('Requires the established forward precision baseline')
        original(model, factor)
        configure_residual_joins(model)
        if moe_backward_factor == 'none':
            configure_qkv(model, 'qkv-forward')

    moe_precision_builder.configure_factor = configure
    try:
        if moe_backward_factor == 'none':
            return moe_precision_builder.build_jt_model(**kwargs)
        return moe_backward_builder.build_jt_model(backward_factor=moe_backward_factor, **kwargs)
    finally:
        moe_precision_builder.configure_factor = original
