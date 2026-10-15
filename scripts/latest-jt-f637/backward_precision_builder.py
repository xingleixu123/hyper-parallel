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

"""Isolate shared-head and residual RMSNorm backward rounding boundaries."""
from typing import Any

import torch
from torch import nn
from torch.nn import functional
import torch_npu

from hyper_parallel.models.jt_deepseek_v3 import modeling_jt_deepseek_v3 as jt
import moe_precision_builder
from one_factor_probe import MFNorm


class HeadMatMul(torch.autograd.Function):
    """Use the archive's matrix orientation for the two linear gradients."""

    @staticmethod
    def forward(ctx, values, weight):
        ctx.save_for_backward(values, weight)
        return functional.linear(values, weight)

    @staticmethod
    def backward(ctx, gradient):
        values, weight = ctx.saved_tensors
        dy = gradient.reshape(-1, gradient.shape[-1])
        inputs = values.reshape(-1, values.shape[-1])
        dx = torch.mm(dy, weight).reshape(values.shape) if ctx.needs_input_grad[0] else None
        dw = torch.mm(dy.transpose(0, 1), inputs) if ctx.needs_input_grad[1] else None
        return dx, dw


class SharedHead(nn.Linear):
    """Cast each forward use separately so shared gradients accumulate in FP32."""
    keep_compute_in_fp32 = True

    def forward(self, values):
        weight = self.weight.to(values.dtype)
        if getattr(self, "alignment_native_gradient", False):
            return HeadMatMul.apply(values, weight)
        return functional.linear(values, weight)


class NormResidual(torch.autograd.Function):
    """Round once after joining the residual gradient and RMSNorm input gradient."""

    @staticmethod
    def forward(ctx, values, weight, epsilon):
        inputs = values.float()
        normalized, rstd = torch_npu.npu_rms_norm(inputs, weight.float(), epsilon=epsilon)
        ctx.save_for_backward(inputs, weight, rstd)
        ctx.input_dtype = values.dtype
        return normalized.to(values.dtype), values

    @staticmethod
    def backward(ctx, norm_gradient, residual_gradient):
        inputs, weight, rstd = ctx.saved_tensors
        dx, dw = torch.ops.npu.npu_rms_norm_backward(norm_gradient.float(), inputs, weight.float(), rstd)
        return (dx + residual_gradient.float()).to(ctx.input_dtype), dw, None


class DualNorm(MFNorm):
    """Expose the residual identity at the norm's autograd boundary."""

    def forward(self, hidden_states):
        return NormResidual.apply(hidden_states, self.weight, self.variance_epsilon)


class ResidualGradientDecoder(jt.JTDeepseekV3Decoder):
    """Keep forward values while changing only the two backward merge boundaries."""

    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_values=None, use_cache=False, position_embeddings=None, **kwargs):
        if past_key_values is not None or use_cache:
            raise ValueError("This training probe does not support cached decoding")
        normalized, residual = self.input_layernorm(hidden_states)
        branch, _ = self.self_attn(normalized, attention_mask=attention_mask,
                                   position_ids=position_ids, position_embeddings=position_embeddings, **kwargs)
        hidden_states = residual + branch
        normalized, residual = self.post_attention_layernorm(hidden_states)
        return (residual + self.mlp(normalized)).to(hidden_states.dtype)


def build_jt_model(*, backward_factor: str, **kwargs: Any) -> torch.nn.Module:
    """Apply one backward factor to the measured forward precision baseline."""
    if backward_factor not in {"head-fp32", "head-native", "residual-norm", "embedding-fp32"}:
        raise ValueError(f"Unknown backward factor: {backward_factor}")
    original = moe_precision_builder.configure_factor

    def configure(model, factor):
        if factor != "norm-swiglu-both-router":
            raise ValueError("Requires the established forward precision baseline")
        original(model, factor)
        if backward_factor.startswith("head-"):
            if type(model.lm_head) is not nn.Linear or model.lm_head.bias is not None:
                raise ValueError("Expected the bias-free shared linear head")
            model.lm_head.__class__ = SharedHead
            model.lm_head.alignment_native_gradient = backward_factor == "head-native"
        elif backward_factor == "embedding-fp32":
            if type(model.model.embed_tokens) is not nn.Embedding:
                raise ValueError("Expected the native shared embedding")
            model.model.embed_tokens.keep_compute_in_fp32 = True
        else:
            changed = 0
            for module in model.modules():
                if isinstance(module, jt.JTDeepseekV3Decoder):
                    if not isinstance(module.input_layernorm, MFNorm):
                        raise ValueError("Requires FP32 norms before modifying backward merges")
                    module.__class__ = ResidualGradientDecoder
                    module.input_layernorm.__class__ = DualNorm
                    module.post_attention_layernorm.__class__ = DualNorm
                    changed += 1
            if changed != 3:
                raise ValueError(f"Expected exactly three decoder layers, got {changed}")
        print("[P12-BACKWARD-FACTOR]", backward_factor, flush=True)

    moe_precision_builder.configure_factor = configure
    try:
        return moe_precision_builder.build_jt_model(**kwargs)
    finally:
        moe_precision_builder.configure_factor = original
