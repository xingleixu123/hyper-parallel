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

"""Compose independently observed FP32 embedding and shared-head accumulation."""
from typing import Any
import torch
from torch import nn
import combined_norm_join_builder
from backward_precision_builder import SharedHead


def build_jt_model(**kwargs: Any) -> torch.nn.Module:
    original = combined_norm_join_builder.norm_join_builder.configure_residual_joins

    def configure(model):
        original(model)
        if type(model.lm_head) is not nn.Linear or model.lm_head.bias is not None:
            raise ValueError('Expected native bias-free output head')
        if type(model.model.embed_tokens) is not nn.Embedding:
            raise ValueError('Expected native embedding')
        model.lm_head.__class__ = SharedHead
        model.lm_head.alignment_native_gradient = False
        model.model.embed_tokens.keep_compute_in_fp32 = True
        print('[P12-VOCABULARY-BOTH] FP32 embedding and shared-head gradients',flush=True)

    combined_norm_join_builder.norm_join_builder.configure_residual_joins = configure
    try:
        return combined_norm_join_builder.build_jt_model(**kwargs)
    finally:
        combined_norm_join_builder.norm_join_builder.configure_residual_joins = original
