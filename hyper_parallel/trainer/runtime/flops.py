# Copyright 2026 Huawei Technologies Co., Ltd
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
"""Model FLOPs estimates for the DeepSeek MLA training path."""

from __future__ import annotations

from typing import Any


def estimate_deepseek_flops(
    config: Any, batch_size: int, sequence_length: int, *, mtp_depths: int = 0,
) -> float | None:
    """Estimate forward + backward FLOPs using the MindFormers MLA formula.

    Counts causal attention, dense/routed/shared SwiGLU experts, MTP projection
    and norms, and output heads. Uses padded sequence length, not supervised
    label count. Excludes optimizer, communication, recomputation and extra
    kernel padding. This is model FLOPs, not measured hardware utilization.

    Args:
        config: HF-compatible DeepSeek V2/V3 or JT model configuration.
        batch_size: Number of full logical sequences.
        sequence_length: Full sequence length before CP partitioning.
        mtp_depths: Number of executed MTP layers; do not infer from checkpoint
            configuration because HF models may not execute their MTP weights.

    Returns:
        Estimated FLOPs, or ``None`` for unsupported model families/activations.

    Raises:
        ValueError: If supported model dimensions or layer types are invalid.
    """
    model_type = getattr(config, "model_type", None)
    if model_type not in ("deepseek_v2", "deepseek_v3", "jt_deepseek_v3"):
        return None
    if getattr(config, "hidden_act", "silu") != "silu":
        return None
    if batch_size <= 0 or sequence_length <= 0 or mtp_depths < 0:
        raise ValueError("FLOPs require positive batch/sequence lengths and nonnegative MTP depth")

    hidden = config.hidden_size
    layers = config.num_hidden_layers
    heads = config.num_attention_heads
    intermediate = config.intermediate_size
    vocab = config.vocab_size
    qk_dim = config.qk_nope_head_dim
    rope_dim = config.qk_rope_head_dim
    value_dim = config.v_head_dim
    kv_rank = config.kv_lora_rank
    q_rank = config.q_lora_rank
    dimensions = (hidden, layers, heads, intermediate, vocab, qk_dim, rope_dim, value_dim, kv_rank)
    if any(value <= 0 for value in dimensions) or (q_rank is not None and q_rank <= 0):
        raise ValueError("DeepSeek FLOPs dimensions must be positive")

    layer_types = getattr(config, "mlp_layer_types", None)
    if layer_types is None:
        first_dense = getattr(config, "first_k_dense_replace", 0)
        frequency = getattr(config, "moe_layer_freq", 1)
        if not isinstance(frequency, int) or frequency <= 0:
            raise ValueError("DeepSeek moe_layer_freq must be a positive integer")
        has_experts = getattr(config, "n_routed_experts", None) is not None
        layer_types = [
            "sparse" if has_experts and index >= first_dense and index % frequency == 0 else "dense"
            for index in range(layers)
        ]
    if len(layer_types) != layers or any(kind not in ("dense", "sparse") for kind in layer_types):
        raise ValueError("DeepSeek mlp_layer_types must contain one dense/sparse entry per layer")

    dense_layers = layer_types.count("dense")
    # JT's executed MTP decoder is sparse, including when the base is dense.
    mtp_is_sparse = model_type == "jt_deepseek_v3" or layer_types[-1] == "sparse"
    moe_layers = layers - dense_layers + (mtp_depths if mtp_is_sparse else 0)
    dense_layers += 0 if mtp_is_sparse else mtp_depths
    mlp_term = intermediate * dense_layers
    if moe_layers:
        moe_intermediate = config.moe_intermediate_size
        topk = config.num_experts_per_tok
        shared = getattr(config, "n_shared_experts", 0) or 0
        if moe_intermediate <= 0 or topk <= 0 or shared < 0:
            raise ValueError("DeepSeek expert dimensions must be positive and shared count nonnegative")
        mlp_term += moe_intermediate * (topk + shared) * moe_layers

    query_term = (hidden * heads * (qk_dim + rope_dim) if q_rank is None
                  else q_rank * (hidden + heads * (qk_dim + rope_dim) + 1))
    attention_term = (
        query_term + kv_rank * (hidden + heads * (qk_dim + value_dim) + 1)
        + hidden * rope_dim + heads * value_dim * hidden
        + sequence_length * heads * (qk_dim + rope_dim + value_dim) / 2
    ) * (layers + mtp_depths)
    # 2 operations per multiply-add, times forward/dgrad/wgrad, as in MF.
    return float(6 * batch_size * sequence_length * (
        3 * hidden * mlp_term + attention_term
        + mtp_depths * (3 * hidden + 2 * hidden * hidden)
        + hidden * vocab * (mtp_depths + 1)
    ))
