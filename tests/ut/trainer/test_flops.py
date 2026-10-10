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
"""Checks for the MF-compatible DeepSeek model FLOPs estimate."""

from types import SimpleNamespace
import unittest

from hyper_parallel.trainer.runtime.flops import estimate_deepseek_flops


def deepseek_config(**overrides: object) -> SimpleNamespace:
    """Return a small MLA config with one dense and one sparse layer."""
    values = {
        "model_type": "deepseek_v3", "hidden_act": "silu", "hidden_size": 16,
        "num_hidden_layers": 2, "num_attention_heads": 2, "intermediate_size": 32,
        "vocab_size": 32, "qk_nope_head_dim": 4, "qk_rope_head_dim": 4, "v_head_dim": 4,
        "kv_lora_rank": 8, "q_lora_rank": 8, "moe_intermediate_size": 16,
        "num_experts_per_tok": 2, "n_shared_experts": 1, "n_routed_experts": 4,
        "first_k_dense_replace": 1, "moe_layer_freq": 1,
    }
    return SimpleNamespace(**(values | overrides))


class TestDeepseekFlops(unittest.TestCase):
    """Use independent operator dimensions and explicit unsupported cases."""

    def test_dense_moe_mtp_components(self) -> None:
        """Count MLA projections, causal attention, active experts and MTP heads."""
        config = deepseek_config()
        # Forward matmul dimensions for B=1, S=8; backward costs twice forward.
        mla = 8 * (16 + 2 * 8 + 1) + 8 * (16 + 2 * 8 + 1) + 16 * 4 + 2 * 4 * 16
        causal = 8 * 2 * (4 + 4 + 4) / 2
        for depths in (0, 2):
            with self.subTest(depths=depths):
                expected = 6 * 8 * (
                    (mla + causal) * (2 + depths)
                    + 3 * 16 * (32 + 16 * (2 + 1) * (1 + depths))
                    + depths * (3 * 16 + 2 * 16 * 16) + 16 * 32 * (1 + depths)
                )
                actual = estimate_deepseek_flops(config, 1, 8, mtp_depths=depths)
                self.assertEqual(actual, expected, f"expected={expected}, actual={actual}")
                doubled = estimate_deepseek_flops(config, 2, 8, mtp_depths=depths)
                self.assertEqual(doubled, 2 * expected, f"expected={2 * expected}, actual={doubled}")

    def test_config_does_not_imply_executed_mtp(self) -> None:
        """HF configs can name checkpoint MTP layers which the model never executes."""
        config = deepseek_config(num_nextn_predict_layers=4, mlp_layer_types=["dense", "dense"])
        expected = estimate_deepseek_flops(config, 1, 8, mtp_depths=0)
        actual = estimate_deepseek_flops(config, 1, 8)
        self.assertEqual(actual, expected, f"expected={expected}, actual={actual}")
        self.assertGreater(estimate_deepseek_flops(config, 1, 16), 2 * actual)

    def test_unsupported_family_and_invalid_geometry(self) -> None:
        """Do not report plausible FLOPs for an unsupported sparse-attention model."""
        for model_type in (None, "deepseek_v32", "deepseek_v41", "qwen3_5"):
            with self.subTest(model_type=model_type):
                self.assertIsNone(estimate_deepseek_flops(deepseek_config(model_type=model_type), 1, 8))
        for config, batch, length in (
            (deepseek_config(), 0, 8), (deepseek_config(), 1, 0),
            (deepseek_config(mlp_layer_types=["dense"]), 1, 8),
            (deepseek_config(q_lora_rank=0), 1, 8),
        ):
            with self.subTest(config=config, batch=batch, length=length), self.assertRaises(ValueError):
                estimate_deepseek_flops(config, batch, length)
