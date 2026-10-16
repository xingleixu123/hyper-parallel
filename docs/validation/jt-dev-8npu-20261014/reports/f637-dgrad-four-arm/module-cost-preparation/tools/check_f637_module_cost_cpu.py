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

"""待执行的真实 CPU checkpoint 合约检查；不覆盖 NPU fused kernel 或分布式性能。"""

# 当前规范要求 Torch 顶层导入；冻结 lint 插件仍包含已退役 Platform 规则。
# pylint: disable=forbidden-backend-import

import argparse
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional
from torch.utils.checkpoint import DefaultDeviceType

from f637_module_cost_experiment import ModuleCallCounter, TargetInstallation, validate_targets, verify_coverage
from f637_module_cost_spec import TARGETS, write_new_json

from hyper_parallel.core.activation_memory import checkpoint_wrapper
from hyper_parallel.core.fully_shard import api as hsdp_api
from hyper_parallel.models._transformers import model_builder
from hyper_parallel.models.jt_deepseek_v3.modeling_jt_deepseek_v3 import (
    JTDeepseekV3Config, JTDeepseekV3ForCausalLM, JTDeepseekV3MoE,
)


def small_model() -> nn.Module:
    """保留两层主干、一个 MTP、真实路由与 chunk CE，缩小 CPU 算例。"""
    config = JTDeepseekV3Config(
        vocab_size=32, hidden_size=16, intermediate_size=32, moe_intermediate_size=16,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        n_routed_experts=4, n_shared_experts=1, num_experts_per_tok=2, n_group=1, topk_group=1,
        q_lora_rank=8, kv_lora_rank=8, qk_rope_head_dim=4, qk_nope_head_dim=4, v_head_dim=4,
        mlp_layer_types=["dense", "sparse"], max_position_embeddings=32, tie_word_embeddings=False,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        architectures=["JTDeepseekV3ForCausalLM"], num_nextn_predict_layers=1,
        use_pad_tokens=True, norm_topk_prob=True, routed_scaling_factor=1.0,
        moe_aux_loss_coeff=0.01, mtp_loss_factor=0.3, enable_chunk_loss=True, chunk_loss_size=3,
        expert_fp32_weight_gradients=False,
    )
    config.rope_interleave = True
    return JTDeepseekV3ForCausalLM(config)


def cpu_grouped_swiglu(inputs: torch.Tensor, gate_up: torch.Tensor, down: torch.Tensor,
                       counts: torch.Tensor) -> torch.Tensor:
    """只替换 NPU-only GMM，使用可微 CPU SwiGLU；不替换路由/aux/checkpoint。"""
    outputs = []
    for index, tokens in enumerate(inputs.split(counts.tolist())):
        gate, up = functional.linear(tokens, gate_up[index]).chunk(2, dim=-1)
        outputs.append(functional.linear(functional.silu(gate) * up, down[index]))
    return torch.cat(outputs)


class ModuleCostContract(unittest.TestCase):
    """覆盖真实反向与安装失败边界；所有状态均通过 cleanup 恢复。"""

    def setUp(self) -> None:
        """避免 checkpoint 根据已安装 backend 选择加速器设备。"""
        self.addCleanup(DefaultDeviceType.set_device_type, DefaultDeviceType.get_device_type())
        DefaultDeviceType.set_device_type("cpu")
        # enterContext 将 context 的退出登记到 TestCase cleanup，覆盖 setUp/测试异常。
        directory = self.enterContext(tempfile.TemporaryDirectory(  # pylint: disable=consider-using-with
            prefix="jt-module-cost-cpu-"))
        self.output = Path(directory)
        torch.manual_seed(71)

    def _apply(self, model: nn.Module, arm: str, output: Path) -> None:
        with TargetInstallation(arm, output, 0) as installation:
            if TARGETS[arm]:
                model_builder._apply_activation_checkpointing(model, "full")
            installation.verify_after_build(model)

    def _assert_gradients(self, expected: nn.Module, actual: nn.Module) -> None:
        for left, right in zip(expected.parameters(), actual.parameters(), strict=True):
            if left.grad is None:
                self.assertIsNone(right.grad)
            else:
                self.assertIsNotNone(right.grad)
                torch.testing.assert_close(right.grad, left.grad, rtol=1e-5, atol=1e-7)

    @patch("hyper_parallel.models.jt_deepseek_v3.modeling_jt_deepseek_v3.npu_grouped_swiglu",
           side_effect=cpu_grouped_swiglu)
    def test_real_losses_gradients_router_counts_and_hook_removal(self, _grouped: object) -> None:
        """四组×aux/total：三步重放及第四步无 hooks，保留 off 的 head checkpoint。"""
        initial = small_model()
        for arm, targets in TARGETS.items():
            for objective in ("aux", "total"):
                with self.subTest(arm=arm, objective=objective):
                    output = self.output / f"{arm}-{objective}"
                    output.mkdir()
                    expected, actual = copy.deepcopy(initial), copy.deepcopy(initial)
                    self._apply(actual, arm, output)
                    trainer = SimpleNamespace(model=actual, mesh=None, _callbacks=[])
                    counter = ModuleCallCounter(trainer, targets, output, 0)
                    trainer._callbacks.append(counter)
                    tail_steps = []
                    tail = SimpleNamespace(on_step_begin=lambda state: None,
                                           on_step_end=lambda state: tail_steps.append(state.global_step))
                    trainer._callbacks.append(tail)
                    state = SimpleNamespace(global_step=0)
                    owners = [module for module in actual.modules() if isinstance(module, JTDeepseekV3MoE)]
                    reference_owners = [module for module in expected.modules() if isinstance(module, JTDeepseekV3MoE)]
                    head_calls = []
                    head_handle = actual.lm_head.register_forward_pre_hook(
                        lambda _module, _args: head_calls.append(1))
                    try:
                        for step in range(4):
                            for callback in trainer._callbacks:
                                callback.on_step_begin(state)
                            tokens = (torch.arange(8).unsqueeze(0) + step) % 32
                            left = expected(tokens, (tokens + 1) % 32).loss
                            right = actual(tokens, (tokens + 1) % 32).loss
                            torch.testing.assert_close(right, left, rtol=0, atol=0)
                            routed_before_backward = [owner.tokens_per_expert.detach().clone() for owner in owners]
                            before_backward = len(head_calls)
                            key = "foundation_loss/aux"
                            (left[key] if objective == "aux" else sum(left.values())).backward()
                            (right[key] if objective == "aux" else sum(right.values())).backward()
                            if arm == "off" and objective == "total":
                                self.assertGreater(len(head_calls), before_backward)
                            for original, replayed, before in zip(
                                    reference_owners, owners, routed_before_backward, strict=True):
                                torch.testing.assert_close(replayed.tokens_per_expert, original.tokens_per_expert,
                                                           rtol=0, atol=0)
                                torch.testing.assert_close(replayed.tokens_per_expert, before, rtol=0, atol=0)
                                self.assertEqual(before.sum().item(), (step + 1) * tokens.numel() * 2)
                                self.assertIsNone(replayed.tokens_per_expert.grad_fn)
                                self.assertIsNone(replayed.auxiliary_loss)
                            self._assert_gradients(expected, actual)
                            state.global_step += 1
                            for callback in trainer._callbacks:
                                callback.on_step_end(state)
                            if step >= 2:
                                self.assertTrue(counter.closed)
                                self.assertEqual(counter.handles, [])
                                self.assertNotIn(counter, trainer._callbacks)
                        self.assertEqual(tail_steps, [1, 2, 3, 4])
                        self.assertEqual(counter.steps, [1, 2, 3])
                        records = [json.loads(line) for line in counter.path.read_text(encoding="utf-8").splitlines()]
                        self.assertEqual([row["step"] for row in records], [1, 2, 3])
                        for original, replayed in zip(reference_owners, owners, strict=True):
                            original.update_expert_bias(lr=0.001)
                            replayed.update_expert_bias(lr=0.001)
                            self.assertIsNone(replayed.tokens_per_expert)
                            torch.testing.assert_close(replayed.gate.e_score_correction_bias,
                                                       original.gate.e_score_correction_bias, rtol=0, atol=0)
                    finally:
                        head_handle.remove()
                        counter.close("cpu-finally")

    def test_invalid_missing_nested_and_aliased_targets(self) -> None:
        """在任何 wrapper 修改之前拒绝输入错误。"""
        model = nn.Sequential(nn.Linear(3, 3), nn.Sequential(nn.Linear(3, 3)))
        for targets in (("missing",), ("0", "0"), ("1", "1.0"), ("*",), ("",), (None,)):
            with self.subTest(targets=targets):
                with self.assertRaises(ValueError):
                    validate_targets(model, targets)
                self.assertEqual(verify_coverage(model, ()), [])
        model.alias = model[0]
        with self.assertRaisesRegex(ValueError, "Aliased"):
            validate_targets(model, ("0",))

    def test_reject_existing_wrapper_and_unexpected_coverage(self) -> None:
        """拒绝二次包装，且漏计/额外 wrapper 都失败。"""
        model = nn.Sequential(checkpoint_wrapper(nn.Linear(3, 3)))
        with self.assertRaisesRegex(ValueError, "coverage"):
            validate_targets(model, ("0",))
        with self.assertRaisesRegex(ValueError, "coverage"):
            verify_coverage(model, ("1",))

    def test_real_dynamic_fsdp_class_identity(self) -> None:
        """使用真实 class-extension 函数，不初始化 FSDP 状态或进程组。"""
        model = nn.Sequential(checkpoint_wrapper(nn.Linear(3, 3)))
        with patch.dict(hsdp_api.origin_class_to_extend_class):
            hsdp_api._extend_module_with_hsdp_interface(model[0])
        rows = verify_coverage(model, ("0",))
        self.assertTrue(rows[0]["checkpoint_class_in_mro"])
        self.assertTrue(any("HSDPModule" in name for name in rows[0]["mro"]))

    def test_skipped_installer_and_unknown_arm(self) -> None:
        """配置关掉入口不能被记录为命中。"""
        with self.assertRaisesRegex(ValueError, "Unknown"):
            TargetInstallation("other", self.output, 0)
        with TargetInstallation("dense1", self.output, 0) as installation:
            with self.assertRaisesRegex(ValueError, "skipped"):
                installation.verify_after_build(small_model())

    def test_patch_restored_on_error_and_external_owner_preserved(self) -> None:
        """失败恢复自有全局状态；遇所有权冲突保留外部函数。"""
        original = model_builder._apply_activation_checkpointing
        self.addCleanup(setattr, model_builder, "_apply_activation_checkpointing", original)
        with self.assertRaisesRegex(ValueError, "Missing"):
            with TargetInstallation("dense1", self.output, 0):
                model_builder._apply_activation_checkpointing(nn.Linear(3, 3), "full")
        self.assertIs(model_builder._apply_activation_checkpointing, original)
        def replacement(model: nn.Module, _mode: str) -> nn.Module:
            """代表外部稍后安装的无副作用构造入口。"""
            return model

        with self.assertRaisesRegex(RuntimeError, "ownership"):
            with TargetInstallation("dense1", self.output, 0):
                model_builder._apply_activation_checkpointing = replacement
        self.assertIs(model_builder._apply_activation_checkpointing, replacement)

    def test_partial_hook_installation_cleans_owned_hooks(self) -> None:
        """第一个 module 成功注册后漏第二个路径，不得遗留 hooks。"""
        model = small_model()
        first = model.model.layers[0].input_layernorm
        del model.model.layers[0].post_attention_layernorm
        trainer = SimpleNamespace(model=model, mesh=None, _callbacks=[])
        with self.assertRaises(AttributeError):
            ModuleCallCounter(trainer, (), self.output, 0)
        self.assertEqual(len(first._forward_hooks), 0)
        self.assertEqual(len(first._forward_pre_hooks), 0)
        cleanup = json.loads((self.output / "module-cost-cleanup-rank0.json").read_bytes())
        self.assertEqual(cleanup["remaining_owned_hooks"], 0)


def main() -> None:
    """运行后无论成功失败均保存新结果；本文件的编写不代表已经执行。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if any(Path("/dev").glob("davinci[0-9]*")) or torch.npu.is_initialized() or torch.cuda.is_initialized():
        raise RuntimeError("CPU contracts require a deviceless, uninitialized runtime")
    log = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ModuleCostContract)
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    initialized = torch.npu.is_initialized() or torch.cuda.is_initialized()
    write_new_json(args.output, {"status": "PASS" if result.wasSuccessful() and not initialized else "FAIL",
                                  "tests": result.testsRun, "failures": len(result.failures),
                                  "errors": len(result.errors), "accelerators_initialized": initialized,
                                  "log": log.getvalue(),
                                  "scope": "CPU真实JT路由/aux/CE/checkpoint；GMM为可微CPU替代，不是NPU kernel证据。"})
    print(log.getvalue())
    raise SystemExit(0 if result.wasSuccessful() and not initialized else 1)


if __name__ == "__main__":
    main()
