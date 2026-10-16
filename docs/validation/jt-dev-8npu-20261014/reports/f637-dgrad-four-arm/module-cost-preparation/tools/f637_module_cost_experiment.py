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

"""模块实验专用接入：精确区域、FSDP 后身份核对及前三步有界计数。"""

from __future__ import annotations

# 当前规范要求 Torch 顶层导入；冻结 lint 插件仍包含已退役 Platform 规则。
# pylint: disable=forbidden-backend-import

from collections import defaultdict
from collections.abc import Callable
import json
from pathlib import Path
from typing import Any

from torch import nn

from f637_module_cost_spec import OBSERVED_TARGETS, TARGETS, write_new_json
from module_profile import install_target_ablation

from hyper_parallel.core.activation_memory import is_recomputing
from hyper_parallel.core.activation_memory.wrapper import CheckpointWrapper
from hyper_parallel.models._transformers import model_builder
from hyper_parallel.trainer.callbacks.base import Callback


def canonical_name(name: str) -> str:
    """仅去除 HP/Torch checkpoint 的结构性路径片段。"""
    return ".".join(part for part in name.split(".")
                    if part not in ("_wrapped_module", "_checkpoint_wrapped_module"))


def verify_coverage(model: nn.Module, expected: tuple[str, ...]) -> list[dict[str, Any]]:
    """按真实类身份识别动态派生 wrapper，并拒绝额外、漏掉或别名区域。"""
    rows = []
    for name, module in model.named_modules(remove_duplicate=False):
        if isinstance(module, CheckpointWrapper):
            rows.append({"fqn": canonical_name(name), "runtime_path": name,
                         "mro": [f"{cls.__module__}.{cls.__qualname__}" for cls in type(module).__mro__],
                         "checkpoint_class_in_mro": CheckpointWrapper in type(module).__mro__})
    actual = [row["fqn"] for row in rows]
    if sorted(actual) != sorted(expected) or len(actual) != len(set(actual)):
        raise ValueError(f"Checkpoint coverage differs: expected={expected}, actual={actual}")
    return rows


def validate_targets(model: nn.Module, targets: tuple[str, ...]) -> None:
    """包装前拒绝通配符、空路径、重复、父子嵌套、漏命中和共享模块别名。"""
    for name in targets:
        if not isinstance(name, str) or not name or any(char in name for char in "*?[]"):
            raise ValueError(f"Target must be an exact nonempty FQN: {name!r}")
    if len(targets) != len(set(targets)):
        raise ValueError("Duplicate recompute targets")
    for name in targets:
        if any(other != name and other.startswith(name + ".") for other in targets):
            raise ValueError(f"Nested recompute targets: {name}")
    verify_coverage(model, ())
    named = dict(model.named_modules(remove_duplicate=False))
    aliases = defaultdict(list)
    for name, module in named.items():
        aliases[id(module)].append(name)
    for name in targets:
        if name not in named:
            raise ValueError(f"Missing recompute target: {name}")
        if len(aliases[id(named[name])]) != 1:
            raise ValueError(f"Aliased recompute target: {name}")


class TargetInstallation:
    """复用原 helper，仅在当前构造阶段替换无公开注册入口的 builder 函数。"""

    def __init__(self, arm: str, output: Path, rank: int) -> None:
        """记录固定 arm 与输出位置；进入 context 后才安装临时入口。"""
        if arm not in TARGETS:
            raise ValueError(f"Unknown module-cost arm: {arm}")
        self.arm = arm
        self.targets = TARGETS[arm]
        self.output = output
        self.rank = rank
        self.original: Callable[..., Any] | None = None
        self.installed: Callable[..., Any] | None = None
        self.applied = False

    def __enter__(self) -> "TargetInstallation":
        """安装仅构造期间生效的检查与既有 wrapper 路径。"""
        if self.original is not None:
            raise RuntimeError("Target installation cannot be entered twice")
        self.original = model_builder._apply_activation_checkpointing  # pylint: disable=protected-access
        if not self.targets:
            return self
        try:
            install_target_ablation(list(self.targets), self.output, self.rank)
            legacy_apply = model_builder._apply_activation_checkpointing  # pylint: disable=protected-access

            def checked_apply(model: nn.Module, mode: str, **kwargs: Any) -> nn.Module:
                """在既有 helper 修改模型之前校验精确目标与单次调用契约。"""
                if self.applied or mode != "full":
                    raise ValueError("Expected one full-mode module-target construction")
                validate_targets(model, self.targets)
                path = self.output / f"module-ablation-rank{self.rank}.json"
                if path.exists():
                    raise FileExistsError(path)
                result = legacy_apply(model, mode, **kwargs)
                verify_coverage(result, self.targets)
                self.applied = True
                return result

            self.installed = checked_apply
            model_builder._apply_activation_checkpointing = checked_apply  # pylint: disable=protected-access
        except BaseException:
            model_builder._apply_activation_checkpointing = self.original  # pylint: disable=protected-access
            raise
        return self

    def __exit__(self, _kind: Any, _value: Any, _traceback: Any) -> None:
        """只恢复仍由本实验持有的 patch，不覆盖外部后来安装的函数。"""
        current = model_builder._apply_activation_checkpointing  # pylint: disable=protected-access
        expected = self.installed if self.targets else self.original
        if current is not expected:
            raise RuntimeError("Builder patch ownership changed during module experiment")
        model_builder._apply_activation_checkpointing = self.original  # pylint: disable=protected-access

    def verify_after_build(self, model: nn.Module) -> None:
        """在真实 FSDP 完成后保存精确区域与动态类 MRO，不依赖类名猜测。"""
        if bool(self.targets) != self.applied:
            raise ValueError("Target installer was skipped or unexpectedly called")
        rows = verify_coverage(model, self.targets)
        write_new_json(self.output / f"module-cost-coverage-rank{self.rank}.json", {
            "arm": self.arm, "expected_targets": self.targets, "observed": rows,
            "ordinary_head_chunk_checkpoint_unchanged": True,
            "scope": "仅 HP 模块 wrapper；off 不代表关闭独立 Torch chunk CE checkpoint。",
        })


class ModuleCallCounter(Callback):
    """前 3 个 TextTrainer 步只计数，不保存 Tensor、计时或调用设备同步。"""

    def __init__(self, trainer: Any, targets: tuple[str, ...], output: Path, rank: int) -> None:
        """为全部九个固定边界安装自有 hooks，失败时回滚已安装部分。"""
        super().__init__(trainer)
        self.targets = targets
        self.output = output
        self.rank = rank
        self.handles = []
        self.pending: dict[str, list[str]] = defaultdict(list)
        self.counts: dict[str, dict[str, int]] = {}
        self.steps: list[int] = []
        self.active_step: int | None = None
        self.closed = False
        self.path = output / f"module-cost-calls-rank{rank}.jsonl"
        with self.path.open("x", encoding="utf-8"):
            pass
        try:
            for name in OBSERVED_TARGETS:
                module = trainer.model.get_submodule(name)
                while isinstance(module, CheckpointWrapper):
                    module = module._wrapped_module  # pylint: disable=protected-access
                self.handles.append(module.register_forward_pre_hook(self._before(name)))
                self.handles.append(module.register_forward_hook(self._after(name), always_call=True))
        except BaseException:
            self.close("partial-installation-failed")
            raise

    def _before(self, name: str) -> Callable[..., None]:
        def hook(_module: nn.Module, _args: tuple[Any, ...]) -> None:
            """只记录调用阶段和次数，不保留输入或输出。"""
            if self.active_step is None:
                return
            phase = "replay" if is_recomputing() else "forward"
            row = self.counts[name]
            row[phase] += 1
            if row[phase] > 64:
                raise RuntimeError(f"Unexpected unbounded module calls: {name}")
            self.pending[name].append(phase)
        return hook

    def _after(self, name: str) -> Callable[..., None]:
        def hook(_module: nn.Module, _args: tuple[Any, ...], result: Any) -> None:
            """平衡正常完成和 checkpoint 早停的对应 pre-hook。"""
            if self.active_step is None or not self.pending[name]:
                return
            phase = self.pending[name].pop()
            suffix = "completed" if result is not None else "early_stop_or_error"
            self.counts[name][f"{phase}_{suffix}"] += 1
        return hook

    def on_step_begin(self, state: Any, **_kwargs: Any) -> None:
        """TextTrainer 在 step_end 前递增 global_step，因此此处加一。"""
        step = state.global_step + 1
        if self.closed or step not in (1, 2, 3) or step != len(self.steps) + 1:
            raise ValueError(f"Unexpected counter lifecycle at step {step}")
        self.active_step = step
        self.counts = {name: {key: 0 for key in (
            "forward", "replay", "forward_completed", "replay_completed",
            "forward_early_stop_or_error", "replay_early_stop_or_error",
        )} for name in OBSERVED_TARGETS}

    def on_step_end(self, state: Any, **_kwargs: Any) -> None:
        """落盘有界计数，在第 3 步移除 hooks 和自身 callback。"""
        if state.global_step != self.active_step or any(self.pending.values()):
            raise RuntimeError("Unbalanced module hooks or mismatched completed step")
        for name, row in self.counts.items():
            if row["forward"] == 0 or (name in self.targets and row["replay"] == 0):
                raise ValueError(f"Module target did not execute/replay: {name}")
            if name not in self.targets and row["replay"]:
                raise ValueError(f"Unexpected HP replay outside target: {name}")
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"step": state.global_step, "rank": self.rank,
                                     "counts": self.counts}) + "\n")
        self.steps.append(state.global_step)
        self.active_step = None
        if state.global_step == 3:
            self.close("three-steps-complete")

    def close(self, reason: str) -> None:
        """幂等释放自有观察；替换列表使当前迭代的其余 callbacks 仍能运行。"""
        if self.closed:
            return
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.pending.clear()
        self.active_step = None
        self.closed = True
        # Trainer 无公开 callback 移除接口；不原地删除正在迭代的列表。
        self.trainer._callbacks = [item for item in self.trainer._callbacks if item is not self]
        write_new_json(self.output / f"module-cost-cleanup-rank{self.rank}.json", {
            "reason": reason, "observed_steps": self.steps, "remaining_owned_hooks": len(self.handles),
            "removed_from_callbacks": self not in self.trainer._callbacks,
        })
