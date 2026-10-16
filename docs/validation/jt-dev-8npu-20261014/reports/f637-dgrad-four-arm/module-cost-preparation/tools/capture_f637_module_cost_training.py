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

"""当前配方的模块成本入口；保留 capture_training 与 NS shape provenance。"""

# 当前规范要求 Torch 顶层导入；冻结 lint 插件仍包含已退役 Platform 规则。
# pylint: disable=forbidden-backend-import

import os
from typing import Any

import torch
import torch.distributed as dist

import capture_training
from f637_module_cost_experiment import ModuleCallCounter, TargetInstallation
from f637_module_cost_spec import CASES, DISABLED_OVERRIDES, TARGETS, write_new_json
from ns_batch_trace import install_ns_batch_trace

from hyper_parallel.core.optimizer.muon import Muon
from hyper_parallel.models.jt_deepseek_v3.adapter import jt_builder


def validate_environment() -> str:
    """仅接受四组已命名 case，并拒绝残留数学或高频观测 override。"""
    case = os.environ["JT_MODULE_COST_CASE"]
    if case not in CASES or capture_training.OUTPUT.name != case:
        raise ValueError("Module-cost case/output mismatch")
    if os.environ.get("JT_MODULE_COST_ARM") != CASES[case]:
        raise ValueError("Module-cost arm does not match its case")
    for name in DISABLED_OVERRIDES:
        if os.environ.get(name, "0") not in ("", "0"):
            raise ValueError(f"Disabled experiment override is active: {name}")
    if os.environ.get("JT_RECOMPUTE_TARGETS", ""):
        raise ValueError("Module targets must come only from the frozen arm specification")
    return CASES[case]


def main() -> None:
    """仅临时替换构造入口；每组进程独立，退出时恢复所有自有 patch。"""
    arm = validate_environment()
    original_trainer = capture_training.TextTrainer
    original_init = jt_builder._initialize_model_weights  # pylint: disable=protected-access
    original_updates = Muon._compute_batched_ns_updates  # pylint: disable=protected-access
    original_outputs = Muon._compute_batched_ns_outputs_for_tensors  # pylint: disable=protected-access
    owned_ns = None
    counter = None

    def build_trainer(*args: Any, **kwargs: Any) -> Any:
        """仅构造时选择 checkpoint 区域，随后校验真实 FSDP 覆盖。"""
        nonlocal counter
        with TargetInstallation(arm, capture_training.OUTPUT, capture_training.RANK) as installation:
            trainer = original_trainer(*args, **kwargs)
            base = trainer.base
            if (dist.get_world_size() != 8 or base.mesh.dp_shard_size != 8
                    or base.train_iters != 40 or base.num_micro_batches != 1
                    or any(getattr(base.mesh, name) != 1 for name in ("tp_size", "cp_size", "ep_size", "pp_size"))):
                raise ValueError("Module-cost worker requires FSDP8, TP/CP/EP/PP1, micro accumulation1 and 40 steps")
            if torch.npu.matmul.allow_hf32 is not False:
                raise ValueError("Module-cost recipe requires NPU matmul HF32 disabled")
            installation.verify_after_build(base.model)
            write_new_json(capture_training.OUTPUT / f"module-cost-runtime-rank{capture_training.RANK}.json", {
                "arm": arm, "world_size": dist.get_world_size(), "npu_matmul_allow_hf32": False,
                "ns_provenance": "unchanged install_ns_batch_trace metadata-only observer",
                "ordinary_head_chunk_checkpoint_unchanged": True,
            })
        counter = ModuleCallCounter(base, TARGETS[arm], capture_training.OUTPUT, capture_training.RANK)
        base._callbacks.append(counter)  # pylint: disable=protected-access
        return trainer

    try:
        install_ns_batch_trace(capture_training.OUTPUT, capture_training.RANK)
        owned_ns = (Muon._compute_batched_ns_updates, Muon._compute_batched_ns_outputs_for_tensors)
        capture_training.TextTrainer = build_trainer
        capture_training.main()
        if counter is None or counter.steps != [1, 2, 3] or not counter.closed:
            raise RuntimeError("Missing complete first-three-step module coverage")
    finally:
        if counter is not None:
            counter.close("worker-finally")
        if capture_training.TextTrainer is build_trainer:
            capture_training.TextTrainer = original_trainer
        if jt_builder._initialize_model_weights is capture_training.initialize_and_hash:
            jt_builder._initialize_model_weights = original_init
        if owned_ns is not None:
            if Muon._compute_batched_ns_updates is owned_ns[0]:
                Muon._compute_batched_ns_updates = original_updates
            if Muon._compute_batched_ns_outputs_for_tensors is owned_ns[1]:
                Muon._compute_batched_ns_outputs_for_tensors = original_outputs


if __name__ == "__main__":
    main()
