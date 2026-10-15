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
"""Exercise actual callback collectives on eight CPU ranks with resource sentinels."""

import argparse
from datetime import timedelta
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import distributed as dist

import hyper_parallel
import hyper_parallel.trainer.callbacks.environ_meter_callback as callback_module
from hyper_parallel.trainer.callbacks.environ_meter_callback import EnvironMeterCallback
from hyper_parallel.trainer.runtime import distributed as runtime_distributed
from hyper_parallel.trainer.state import TrainerState


RESOURCE_VALUES = {
    "time": (2, 8, 3, 4, 5, 6, 7, 9),
    "allocated": (1, 13, 2, 3, 4, 5, 6, 7),
    "reserved": (2, 17, 3, 4, 5, 6, 7, 8),
}
LAYOUTS = (
    ("dp2_cp2_tp2_group_boundary", ((0, 2, 4, 6), (1, 3, 5, 7)), 2),
    ("dp2_cp2_pp2_group_boundary", ((0, 1, 2, 3), (4, 5, 6, 7)), 2),
    ("world_dp8_control", (tuple(range(8)),), 1),
)


def run_layout(name: str, groups: tuple, cp_size: int, rank: int) -> list[dict]:
    """Use real Gloo groups; inject only clocks and accelerator-memory counters."""
    selected_group = None
    selected_ranks = None
    created_groups = []
    for ranks in groups:
        group = dist.new_group(ranks=list(ranks), backend="gloo", timeout=timedelta(seconds=40))
        if rank in ranks:
            selected_group = group
            selected_ranks = ranks
            created_groups.append(group)
    if selected_ranks is None:
        raise RuntimeError("Every worker must belong to one data group")
    mesh = SimpleNamespace(dp_cp_mesh=SimpleNamespace(get_group=lambda: selected_group), cp_size=cp_size)
    trainer = SimpleNamespace(mesh=mesh, lr_scheduler=None, optimizer=SimpleNamespace(param_groups=[{"lr": 0.001}]))
    callback = EnvironMeterCallback(trainer)
    records = []
    for step in (1, 2):
        state = TrainerState(global_step=step, epoch=0)
        ticks = iter((100.0, 100.0 + step * RESOURCE_VALUES["time"][rank]))
        clock = SimpleNamespace(perf_counter=lambda: next(ticks))
        device = SimpleNamespace(
            max_memory_allocated=lambda: step * RESOURCE_VALUES["allocated"][rank] * 1024 ** 3,
            max_memory_reserved=lambda: step * RESOURCE_VALUES["reserved"][rank] * 1024 ** 3,
        )
        with (
            patch.object(callback_module, "time", clock),
            patch.object(callback_module, "get_device_type", return_value="synthetic_accelerator_counters"),
            patch.object(callback_module, "get_torch_device", return_value=device),
        ):
            callback.on_step_begin(state)
            callback.on_micro_step_begin(state, {"input_ids": torch.ones(2, 4), "token_count": torch.tensor(6)})
            callback.on_step_end(state, loss=float(rank + 1),
                                 loss_dict={"foundation_loss/lm": float(rank + 1)}, grad_norm=0.5)
        tokens = len(selected_ranks) * 6
        samples = len(selected_ranks) * 2 / cp_size
        mean_loss = sum(peer + 1 for peer in selected_ranks) / len(selected_ranks)
        expected = {
            "performance/step_time": float(9 * step),
            "performance/tokens_per_second": tokens / (9 * step),
            "memory/device_max_allocated_gb": float(13 * step),
            "memory/device_max_reserved_gb": float(17 * step),
            "data/step_tokens": float(tokens),
            "data/consumed_tokens": float(tokens * step),
            "data/step_samples": float(samples),
            "data/consumed_samples": float(samples * step),
            "training/foundation_loss/lm": mean_loss,
            "training/total_loss": mean_loss,
            "training/grad_norm": 0.5,
            "training/lr": 0.001,
        }
        actual = trainer.step_env_metrics
        mismatches = {key: {"actual": actual.get(key), "expected": value}
                      for key, value in expected.items() if actual.get(key) != value}
        records.append({"layout": name, "rank": rank, "step": step, "data_group_ranks": selected_ranks,
                        "cp_size": cp_size, "expected": expected, "actual": actual, "mismatches": mismatches})
    dist.barrier()
    for group in created_groups:
        dist.destroy_process_group(group)
    return records


def main() -> None:
    """Save every worker's measurements and fail unless the expected regression is reproduced."""
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("base", "head"))
    parser.add_argument("source_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if not Path(hyper_parallel.__file__).resolve().is_relative_to(args.source_root.resolve()):
        raise RuntimeError("Editable import does not match the verified source tree")
    if runtime_distributed.get_device_type() != "cpu":
        raise RuntimeError("This diagnostic requires CPU tensors and no accelerator initialization")
    if callable(getattr(getattr(torch, "npu", None), "is_initialized", None)) and torch.npu.is_initialized():
        raise RuntimeError("Unexpected NPU initialization")
    dist.init_process_group("gloo", timeout=timedelta(seconds=45))
    rank = dist.get_rank()
    if dist.get_world_size() != 8:
        raise RuntimeError("The diagnostic requires exactly eight CPU processes")
    try:
        records = []
        for name, groups, cp_size in LAYOUTS:
            records.extend(run_layout(name, groups, cp_size, rank))
        callback_path = Path(callback_module.__file__)
        payload = {
            "mode": args.mode, "rank": rank, "world_size": 8, "backend": str(dist.get_backend()),
            "torch_version": torch.__version__, "source_root": str(args.source_root),
            "callback_sha256": hashlib.sha256(callback_path.read_bytes()).hexdigest(),
            "runtime_distributed_sha256": hashlib.sha256(Path(runtime_distributed.__file__).read_bytes()).hexdigest(),
            "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "npu_initialized": bool(torch.npu.is_initialized()), "records": records,
            "scope_zh": "真实CPU/Gloo通信，时钟和显存读数为注入哨兵；仅检验归约边界，不是NPU显存测量或TP/PP训练。",
        }
        with (args.output / f"{args.mode}-rank{rank}.json").open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        counts = [{} for _ in range(8)]
        local_counts = {
            "all": sum(len(row["mismatches"]) for row in records),
            "data_or_loss": sum(sum(not key.startswith(("performance/", "memory/"))
                                    for key in row["mismatches"]) for row in records),
        }
        dist.all_gather_object(counts, local_counts)
        all_mismatches = sum(item["all"] for item in counts)
        unexpected_data_mismatches = sum(item["data_or_loss"] for item in counts)
        passed = (unexpected_data_mismatches == 0
                  and (all_mismatches == 0 if args.mode == "head" else all_mismatches > 0))
        if rank == 0:
            summary = {"mode": args.mode, "status": "PASS" if passed else "FAIL", "rank_counts": counts,
                       "observations": 8 * len(LAYOUTS) * 2, "metric_mismatches": all_mismatches,
                       "data_or_loss_mismatches": unexpected_data_mismatches}
            with (args.output / f"{args.mode}-summary.json").open("x", encoding="utf-8") as handle:
                json.dump(summary, handle, indent=2)
                handle.write("\n")
            print(json.dumps(summary), flush=True)
        if not passed:
            raise AssertionError("The measured callback result does not match the expected regression verdict")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
