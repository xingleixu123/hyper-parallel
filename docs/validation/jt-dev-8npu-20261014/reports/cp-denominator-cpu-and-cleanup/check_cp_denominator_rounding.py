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

"""Measure CPU rounding at the actual projected CE normalization boundary."""

import hashlib
import inspect
import json
from pathlib import Path
import sys

import torch
import torch_npu
from torch.utils.checkpoint import DefaultDeviceType
from transformers.loss.loss_utils import ForCausalLMLoss

from hyper_parallel.components.losses.chunked_cross_entropy import projected_cross_entropy


def _evaluate(states, weight, targets, mode):
    hidden = states.detach().clone().requires_grad_(True)
    head_weight = weight.detach().clone().requires_grad_(True)
    global_count = targets.ne(-100).sum().clamp_min(1)

    def head(value):
        return torch.nn.functional.linear(value, head_weight)

    def global_loss(**kwargs):
        kwargs["num_items_in_batch"] = global_count
        return ForCausalLMLoss(**kwargs)

    def project(value, labels, loss_fn):
        return projected_cross_entropy(value, labels, head=head, loss_fn=loss_fn,
                                       vocab_size=weight.shape[0], chunk_size=16)

    if mode == "reference_global_denominator":
        loss = project(hidden, targets, ForCausalLMLoss)
    else:
        contributions = []
        for start in (0, 16):
            labels = targets[:, start:start + 16]
            value = hidden[:, start:start + 16]
            if mode == "cp_local_mean_then_scale":
                local_count = labels.ne(-100).sum().float()
                contribution = project(value, labels, ForCausalLMLoss) * (local_count / global_count.float())
            elif mode == "cp_global_denominator":
                contribution = project(value, labels, global_loss)
            else:
                raise ValueError("Unknown diagnostic mode")
            contributions.append(contribution)
        loss = contributions[0] + contributions[1]
    loss.backward()
    return loss.detach(), hidden.grad.detach(), head_weight.grad.detach()


def _difference(reference, candidate):
    result = {}
    for name, expected, observed in zip(("loss", "hidden_gradient", "weight_gradient"), reference, candidate):
        difference = (expected.double() - observed.double()).abs()
        result[name] = {"max_abs": difference.max().item(), "unequal_elements": difference.ne(0).sum().item(),
                        "elements": difference.numel()}
    return result


def main() -> None:
    """Run matched chunk-shape observations without modifying production code or devices."""
    output = Path(sys.argv[1])
    if output.exists():
        raise FileExistsError("Refusing to overwrite diagnostic evidence")
    torch.set_num_threads(2)
    original_default = DefaultDeviceType.get_device_type()
    DefaultDeviceType.set_device_type("cpu")
    rows = []
    try:
        for dtype in (torch.float32, torch.bfloat16):
            for seed in range(8):
                generator = torch.Generator(device="cpu").manual_seed(seed)
                states = torch.randn((1, 32, 32), generator=generator).to(dtype)
                weight = (torch.randn((61, 32), generator=generator) * 0.1).to(dtype)
                labels = torch.randint(61, (1, 32), generator=generator)
                for counts in ((1, 15), (5, 13), (0, 13), (15, 1), (3, 7)):
                    targets = torch.full_like(labels, -100)
                    for part, count in enumerate(counts):
                        start = part * 16
                        targets[:, start:start + count] = labels[:, start:start + count]
                    reference = _evaluate(states, weight, targets, "reference_global_denominator")
                    row = {"dtype": str(dtype), "seed": seed, "valid_counts": counts}
                    for mode in ("cp_local_mean_then_scale", "cp_global_denominator"):
                        row[mode] = _difference(reference, _evaluate(states, weight, targets, mode))
                    rows.append(row)
    finally:
        DefaultDeviceType.set_device_type(original_default)
    if torch_npu.npu.is_initialized():
        raise RuntimeError("This CPU diagnostic unexpectedly initialized NPU")
    source = Path(inspect.getfile(projected_cross_entropy))
    summary = {}
    for mode in ("cp_local_mean_then_scale", "cp_global_denominator"):
        summary[mode] = {}
        for dtype in ("torch.float32", "torch.bfloat16"):
            selected = [row for row in rows if row["dtype"] == dtype]
            summary[mode][dtype] = {
                field: {"cases_with_difference": sum(row[mode][field]["unequal_elements"] > 0 for row in selected),
                        "max_abs": max(row[mode][field]["max_abs"] for row in selected)}
                for field in ("loss", "hidden_gradient", "weight_gradient")
            }
    payload = {"status": "OBSERVED", "cases": len(rows), "npu_initialized": False,
               "source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
               "torch_version": torch.__version__, "summary": summary, "rows": rows,
               "scope_zh": "CPU实际projected_cross_entropy和Transformers CE；参考及分区保持16位置投影块。"
                           "只隔离局部均值再缩放的舍入效应，不包含CP通信、真实NPU算子、Trainer权重或200步训练。"
                           "global分母候选通过局部loss回调注入，无生产源码修改；不能据此判定现有CP漂移根因。"}
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"cases": len(rows), "npu_initialized": False, "summary": summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
