# Copyright 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Show a completed FSDP training/resume chain without implying precision acceptance."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from compare_runs import read_rows


def main() -> None:
    """Render the recorded trajectory and explicitly label the missing reference."""
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("resumed", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    baseline = read_rows(args.baseline, 0)
    resumed = read_rows(args.resumed, 0)
    if sorted(baseline) != list(range(1, 201)) or sorted(resumed) != list(range(201, 401)):
        raise ValueError("Both complete 200-step windows are required")
    config = json.loads((args.baseline / "config.resolved.json").read_text())
    model = config["model"]["config"]
    mtp_scale = model["mtp_loss_factor"]
    aux_scale = model["moe_aux_loss_coeff"]
    plt.rcParams.update({"font.family": "Microsoft YaHei", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none"})
    figure, axes = plt.subplots(2, 2, figsize=(14, 8))
    figure.patch.set_facecolor("#F7F9FC")
    panels = (
        ("training/foundation_loss/lm", "LM loss", 1),
        ("training/foundation_loss/mtp", "MTP loss（按系数还原）", mtp_scale),
        ("training/foundation_loss/aux", "Aux loss（按系数还原）", aux_scale),
        ("performance/step_time", "单步耗时（秒，含观测开销）", 1),
    )
    for axis, (key, title, scale) in zip(axes.flat, panels):
        for rows, color, label in ((baseline, "#215C9C", "从零训练 1–200"),
                                   (resumed, "#D77B22", "checkpoint200 续训 201–400")):
            steps = sorted(rows)
            axis.plot(steps, [rows[step][key] / scale for step in steps], color=color, lw=1.2, label=label)
        axis.axvline(200.5, color="#697789", ls="--", lw=1)
        axis.set_title(title, loc="left", weight="bold")
        axis.set_xlabel("优化器步数")
        axis.set_xlim(1, 400)
        axis.grid(alpha=0.18)
        axis.set_facecolor("white")
    axes[0, 0].legend(frameon=False, fontsize=9)
    figure.text(0.06, 0.96, "FSDP 八卡：0–200 基线与 200–400 续训轨迹", fontsize=19, weight="bold")
    figure.text(0.06, 0.915, "运行完成；独立连续 400 步参考尚未完成，不能据此判定精度通过。",
                fontsize=12, color="#A45B17")
    figure.text(0.06, 0.045, "CANN 9.2.0-beta.2 · JT_dev f637ef7 + 已检视修复 · 图中为 rank0 原始记录\n"
                "MTP / Aux 按配置系数还原；虚线标记恢复边界。本图不展示对照误差。",
                fontsize=10, color="#526276")
    figure.subplots_adjust(left=0.065, right=0.975, top=0.85, bottom=0.15, hspace=0.36, wspace=0.2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output.with_suffix(".png"), dpi=160, facecolor=figure.get_facecolor())
    figure.savefig(args.output.with_suffix(".svg"), facecolor=figure.get_facecolor())
    plt.close(figure)


if __name__ == "__main__":
    main()
