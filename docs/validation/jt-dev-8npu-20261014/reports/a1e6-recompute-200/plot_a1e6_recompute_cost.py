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

"""Plot the measured memory and callback-time tradeoff for the fixed JT recipe."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    """Render PNG and editable SVG from the verified three-mode summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    rows = summary["resource_measurements"]
    if [row["mode"] for row in rows] != ["off", "full", "selective"]:
        raise ValueError("Expected the off, full and selective observations in that order")
    if any(row["timing_window"] != [21, 200] or row["ranks"] != 8 for row in rows):
        raise ValueError("Cost comparisons must cover the same post-warmup window and all eight ranks")
    plt.rcParams.update({"font.family": "Microsoft YaHei", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})
    figure, axes = plt.subplots(1, 3, figsize=(14.5, 5.4))
    figure.patch.set_facecolor("#F7F9FC")
    labels = ["关闭 decoder 重算", "full", "selective（SAC）"]
    positions = np.arange(3)
    colors = ["#526A8A", "#215C9C", "#D77B22"]
    allocated = [row["allocated_gib"] for row in rows]
    reserved = [row["reserved_gib"] for row in rows]
    axes[0].bar(positions - 0.18, allocated, 0.36, color=colors, label="allocated")
    axes[0].bar(positions + 0.18, reserved, 0.36, color=colors, alpha=0.38, label="reserved")
    for index, (alloc, reserve) in enumerate(zip(allocated, reserved)):
        axes[0].text(index - 0.18, alloc + 0.4, f"{alloc:.2f}", ha="center", fontsize=9)
        axes[0].text(index + 0.18, reserve + 0.4, f"{reserve:.2f}", ha="center", fontsize=9)
    axes[0].set_title("Allocator 累计峰值（越低越省）", loc="left", weight="bold")
    axes[0].set_ylabel("GiB / NPU")
    axes[0].set_ylim(0, max(reserved) * 1.2)
    axes[0].legend(frameon=False, loc="upper right", fontsize=9)
    metrics = [("median_step_seconds", 1000, "步时中位数（越低越快）", "毫秒", 1),
               ("supervised_tokens_per_second", 1, "有效监督 token 吞吐（越高越快）", "token / 秒", 0)]
    for axis, (key, scale, title, ylabel, digits) in zip(axes[1:], metrics):
        values = [row[key] * scale for row in rows]
        axis.bar(positions, values, width=0.58, color=colors)
        for index, value in enumerate(values):
            delta = (value / values[0] - 1) * 100
            suffix = "参考" if index == 0 else f"{delta:+.2f}%"
            axis.text(index, value + max(values) * 0.025, f"{value:.{digits}f}\n{suffix}",
                      ha="center", va="bottom", fontsize=9)
        axis.set_title(title, loc="left", weight="bold")
        axis.set_ylabel(ylabel)
        axis.set_ylim(0, max(values) * 1.24)
    for axis in axes:
        axis.set_xticks(positions, labels, fontsize=9)
        axis.set_facecolor("white")
        axis.grid(axis="y", alpha=0.2, linewidth=0.6)
        axis.set_axisbelow(True)
    figure.text(0.045, 0.94, "FSDP 八卡重算：实测显存与性能取舍", fontsize=18,
                weight="bold", color="#192B40")
    figure.text(0.045, 0.885, "544526f7 · CANN 9.2.0-beta.2 · 129 / 8×910B3 · 相同数值配方与采集方式",
                fontsize=10, color="#526276")
    footer = ("步时先取全部 8 rank 的每步最大值，再取第 21–200 步中位数；吞吐按同窗口 token 总数 / 步时总数。\n"
              "显存从进程启动累计，包含预热；off 仍保留分块输出损失的独立重算。步时包含标量和输入哈希采集开销。\n"
              "off 使用无保存的连续400步参考之前200步；每种策略目前仅一次观测，不代表重复测量的置信区间或物理 HBM 峰值。")
    figure.text(0.045, 0.055, footer, fontsize=9, color="#526276", linespacing=1.6)
    figure.subplots_adjust(left=0.055, right=0.98, top=0.77, bottom=0.27, wspace=0.36)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output.with_suffix(".png"), dpi=180, facecolor=figure.get_facecolor())
    svg_path = args.output.with_suffix(".svg")
    figure.savefig(svg_path, facecolor=figure.get_facecolor())
    svg_text = "\n".join(line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines()) + "\n"
    svg_path.write_bytes(svg_text.encode("utf-8"))
    plt.close(figure)


if __name__ == "__main__":
    main()
