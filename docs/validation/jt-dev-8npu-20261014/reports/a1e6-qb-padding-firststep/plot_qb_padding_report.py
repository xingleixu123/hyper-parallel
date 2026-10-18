# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
#
# SPDX-License-Identifier: Apache-2.0
"""Render two bounded scientific figures from the retained original numeric JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.axes import Axes
from matplotlib.figure import Figure


COLORS = {"F": "#3D4654", "T": "#326CAB", "P": "#B45325"}
BACKGROUND = "#FFFFFF"
INK = "#172333"
MUTED = "#586577"
GRID = "#DCE3EA"
SOURCE_SHA = "f7a2bdd342474820dcfb34e7448f29d263a6e839dd7194ba2152cdcd7bbbe7ed"
VALUES_SHA = "1ceb050cb0c49817eeb0234a60d76ed628323318f5984c4290053cb2c11daeff"
REVIEW_STATUS_SHA = "ae21b0fe2203f0bd1943fa570307cb816d4e07430bb2f1e5a7096fa232e2f1d2"


def load_values(path: Path) -> dict[str, Any]:
    """Bind immutable numeric values and the separate current review status."""
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != VALUES_SHA:
        raise ValueError("Numeric report snapshot differs from its frozen source")
    value = json.loads(raw)
    if value["source_comparison_sha256"] != SOURCE_SHA:
        raise ValueError("Unexpected causal comparison source")
    status_raw = Path(__file__).with_name("review-status.json").read_bytes()
    if hashlib.sha256(status_raw).hexdigest() != REVIEW_STATUS_SHA:
        raise ValueError("Unexpected current review status")
    status = json.loads(status_raw)
    if (status["paired_scientific_review"] != "PASS"
            or status["publication_independent_review"] != "PENDING"
            or status["frozen_values_original_sha256"] != VALUES_SHA
            or status["source_comparison_sha256"] != SOURCE_SHA):
        raise ValueError("Incomplete current status for these candidate figures")
    # value['review_status'] is the preserved historical author snapshot, not current approval.
    comparison = value["original_comparison_fields"]
    if not all(comparison["native_conditions"].values()):
        raise ValueError("An incomplete native comparison cannot support these figures")
    rows = comparison["first_step_main_LM_by_rank"]
    if len(rows) != 8 or any(row["rank"] != index for index, row in enumerate(rows)):
        raise ValueError("Expected exact eight logged copies")
    for row in rows:
        if {key: item for key, item in row.items() if key != "rank"} != {
                key: item for key, item in rows[0].items() if key != "rank"}:
            raise ValueError("Rank logs differ; a single global point would be misleading")
    return value


def configure_style(font_path: Path) -> None:
    """Use an explicit available Chinese font and deterministic SVG identifiers."""
    if not font_path.is_file():
        raise ValueError("Provide an existing Chinese font file with --font")
    font_manager.fontManager.addfont(str(font_path))
    family = font_manager.FontProperties(fname=str(font_path)).get_name()
    matplotlib.rcParams.update({
        "font.family": family, "font.size": 11, "axes.unicode_minus": False,
        "figure.facecolor": BACKGROUND, "axes.facecolor": BACKGROUND,
        "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": INK,
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
        "axes.edgecolor": GRID, "svg.hashsalt": "qb-padding-report-v1", "svg.fonttype": "path",
        "savefig.facecolor": BACKGROUND,
    })


def horizontal_grid(axis: Axes) -> None:
    """Keep numeric scales visible without turning labels into visual decoration."""
    axis.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)
    axis.set_axisbelow(True)
    axis.tick_params(axis="both", length=0, pad=8)


def native_output_figure(value: dict[str, Any]) -> Figure:
    """Separate complete native identity from actual runtime numeric differences."""
    figure = plt.figure(figsize=(15.5, 10.0))
    figure.text(0.055, 0.949, "选中 q_b 的完整输出：P 有效输出与 FSDP 参考逐块一致", size=21, weight="bold")
    figure.text(0.055, 0.906, "step 1 · micro 0 · 单个 logical batch 17c01af6… · layer 0 q_b · TP2 两个 peer",
                size=12, color=MUTED)
    table_axis = figure.add_axes((0.055, 0.58, 0.89, 0.265))
    table_axis.set_axis_off()
    rows = [
        ["X 输入", "4,096 × 1,536", "96", "T、P 均 = 旧 TP2 = FSDP"],
        ["W 权重", "24,576 × 1,536", "2,304", "T、P 均 = 旧 TP2 = FSDP"],
        ["T 控制臂 Y", "4,096 × 24,576", "1,536", "原始 = 有效 = 旧 TP2"],
        ["P 干预臂 Y", "4,096 × 24,576", "1,536", "原始 = 旧 TP2；有效 = FSDP"],
    ]
    table = table_axis.table(
        cellText=rows, colLabels=["完整 logical 张量", "合并后的 shape", "native SHA 块数", "全部块的精确关系"],
        colWidths=[0.17, 0.24, 0.16, 0.43], cellLoc="left", colLoc="left", bbox=(0, 0, 1, 1),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    for (row, _column), cell in table.get_celld().items():
        cell.set_linewidth(0.6)
        cell.set_edgecolor(GRID)
        cell.set_facecolor("#ECF1F6" if row == 0 else BACKGROUND)
        if row == 0:
            cell.get_text().set_weight("bold")
    figure.text(0.055, 0.538, "下方是本次运行实际统计的 Y原始 与 Y有效 差值；每个 peer 完整比较 50,331,648 个元素",
                size=12, weight="bold")
    stats = value["original_comparison_fields"]["P_original_vs_effective_full_numeric"]
    labels = ["rank 4\n全局列 [0, 12,288)", "rank 5\n全局列 [12,288, 24,576)"]
    for panel, field, title, limit, ticks in [
            (0, "nonzero_count", "非零差元素数", 4750, [0, 1000, 2000, 3000, 4000]),
            (1, "max_abs", "完整输出最大绝对差", 0.0185, [0, 0.004, 0.008, 0.012, 0.016])]:
        axis = figure.add_axes((0.095 + panel * 0.46, 0.205, 0.34, 0.255))
        positions = [0, 1]
        bars = axis.bar(positions, [row[field] for row in stats], width=0.48, color=COLORS["P"], zorder=3)
        axis.scatter(positions, [0, 0], s=55, marker="D", color=COLORS["T"], zorder=4, clip_on=False)
        axis.set_xticks(positions, labels)
        axis.set_ylim(0, limit)
        axis.set_yticks(ticks)
        axis.set_title(title, loc="left", size=13, weight="bold", pad=18)
        horizontal_grid(axis)
        for bar, row in zip(bars, stats):
            label = f"{row[field]:,}" if field == "nonzero_count" else str(row[field])
            axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + limit * 0.035,
                      label, ha="center", size=12, color=COLORS["P"], weight="bold")
        for position in positions:
            axis.annotate("T = 0", (position, 0), xytext=(35, 4), textcoords="offset points",
                          color=COLORS["T"], size=10, ha="left")
    figure.text(0.055, 0.105, "蓝色菱形：T 原始/有效输出全量零差    棕色柱：P 原始/有效输出全量差值", size=11)
    figure.text(0.055, 0.064, "SHA 关系覆盖完整目标张量；数值 max 来自本次原生逐 slab 统计，未从旧 SHA 重建数值。",
                size=10.5, color=MUTED)
    figure.text(0.055, 0.033, "仅此 batch、此层、此调用；不代表全模型或 200 步修复。成对科学独审已通过；公开包发布独审仍待完成。",
                size=10.5, color=MUTED)
    return figure


def first_step_lm_figure(value: dict[str, Any]) -> Figure:
    """Plot one global logged LM per arm and signed differences, without ratios."""
    row = value["original_comparison_fields"]["first_step_main_LM_by_rank"][0]
    figure = plt.figure(figsize=(15.5, 7.6))
    figure.text(0.055, 0.935, "首步主 LM 随 P 臂变化，但与 FSDP 仍有残余差", size=22, weight="bold")
    figure.text(0.055, 0.881, "仅首个优化器更新前的前向解释；8 rank 记录的是同一 global 日志值，不是 8 次独立重复。",
                size=12, color=MUTED)
    left = figure.add_axes((0.10, 0.31, 0.35, 0.43))
    order = [("F", "F · FSDP 参考", 2), ("T", "T · TP2 控制", 1), ("P", "P · TP2 padding", 0)]
    for arm, _label, height in order:
        numeric = row["L_" + arm]
        left.scatter(numeric, height, s=95, color=COLORS[arm], zorder=3)
        left.annotate(repr(numeric), (numeric, height), xytext=(0, 13), textcoords="offset points",
                      ha="center", size=10.5, color=COLORS[arm], weight="bold")
    left.set_yticks([item[2] for item in order], [item[1] for item in order])
    left.set_xlim(12.88035, 12.88127)
    left.set_ylim(-0.5, 2.55)
    left.set_xticks([12.8804, 12.8808, 12.8812], ["12.8804", "12.8808", "12.8812"])
    left.set_xlabel("主 LM（点图横轴局部放大，非零起点）", labelpad=13)
    left.set_title("A  原始日志值", loc="left", size=14, weight="bold", pad=24)
    left.grid(axis="x", color=GRID, linewidth=0.7)
    left.tick_params(axis="both", length=0, pad=8)
    right = figure.add_axes((0.60, 0.31, 0.345, 0.43))
    contrasts = [("T − F", row["delta_T"], COLORS["T"]),
                 ("P − F", row["delta_P"], COLORS["P"]),
                 ("P − T", row["P_minus_T"], "#78634E")]
    for index, (label, numeric, color) in enumerate(contrasts):
        height = 2 - index
        right.barh(height, numeric * 10000, height=0.33, color=color, zorder=3)
        right.text(-5.8, height + 0.29, f"{label} = {numeric:+}",
                   size=10.5, color=color, weight="bold")
    right.axvline(0, color=INK, linewidth=0.9, zorder=2)
    right.set_yticks([2, 1, 0], [item[0] for item in contrasts])
    right.set_xlim(-5.95, 2.8)
    right.set_ylim(-0.5, 2.55)
    right.set_xticks([-5, -4, -3, -2, -1, 0, 1, 2])
    right.set_xlabel("有符号差值（单位 0.0001）", labelpad=13)
    right.set_title("B  差值保留方向", loc="left", size=14, weight="bold", pad=24)
    right.grid(axis="x", color=GRID, linewidth=0.7)
    right.set_axisbelow(True)
    right.tick_params(axis="both", length=0, pad=8)
    figure.text(0.055, 0.171, "T：3 步 × 8 rank 的 6 scalar、4 counter、input、CPU 初态及数值配方均与原 TP2 前缀零差。",
                size=11)
    figure.text(0.055, 0.120, "P：只要求数据、LR、计数与初态/配方一致；首次反向也受影响，第 2、3 步不能当作纯前向因果证据。",
                size=11)
    figure.text(0.055, 0.069, "不计算“贡献百分比”，不外推其他 batch；未确认内核身份、唯一根因、生产最优实现或 200 步修复。",
                size=10.5, color=MUTED)
    figure.text(0.055, 0.031, "固定源 544526f7 · 同机 129 · 选中 layer0 q_b · 成对科学独审已通过；公开包发布独审仍待完成",
                size=10, color=MUTED)
    return figure


def save_figure(figure: Figure, output: Path, name: str) -> list[dict[str, Any]]:
    """Write only new figure files and retain their SHA identities."""
    records = []
    for suffix in ("png", "svg"):
        path = output / f"{name}.{suffix}"
        metadata = {"Software": "Matplotlib"} if suffix == "png" else {"Date": None, "Creator": "Matplotlib"}
        with path.open("xb") as handle:
            figure.savefig(handle, format=suffix, dpi=170, metadata=metadata)
        raw = path.read_bytes()
        records.append({"path": path.name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
    plt.close(figure)
    return records


def main() -> None:
    """Render the two figures into an existing directory without replacing files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--values", type=Path, default=Path(__file__).with_name("source-values.json"))
    parser.add_argument("--output", type=Path, default=Path(__file__).parent)
    parser.add_argument("--font", type=Path, default=Path("C:/Windows/Fonts/msyh.ttc"))
    args = parser.parse_args()
    if not args.output.is_dir():
        raise ValueError("Create the intended output directory before rendering")
    names = ("native-target-output", "first-step-main-lm")
    if any((args.output / f"{name}.{suffix}").exists() for name in names for suffix in ("png", "svg")):
        raise FileExistsError("Refusing to overwrite an existing figure")
    value = load_values(args.values)
    configure_style(args.font)
    records = save_figure(native_output_figure(value), args.output, names[0])
    records.extend(save_figure(first_step_lm_figure(value), args.output, names[1]))
    print(json.dumps({"figures": records, "matplotlib_version": matplotlib.__version__,
                      "font_sha256": hashlib.sha256(args.font.read_bytes()).hexdigest()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
