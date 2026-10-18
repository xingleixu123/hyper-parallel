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

"""Render two editable scientific figures only from the small real-data accounting JSON."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager, pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


REPORT = Path(__file__).resolve().parents[1]
BLUE = "#235c88"
RED = "#b8443c"
PURPLE = "#7553a4"
INK = "#1e2b38"
GRAY = "#667381"


def style() -> None:
    """Use installed Chinese typography and reproducible editable SVG labels."""
    font = Path("C:/Windows/Fonts/msyh.ttc")
    if font.is_file():
        font_manager.fontManager.addfont(str(font))
    matplotlib.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Microsoft YaHei", "DejaVu Sans"],
        "font.size": 11, "axes.unicode_minus": False, "text.color": INK, "axes.labelcolor": INK,
        "axes.edgecolor": "#bcc7d1", "xtick.color": INK, "ytick.color": GRAY,
        "axes.spines.right": False, "axes.spines.top": False, "figure.facecolor": "white",
        "svg.fonttype": "none", "svg.hashsalt": "a1e6-e0-wgrad-before-rs-v1",
    })


def save(figure: plt.Figure, output: Path) -> None:
    """Save original PNG pixels and an editable SVG with normalized line-end spaces."""
    figure.savefig(output.with_suffix(".png"), dpi=180, metadata={"Software": "E0 real-data accounting v1"})
    svg = output.with_suffix(".svg")
    figure.savefig(svg, metadata={"Date": None, "Creator": "E0 real-data accounting v1"})
    svg.write_text("\n".join(line.rstrip(" \t") for line in svg.read_text(encoding="utf-8").splitlines()) + "\n",
                   encoding="utf-8", newline="\n")
    plt.close(figure)


def plot_fair_domains(data: dict, output: Path) -> None:
    """Compare signed errors only after matching each source-DP contribution domain."""
    groups = data["fair_source_DP_groups"]
    figure, axis = plt.subplots(figsize=(12, 7))
    figure.subplots_adjust(left=0.13, right=0.97, bottom=0.25, top=0.76)
    figure.text(0.07, 0.94, "同一 UID / X / dY，本地 wgrad 在 RS 前可见数值差", fontsize=19, weight="bold")
    figure.text(0.07, 0.884, "544526f7 · 129 · expert0 down_proj[306,58] · step1 / micro0 · 两组统一 ÷8",
                fontsize=11, color=GRAY)
    x = np.arange(len(groups))
    width = 0.28
    fsdp = [group["FSDP_error_div8_units_1e9"] for group in groups]
    ep = [group["EP_error_div8_units_1e9"] for group in groups]
    first = axis.bar(x - width / 2, fsdp, width, color=BLUE, label="FSDP：4个原生 partial 的 FP64 合计")
    second = axis.bar(x + width / 2, ep, width, color=RED, label="EP：单次原生 GMM 输出")
    figure.legend(handles=[first, second], loc="upper left", bbox_to_anchor=(0.12, 0.844),
                  frameon=False, ncol=2, fontsize=10.5)
    axis.axhline(0, color="#788b9d", linewidth=1)
    axis.set_axisbelow(True)
    axis.yaxis.grid(True, color="#e5ebf0", linewidth=0.8)
    axis.set_ylabel("相对同 UID 乘积 FP64 参照的有符号误差\n（统一 ÷8；单位 1e-9）", labelpad=13)
    axis.set_xticks(x, ["source DP0–3\n9698 个 UID", "source DP4–7\n12016 个 UID"])
    axis.tick_params(axis="x", length=0, pad=12)
    axis.set_ylim(-9.1, 1.9)
    axis.set_yticks([-8, -6, -4, -2, 0])
    for bars in (first, second):
        for bar in bars:
            value = bar.get_height()
            axis.text(bar.get_x() + bar.get_width() / 2, value + (0.16 if value > 0 else -0.22),
                      f"{value:+.3f}", ha="center", va="bottom" if value > 0 else "top", weight="bold")
    figure.text(0.07, 0.144, "21714 个 UID（含16个 padding）；X[:,58] 与 dY[:,306] 的 BF16 原生编码全0差。",
                fontsize=10.5)
    figure.text(0.07, 0.093, "参照 = 相同 BF16 数值的离线 FP64 乘积 / fsum；FSDP partial 合计也用离线 FP64。不是 HCCL 模拟。",
                fontsize=10, color=GRAY)
    figure.text(0.07, 0.048, "单坐标 / 单步定位证据；未确认产品数学错误、具体 kernel 或200步根因。", fontsize=10, color=GRAY)
    save(figure, output / "fair-source-dp-errors")


def plot_main_lm(data: dict, output: Path) -> None:
    """Display the original main-LM trajectories and separately scoped residuals."""
    rows = data["main_LM_first3"]
    x = [row["step"] for row in rows]
    figure, (top, bottom) = plt.subplots(2, 1, figsize=(12, 8.5), gridspec_kw={"height_ratios": [1.15, 1]})
    figure.subplots_adjust(left=0.105, right=0.97, bottom=0.21, top=0.82, hspace=0.48)
    figure.text(0.07, 0.947, "三步主 LM：自身轨迹复现，跨拓扑残差仍在", fontsize=19, weight="bold")
    figure.text(0.07, 0.895, "544526f7 · 129 · FSDP8 / EP4 · observer 对各自原200步参考的第1–3步", fontsize=11, color=GRAY)
    top.plot(x, [row["FSDP8_rank0_LM"] for row in rows], color=BLUE, marker="o", markersize=9,
             markerfacecolor="white", linewidth=2.6, label="FSDP8")
    top.plot(x, [row["EP4_rank0_LM"] for row in rows], color=RED, marker="x", markersize=8,
             linestyle="--", linewidth=1.8, label="EP4")
    top.set_ylabel("主 LM loss（rank0日志）")
    top.set_xticks(x)
    top.set_xlim(0.75, 3.25)
    top.set_ylim(12.42, 12.94)
    top.legend(frameon=False, loc="lower left", ncol=2)
    top.set_title("(a) 原始主 LM；两条曲线接近重叠", loc="left", fontsize=11, pad=12)
    scale = 1e-4
    bottom.plot(x, [row["cross_topology_rankmax_abs"] / scale for row in rows], color=PURPLE,
                marker="D", linewidth=2, markersize=6, label="EP4 vs FSDP8")
    bottom.plot(x, [row["own_FSDP8_200_prefix_rankmax_abs"] / scale for row in rows], color=BLUE,
                marker="o", linewidth=1.4, markersize=5, label="FSDP8 vs 自身200前缀")
    bottom.plot(x, [row["own_EP4_200_prefix_rankmax_abs"] / scale for row in rows], color=RED,
                marker="s", markerfacecolor="none", linewidth=1.4, linestyle="--", markersize=9,
                label="EP4 vs 自身200前缀")
    for step, row in zip(x, rows):
        value = row["cross_topology_rankmax_abs"]
        bottom.annotate(f"{value:.6g}", (step, value / scale), xytext=(0, 10), textcoords="offset points",
                        ha="center", color=PURPLE, fontsize=10)
    bottom.set_title("(b) 逐步8 rank最大绝对差；各臂对自身原前缀均为0", loc="left", fontsize=11, pad=12)
    bottom.set_ylabel("主 LM 最大绝对差\n（单位 1e-4）")
    bottom.set_xlabel("训练步")
    bottom.set_xticks(x)
    bottom.set_xlim(0.75, 3.25)
    bottom.set_ylim(-0.18, 1.85)
    bottom.set_yticks([0, 0.5, 1.0, 1.5])
    for axis in (top, bottom):
        axis.set_axisbelow(True)
        axis.yaxis.grid(True, color="#e5ebf0", linewidth=0.8)
    handles, labels = bottom.get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.1, 0.145), ncol=3, frameon=False, fontsize=10)
    figure.text(0.07, 0.07, "每臂24个 rank×step 对照；日志为归约后的全局标量，不是8次独立观测。", fontsize=10, color=GRAY)
    figure.text(0.07, 0.032, "三步透明性 ≠ 200步达标；未作性能对比，也未扩大为 MTP / aux 专项验收。", fontsize=10, color=GRAY)
    save(figure, output / "main-lm-first3")


def main() -> None:
    """Allow a separate output directory for independent reproduction without overwrites."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=REPORT / "data/accounting.json")
    parser.add_argument("--output-dir", type=Path, default=REPORT / "figures")
    args = parser.parse_args()
    data = json.loads(args.data.read_text(encoding="utf-8"))
    if data["schema"] != "e0-before-rs-report-accounting-v1":
        raise ValueError("Unknown figure-data schema")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    style()
    plot_fair_domains(data, args.output_dir)
    plot_main_lm(data, args.output_dir)


if __name__ == "__main__":
    main()
