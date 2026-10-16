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

"""Reproduce the two single-run figures from verified, compact CSV tables."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Optional

from matplotlib import font_manager, rcParams
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


ROOT = Path(__file__).resolve().parent
BLUE = "#4C78A8"
LIGHT_BLUE = "#DCE8F2"
INK = "#20252B"
GRAY = "#7C838B"
GRID = "#DDE1E6"


def load_inputs() -> tuple[list[dict], list[dict], dict]:
    """Load hash-bound plot data without reading tensor payloads or remote hosts."""
    provenance = json.loads((ROOT / "data/provenance.json").read_text(encoding="utf-8"))
    for item in provenance["derived_csv"]:
        source = ROOT / item["path"]
        if hashlib.sha256(source.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"CSV fingerprint mismatch: {item['path']}")
    with (ROOT / "data/loss-and-time.csv").open(encoding="utf-8", newline="") as stream:
        losses = list(csv.DictReader(stream))
    with (ROOT / "data/hbm-samples.csv").open(encoding="utf-8", newline="") as stream:
        samples = list(csv.DictReader(stream))
    if [int(row["step"]) for row in losses] != list(range(1, 201)):
        raise ValueError("Expected exactly the ordered 200 optimizer steps")
    if len(samples) != 82 * 8:
        raise ValueError("Expected 82 physical-HBM samples across eight cards")
    return losses, samples, provenance


def configure_font(path: Optional[str]) -> str:
    """Select an installed CJK font; SVG exports retain the glyphs as paths."""
    if path:
        font_manager.fontManager.addfont(path)
        family = font_manager.FontProperties(fname=path).get_name()
    else:
        choices = ("Microsoft YaHei", "Noto Sans CJK SC", "SimHei")
        installed = {item.name for item in font_manager.fontManager.ttflist}
        family = next((name for name in choices if name in installed), "")
        if not family:
            raise ValueError("Provide an installed Chinese font using --font")
    rcParams.update({
        "font.family": family, "font.size": 10, "text.color": INK,
        "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
        "axes.unicode_minus": False, "svg.fonttype": "path",
    })
    return family


def make_figure(title: str, subtitle: str, height: float) -> tuple[Figure, Axes]:
    """Create consistent quiet axes with room for Chinese titles and notes."""
    figure = Figure(figsize=(12, height), facecolor="white")
    FigureCanvasAgg(figure)
    axis = figure.add_subplot(111)
    figure.subplots_adjust(left=0.085, right=0.97, top=0.79, bottom=0.20)
    figure.text(0.085, 0.95, title, fontsize=17, weight="bold", va="top")
    figure.text(0.085, 0.89, subtitle, fontsize=10.5, va="top", color=GRAY)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["bottom", "left"]].set_color(GRAY)
    axis.grid(axis="y", color=GRID, linewidth=0.7)
    axis.set_axisbelow(True)
    return figure, axis


def save_figure(figure: Figure, stem: str) -> list[dict]:
    """Export the same checked canvas to PNG and portable vector SVG."""
    outputs = []
    for extension in ("png", "svg"):
        path = ROOT / "figures" / f"{stem}.{extension}"
        figure.savefig(path, dpi=180, facecolor="white")
        if extension == "svg":
            path.write_bytes(b"\n".join(line.rstrip() for line in path.read_bytes().splitlines()) + b"\n")
        outputs.append({"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    figure.clear()
    return outputs


def plot_loss(rows: list[dict]) -> list[dict]:
    """Plot the unsmoothed LM trace, without inventing a comparison reference."""
    figure, axis = make_figure(
        "201 单次运行｜200 步 LM loss（非跨机对齐）",
        "8 卡 FSDP · 1–200 优化器步 · 原始逐步值 · 无共享机器参考，未判定精度对齐", 5.8,
    )
    steps = [int(row["step"]) for row in rows]
    losses = [float(row["lm_loss"]) for row in rows]
    axis.plot(steps, losses, color=BLUE, linewidth=1.6)
    axis.scatter([steps[0], steps[-1]], [losses[0], losses[-1]], s=30,
                 facecolors="white", edgecolors=BLUE, linewidths=1.5, zorder=3)
    axis.annotate(f"第 1 步  {losses[0]:.4f}", (steps[0], losses[0]),
                  xytext=(12, 12), textcoords="offset points", fontsize=10)
    axis.annotate(f"第 200 步  {losses[-1]:.4f}", (steps[-1], losses[-1]),
                  xytext=(-10, 18), textcoords="offset points", ha="right", fontsize=10)
    axis.set(xlabel="优化器步", ylabel="LM loss（日志记录值）", xlim=(0, 205), ylim=(0, 14.2))
    axis.set_xticks([1, 25, 50, 75, 100, 125, 150, 175, 200])
    figure.text(0.085, 0.085, "来源：metrics-rank0.jsonl；8 rank 的已记录全局 LM 标量逐步一致。", fontsize=9.5)
    figure.text(0.085, 0.045, "rank 一致不等于跨运行自洽；本图没有参考差值，也不代表 0.005 阈值已通过。",
                fontsize=9.5, color=GRAY)
    return save_figure(figure, "201-lm-loss-200")


def plot_hbm(rows: list[dict], provenance: dict) -> list[dict]:
    """Display sampled physical capacity and leave the unobserved tail explicit."""
    figure, axis = make_figure(
        "201 单次运行｜物理 HBM 采样（非跨机对齐）",
        "82 次采样 × 8 卡 · 每卡 32,768 MiB · 采样值不是 allocator 峰值或瞬时物理峰值", 6.4,
    )
    groups = {}
    for row in rows:
        groups.setdefault(int(row["sample_index"]), []).append(row)
    times, minimum, maximum = [], [], []
    for index in sorted(groups):
        group = groups[index]
        if {int(row["device"]) for row in group} != set(range(8)):
            raise ValueError("Incomplete HBM card group")
        if {int(row["capacity_mib"]) for row in group} != {32768}:
            raise ValueError("Unexpected physical-card capacity")
        times.append(float(group[0]["seconds_from_guard_start"]))
        values = [int(row["hbm_mib"]) for row in group]
        minimum.append(min(values))
        maximum.append(max(values))
    axis.fill_between(times, minimum, maximum, color=LIGHT_BLUE, edgecolor=BLUE,
                      linewidth=0.5, label="每次采样的 8 卡最小–最大范围")
    axis.plot(times, maximum, color=BLUE, linewidth=1.4, marker="o", markersize=2.5,
              label="每次采样的 8 卡最大值")
    axis.axhline(32768, color=INK, linestyle="--", linewidth=1.1, label="单卡容量 32,768 MiB")
    finish = float(provenance["guard_duration_seconds"])
    axis.axvspan(times[-1], finish, facecolor="#F1F2F4", edgecolor=GRAY, hatch="///",
                 linewidth=0, alpha=0.5, label="末次采样后无 HBM 观测")
    peak = max(maximum)
    peak_time = times[maximum.index(peak)]
    axis.annotate(f"采样最高 {peak:,} MiB\n对应剩余 {32768 - peak:,} MiB（4.20%）",
                  xy=(peak_time, peak), xytext=(0.58, 0.47), textcoords="axes fraction",
                  arrowprops={"arrowstyle": "-", "color": GRAY, "linewidth": 0.8},
                  fontsize=10, va="center")
    axis.set(xlabel="距 guard 启动的相对时间（秒）", ylabel="物理 HBM 已用（MiB）",
             xlim=(0, finish), ylim=(0, 35000))
    axis.set_yticks([0, 8192, 16384, 24576, 32768], ["0", "8,192", "16,384", "24,576", "32,768"])
    axis.legend(loc="lower left", bbox_to_anchor=(0, 1.015), ncol=2,
                frameon=False, fontsize=9, borderaxespad=0)
    gap = float(provenance["last_sample_to_finish_seconds"])
    interval = float(provenance["observed_sample_interval_median_seconds"])
    figure.text(0.085, 0.09,
                f"实测采样间隔中位数 {interval:.3f} 秒；末次采样至 receipt 结束 {gap:.2f} 秒无观测。",
                fontsize=9.5)
    figure.text(0.085, 0.047,
                "来源：guard device-samples.jsonl；折线仅连接观测点。最小采样余量不能作为 OOM 安全保证。",
                fontsize=9.5, color=GRAY)
    return save_figure(figure, "201-hbm-sampled-capacity")


def main() -> None:
    """Validate CSV hashes and render local figures without framework or NPU imports."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--font", help="Optional path to a CJK TTF/TTC/OTF font")
    args = parser.parse_args()
    family = configure_font(args.font)
    losses, samples, provenance = load_inputs()
    outputs = plot_loss(losses) + plot_hbm(samples, provenance)
    result = {"renderer": "Matplotlib", "font": family, "outputs": outputs,
              "scope": "Single-run figures only; no cross-host alignment or parallel speedup claim"}
    (ROOT / "figures/render-manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
