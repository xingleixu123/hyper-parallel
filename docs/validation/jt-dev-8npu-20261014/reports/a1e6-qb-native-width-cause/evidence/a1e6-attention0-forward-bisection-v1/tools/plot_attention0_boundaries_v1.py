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
"""Plot observed byte-block differences, keeping partial contributors separate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch


STAGE = Path(__file__).resolve().parents[1]
MODULE_PREFIX = "model.layers.0.self_attn."


def main() -> None:
    """Render an auditable figure from the immutable native comparison JSON."""
    source = STAGE / "paired-comparison-v1.json"
    data = json.loads(source.read_text(encoding="utf-8"))
    expected = {"module": MODULE_PREFIX + "q_b_proj", "stage": "output"}
    first = data["first_observed_complete_difference"]
    if any(first[key] != value for key, value in expected.items()):
        raise ValueError("This versioned figure requires the reviewed q_b output boundary")
    boundaries = {(row["module"], row["stage"]): row for row in data["complete_boundaries"]}
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    font_manager.fontManager.addfont(str(font_path))
    font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams.update({"font.family": font_name, "svg.fonttype": "none", "axes.unicode_minus": False})
    destination = STAGE / "figures"
    destination.mkdir(exist_ok=True)
    outputs = [destination / f"attention0-projection-boundaries-v1.{suffix}" for suffix in ("png", "svg")]
    if any(path.exists() for path in outputs):
        raise FileExistsError("The versioned figure already exists")
    figure = plt.figure(figsize=(17, 8.8), facecolor="#f5f7fb")
    axis = figure.add_axes([0.035, 0.23, 0.93, 0.55])
    axis.set_xlim(0, 3.65)
    axis.set_ylim(-0.25, 4.8)
    axis.axis("off")
    figure.text(0.05, 0.92, "第0层 attention：首个观测差异缩到 q_b_proj 输出", fontsize=24, weight="bold")
    figure.text(0.05, 0.865, "544526f7 · 129 / 910B3 · BF16 实际使用点 · 首步8个逻辑batch · FSDP8 对 TP2",
                fontsize=13, color="#596b82")
    for column, title in enumerate(("实际权重 W", "实际输入 X", "原生输出 Y")):
        axis.text(1.04 + 0.89 * column, 4.65, title, ha="center", fontsize=15, weight="bold")
    for row_index, module in enumerate(("linear_qkv", "q_b_proj", "kv_b_proj", "o_proj")):
        y = 3.67 - row_index
        axis.text(0.03, y + 0.35, module, va="center", fontsize=15, weight="bold")
        for column, stage in enumerate(("weight", "input", "output")):
            x = 0.65 + 0.89 * column
            key = (MODULE_PREFIX + module, stage)
            partial = key not in boundaries
            row = boundaries.get(key)
            mismatch = 0 if partial else row["different_batch_blocks"]
            color = "#e7ebf0" if partial else ("#fff0db" if mismatch else "#e2f1eb")
            axis.add_patch(FancyBboxPatch((x, y), 0.8, 0.76, boxstyle="round,pad=0.02", linewidth=0,
                                         facecolor=color))
            if partial:
                label = "K分区贡献\n不可跨分区直接配对"
            elif mismatch:
                label = (f"{mismatch:,} / {row['batch_block_comparisons']:,} 块不同\n"
                         f"网格max差 {row['sample_grid_max_abs_difference']:g}")
            else:
                label = f"{row['batch_block_comparisons']:,} 块全同\n原生字节一致"
            axis.text(x + 0.4, y + 0.38, label, ha="center", va="center", fontsize=11.5,
                      color="#925717" if mismatch else "#2b4655")
    parent = boundaries[("model.layers.0.self_attn", "output")]
    figure.text(0.05, 0.255,
                f"完成生产归约后的 attention 输出：{parent['different_batch_blocks']:,} / "
                f"{parent['batch_block_comparisons']:,} 块不同；"
                f"{parent['different_grid_samples']} / {parent['sample_grid_comparisons']} 个网格点不同，"
                f"网格max差 {parent['sample_grid_max_abs_difference']:g}",
                fontsize=13, weight="bold", color="#a94f20")
    figure.text(0.05, 0.185, "q_b 的648个网格点恰好零差，但4,722个完整块不同；稀疏数值采样不能证明张量相同。",
                fontsize=13, color="#9d3434")
    figure.text(0.05, 0.135, "q_b 实际 M/N/K：4096/24576/1536 → 4096/12288/1536；形状是否足以导致差异仍待因果实验。",
                fontsize=12, color="#596b82")
    figure.text(0.05, 0.085, "两侧各3步×8rank标量、输入、计数均与自身原200步前缀零差。块覆盖完整字节；网格max只覆盖采样点。",
                fontsize=11.5, color="#596b82")
    figure.text(0.05, 0.04, "这里只定位观测边界；未认定具体kernel或训练漂移根因，也不代表完整200步对齐通过。",
                fontsize=11.5, color="#596b82")
    for path in outputs:
        figure.savefig(path, dpi=160, facecolor=figure.get_facecolor())
    plt.close(figure)
    print(json.dumps({"source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                      "outputs": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in outputs}},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
