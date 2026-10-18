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
"""Plot full-element use-point shadow statistics and native-byte controls."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


STAGE = Path(__file__).resolve().parents[1]


def main() -> None:
    """Render runtime full-element statistics without treating old sparse grids as maxima."""
    source = STAGE / "paired-comparison-v1.json"
    prefix = STAGE / "own-prefix-root-review-v1.json"
    data = json.loads(source.read_text(encoding="utf-8"))
    control = json.loads(prefix.read_text(encoding="utf-8"))
    if (data["status"] != "N_WIDTH_SHADOW_REPRODUCES_OLD_TP2_NATIVE_BYTES"
            or control["transparent"]["status"] != "PASS"
            or not data["new_production_equals_old_FSDP_native"]):
        raise ValueError("Expected verified native-width result and exact own-prefix transparency")
    halves = [data["full_element_shadow_vs_production"][str(index)] for index in range(2)]
    for row in halves:
        if (row["production_vs_repeat1"] != row["production_vs_repeat2"]
                or row["repeat1_vs_repeat2"]["nonzero_count"] != 0):
            raise ValueError("Both native half-width repeats must remain stable")
    if any(row["different_native_blocks"] != 0 or row["compared_native_blocks"] != 768
           for row in data["comparisons_to_old_TP2"]):
        raise ValueError("Expected all four half-width repeats to match actual old TP2 native blocks")
    stats = [row["production_vs_repeat1"] for row in halves]
    if any(not row["finite"] or row["compared_numel"] != 50_331_648 for row in stats):
        raise ValueError("Unexpected full-element finite comparison scope")
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    font_manager.fontManager.addfont(str(font_path))
    font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams.update({"font.family": font_name, "svg.fonttype": "none", "axes.unicode_minus": False})
    destination = STAGE / "figures"
    destination.mkdir(exist_ok=True)
    outputs = [destination / f"qb-native-width-shadow-v1.{suffix}" for suffix in ("png", "svg")]
    if any(path.exists() for path in outputs):
        raise FileExistsError("The versioned figure already exists")
    figure = plt.figure(figsize=(17, 9), facecolor="#f5f7fb")
    figure.text(0.05, 0.93, "真实 q_b：输出半宽重现 TP2 原生差异", fontsize=25, weight="bold")
    figure.text(0.05, 0.875,
                "544526f7 · 129 / 910B3 · CANN 9.2.0-beta.2 · 首步一个真实 batch / layer0 q_b_proj",
                fontsize=13, color="#596b82")
    labels = ["全宽原算 P vs 全宽重算 R", "P 对应半片0 vs 半宽 V0", "P 对应半片1 vs 半宽 V1"]
    colors = ["#3b8c75", "#bf7833", "#bf7833"]
    axes = [figure.add_axes([0.25, 0.42, 0.31, 0.32]), figure.add_axes([0.66, 0.42, 0.28, 0.32])]
    values = [[0.0] + [row["max_abs"] for row in stats], [0] + [row["nonzero_count"] for row in stats]]
    titles = ["全元素最大绝对差（投影输出，非 loss）", "不同元素数（全元素运行时统计）"]
    for index, (axis, numbers, title) in enumerate(zip(axes, values, titles)):
        axis.barh(range(3), numbers, color=colors, height=0.52)
        axis.invert_yaxis()
        axis.set_yticks(range(3), labels if index == 0 else ["", "", ""])
        axis.set_title(title, fontsize=14, pad=16, weight="bold")
        axis.set_xlim(0, max(numbers) * 1.27)
        axis.grid(axis="x", alpha=0.2)
        axis.set_axisbelow(True)
        axis.tick_params(axis="both", labelsize=11)
        for spine in ("top", "right", "left"):
            axis.spines[spine].set_visible(False)
        for row, number in enumerate(numbers):
            label = f"{number:g}" if index == 0 else f"{number:,}"
            axis.text(number + max(numbers) * 0.025, row, label, va="center", fontsize=13, weight="bold")
    figure.text(0.05, 0.315,
                "M/K 固定 4096/1536；N 从 24576 改为 12288。两半各比较 50,331,648 个元素，合计 7,774 个非零差。",
                fontsize=13, weight="bold", color="#9d5e21")
    figure.text(0.05, 0.253,
                "全宽 R 与 P 原生字节完全一致；两次 V0、两次 V1 各自零差，并分别匹配旧 TP2 的全部 768 块。",
                fontsize=12.5, color="#2d735f")
    figure.text(0.05, 0.193,
                "P 与此前 FSDP 同一使用点也完全一致；本次旁路不回填训练，3步 × 8rank 的训练标量/输入/计数零差。",
                fontsize=12, color="#596b82")
    figure.text(0.05, 0.133,
                "数值最大差采自本次有界 CPU slab 全元素比较；旧 TP2 只存原生 SHA 与稀疏网格，未离线恢复全部数值。",
                fontsize=11.5, color="#596b82")
    figure.text(0.05, 0.073,
                "结论限于一个真实投影使用点：执行形状足以重现该局部差异；未识别具体 kernel，完整200步漂移根因仍未解决。",
                fontsize=11.5, color="#596b82")
    for path in outputs:
        figure.savefig(path, dpi=160, facecolor=figure.get_facecolor())
    plt.close(figure)
    print(json.dumps({"source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                      "prefix_sha256": hashlib.sha256(prefix.read_bytes()).hexdigest(),
                      "outputs": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in outputs}},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
