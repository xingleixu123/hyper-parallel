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

"""Plot reviewed native member equality, without deriving numeric error from hashes."""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager, pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle


CSV_SHA256 = '236f6b09e3e0c535a67ab9b010bbad2e291f86587927770fa3291102599b18e9'
LABELS = ["layer0 QKV", "layer1 QKV（对照）", "MTP QKV", "layer1 gate", "MTP gate（对照）"]
STAGES = ["局部 U", "完整 U", "NS 输入", "X0", "X1", "X2", "X3", "X4", "X5",
          "NS 输出", "缩放后", "局部切片", "参数写回"]



def main() -> None:
    """Redraw the reviewed65cell table without private sources or framework execution."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--font", type=Path)
    args = parser.parse_args()
    raw = (Path(__file__).resolve().parent / "muon-first-boundary.csv").read_bytes()
    if hashlib.sha256(raw).hexdigest() != CSV_SHA256:
        raise ValueError("Portable scientific CSV digest differs")
    cells = list(csv.DictReader(io.StringIO(raw.decode("utf-8"))))
    if len(cells) != 65:
        raise ValueError("The table must contain65cells")
    rows, matrix = [], []
    for offset in range(0, 65, 13):
        group = cells[offset:offset + 13]
        names = {x["fqn"] for x in group}
        first = group[0]["first_unequal_boundary"] or None
        if len(names) != 1 or len({x["stage"] for x in group}) != 13:
            raise ValueError("Invalid FQN orstage coverage")
        stages = []
        for x in group:
            equal, total = int(x["equal_parts"]), int(x["compared_parts"])
            same = x["all_native_equal"] == "True"
            if not 0 <= equal <= total or same != (equal == total):
                raise ValueError("Invalid equality count")
            stages.append({"stage": x["stage"], "equal_parts": equal,
                           "compared_parts": total, "all_native_equal": same})
        rows.append({"name": group[0]["fqn"], "first_observed_different_boundary": first,
                     "stages": stages})
        matrix.append([0 if x["all_native_equal"] else 1 for x in stages])
    output = args.output_dir.resolve()
    if output.exists():
        raise ValueError("Output directory must not already exist")
    output.mkdir(parents=True)
    figure_path = output / "muon-first-boundary.png"
    svg_path = output / "muon-first-boundary.svg"
    if args.font is not None:
        font_manager.fontManager.addfont(str(args.font))
        font_name = font_manager.FontProperties(fname=str(args.font)).get_name()
    else:
        available = {item.name for item in font_manager.fontManager.ttflist}
        font_name = next((name for name in ("Microsoft YaHei", "Noto Sans CJK SC", "SimHei")
                          if name in available), None)
        if font_name is None:
            raise ValueError("Install a Chinese font or pass --font")
    plt.rcParams.update({"font.family": font_name, "axes.unicode_minus": False,
                         "svg.hashsalt": "muon-boundary-544526f7", "svg.fonttype": "path"})
    figure, axis = plt.subplots(figsize=(16, 6.2), dpi=150)
    figure.patch.set_facecolor("#f6f8fb")
    axis.imshow(matrix, cmap=ListedColormap(["#d6e9f1", "#e5a667"]), vmin=0, vmax=1, aspect="auto")
    axis.set_xticks(range(13), STAGES, fontsize=11)
    axis.set_yticks(range(5), LABELS, fontsize=11)
    axis.tick_params(axis="both", length=0, pad=10)
    for row_index, row in enumerate(rows):
        for column, stage in enumerate(row["stages"]):
            axis.text(column, row_index, f"{stage['equal_parts']}/{stage['compared_parts']}",
                      ha="center", va="center", fontsize=11, color="#152d40")
        first = row["first_observed_different_boundary"]
        if first:
            column = [item["stage"] for item in row["stages"]].index(first)
            axis.add_patch(Rectangle((column - 0.48, row_index - 0.46), 0.96, 0.92,
                                     fill=False, edgecolor="#a12c26", linewidth=2.5))
    for x in [index - 0.5 for index in range(14)]:
        axis.axvline(x, color="white", linewidth=3)
    for y in [index - 0.5 for index in range(6)]:
        axis.axhline(y, color="white", linewidth=3)
    for spine in axis.spines.values():
        spine.set_visible(False)
    figure.suptitle("Muon 首次更新：跨并行首差边界", x=0.02, ha="left", y=0.97, fontsize=19, weight="bold")
    figure.text(0.02, 0.875, "FSDP8 对 EP4（首步专家梯度与裁剪范数重放） · 固定源码 544526f7",
                fontsize=11, color="#435468")
    figure.legend(handles=[Patch(facecolor="#d6e9f1", label="全部逻辑区间编码哈希一致"),
                           Patch(facecolor="#e5a667", label="存在编码哈希差异"),
                           Patch(facecolor="none", edgecolor="#a12c26", label="首次观测到差异")],
                  loc="lower left", bbox_to_anchor=(0.015, 0.09), ncol=3, frameon=False, fontsize=10)
    figure.text(0.02, 0.025, "单元格：一致区间数 / 对比区间数。X0 为归一化后，X1 为首次 NS 迭代后。\n"
                "3 步定位实验，非 200 步验收；哈希差异不提供数值误差大小，也不能确定唯一算子根因。",
                fontsize=9, color="#435468", linespacing=1.6)
    figure.subplots_adjust(left=0.16, right=0.99, top=0.80, bottom=0.26)
    figure.savefig(figure_path, facecolor=figure.get_facecolor())
    figure.savefig(svg_path, facecolor=figure.get_facecolor(), metadata={"Date": None})
    svg_raw = svg_path.read_bytes()
    svg_path.write_bytes(b"\n".join(line.rstrip(b" \t") for line in svg_raw.splitlines()) + b"\n")
    plt.close(figure)
    print(json.dumps({"cells": len(cells), "PNG": figure_path.name, "SVG": svg_path.name}))


if __name__ == "__main__":
    main()
