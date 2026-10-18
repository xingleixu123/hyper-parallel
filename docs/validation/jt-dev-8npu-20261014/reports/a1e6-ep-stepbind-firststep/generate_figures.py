# Copyright 2026
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

"""Render the completed same-host three-step diagnostics without implying 200-step acceptance."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE
CASES = ("r125-a1e6-ep-stepbind-fsdp8-3-v1", "r125-a1e6-ep-stepbind-ep4-3-v1")
PAIR_SHA = "18254a7b45a934e88577fe5d806059f874fb541fc5ca68b177dc56bace06da8d"
LM = "training/foundation_loss/lm"


def collect_raw() -> dict:
    """Bind all 16 raw metric files and check the exact three-step input and rank scope."""
    pair_path = (HERE / "evidence/a1e6-ep-firststep-runtime-probe-v4"
                 / "125-v4-fsdp-vs-ep4-native-comparison-v1.json")
    raw = pair_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != PAIR_SHA:
        raise ValueError("The completed native comparison changed")
    pair = json.loads(raw)
    if pair["status"] != "SELECTED_WINDOWS_COMPARED_NOT_CAUSAL" or len(pair["comparisons"]) != 31:
        raise ValueError("Incomplete selected native comparison")
    bindings = [{"path": str(pair_path.relative_to(ROOT)), "sha256": PAIR_SHA, "bytes": len(raw)}]
    metrics = []
    for case in CASES:
        ranks = []
        for rank in range(8):
            path = ROOT / "runs" / case / f"metrics-rank{rank}.jsonl"
            data = path.read_bytes()
            rows = [json.loads(line) for line in data.splitlines()]
            if [row["step"] for row in rows] != [1, 2, 3] or any(row["rank"] != rank for row in rows):
                raise ValueError("Expected exactly three ordered steps for each physical rank")
            bindings.append({"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(data).hexdigest(),
                             "bytes": len(data)})
            ranks.append(rows)
        expected = [row[LM] for row in ranks[0]]
        if not np.isfinite(expected).all() or any([row[LM] for row in rows] != expected for rows in ranks):
            raise ValueError("Main LM differs across ranks or is not finite")
        metrics.append(ranks)
    for left, right in zip(metrics[0], metrics[1]):
        for first, second in zip(left, right):
            if first["input_identity"] != second["input_identity"]:
                raise ValueError("Input identities differ between same-host runs")
    first = np.array([row[LM] for row in metrics[0][0]])
    second = np.array([row[LM] for row in metrics[1][0]])
    return {"scope": "same125 selected first-step D/E0/Q and three-step main-LM; not causal or 200-step acceptance",
            "source": "544526f7db8342c607244fd41009b2a74adb84c8", "cases": CASES,
            "bindings": bindings, "main_lm": {"fsdp": first.tolist(), "ep4": second.tolist(),
                                               "max_abs_per_step": np.abs(first - second).tolist()},
            "all31_native_roles": pair["comparisons"], "performance_comparison_valid": False}


def main() -> None:
    """Render from raw evidence, or regenerate the portable figures from the saved data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-data", action="store_true", help="Regenerate from adjacent figure-data.json only")
    args = parser.parse_args()
    data_path = HERE / "figure-data.json"
    data = json.loads(data_path.read_text(encoding="utf-8")) if args.from_data else collect_raw()
    if len(data["all31_native_roles"]) != 31 or len(data["main_lm"]["fsdp"]) != 3:
        raise ValueError("Incomplete three-step selected-role figure data")
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "DejaVu Sans"], "axes.unicode_minus": False,
                         "font.size": 11, "figure.facecolor": "white", "axes.spines.top": False,
                         "axes.spines.right": False})
    steps = np.arange(1, 4)
    first = np.array(data["main_lm"]["fsdp"])
    second = np.array(data["main_lm"]["ep4"])
    delta = np.abs(first - second)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    for values, name, color, marker in ((first, "FSDP8", "#315a98", "o"),
                                        (second, "EP4 + FSDP", "#bb3b38", "x")):
        axes[0].plot(steps, values, marker=marker, color=color, label=name)
    axes[0].set(xlabel="训练步", ylabel="主 LM loss", xticks=steps)
    axes[0].legend(frameon=False)
    axes[1].bar(steps, delta, color="#bb3b38", width=.5)
    axes[1].set(xlabel="训练步", ylabel="主 LM loss 绝对差", xticks=steps, ylim=(0, max(delta) * 1.45))
    for step, value in zip(steps, delta):
        axes[1].annotate(f"{value:.8f}", (step, value), xytext=(0, 7), textcoords="offset points", ha="center")
    figure.suptitle("125 同机、同源、同输入：三步诊断结果（8 ranks）\n仅覆盖 1–3 步，不是 200 步验收", fontsize=14)
    figure.savefig(HERE / "loss-three-step.png", dpi=180)
    figure.savefig(HERE / "loss-three-step.svg")
    plt.close(figure)

    stages = ("G0.all", "G1.grad", "G3", "U0.weight", "U2.weight")
    labels = ("裁剪前梯度", "裁剪后梯度", "优化器梯度入口", "优化器后权重", "QK 裁剪后权重")
    titles = {"D": "D：第 0 层 input_layernorm，完整 5120 元素",
              "E": "E0：第 1 层专家 0 down_proj，完整 5120 × 1536",
              "Q": "Q：第 0 层 q_b_proj，前 192 × 1536 窗口"}
    figure, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, layout="constrained")
    selected = []
    for axis, target in zip(axes, ("D", "E", "Q")):
        records = {item["stage"]: item for item in data["all31_native_roles"] if item["target"] == target}
        resolved = [records["G3A.grad" if target == "D" else "G3M.grad"] if name == "G3"
                    else records[name] for name in stages]
        selected.extend(resolved)
        for index, item in enumerate(resolved):
            maximum = item["max_abs"]
            if maximum == 0:
                axis.plot(index, 1e-12, "s", color="#315a98")
                axis.annotate("逐字节一致", (index, 1e-12), xytext=(0, 8), textcoords="offset points", ha="center")
            else:
                axis.plot(index, maximum, "o", color="#bb3b38")
                fraction = item["nonzero_count"] / item["numel"]
                axis.annotate(f"{maximum:.3g}\n{fraction:.1%} 元素不同", (index, maximum),
                              xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9)
        axis.set(yscale="log", ylim=(5e-13, 3e-7), ylabel="最大绝对差", title=titles[target])
        axis.grid(axis="y", alpha=.18)
    axes[-1].set(xticks=range(len(stages)), xticklabels=labels)
    figure.suptitle("第 1 步已捕获的梯度／更新分歧边界\n蓝色方块是精确 0，固定置于轴底；红点为实测最大差", fontsize=14)
    figure.savefig(HERE / "gradient-update-boundaries.png", dpi=180)
    figure.savefig(HERE / "gradient-update-boundaries.svg")
    plt.close(figure)
    if not args.from_data:
        data["selected_native_boundaries"] = selected
        data_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"steps": 3, "ranks": 8, "main_lm_max_abs": float(max(delta)), "native_roles": 31}))


if __name__ == "__main__":
    main()
