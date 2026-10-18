"""Render auditable scalar curves and all-rank per-step absolute errors."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from compare_runs import LOSSES, read_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--title", required=True)
    parser.add_argument("--source-label", default="JT_dev + PR #180")
    parser.add_argument("--reference-label", default="FSDP8 reference")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--raw-components", action="store_true")
    parser.add_argument("--chinese", action="store_true")
    parser.add_argument("--partial-observation", action="store_true")
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    if args.partial_observation and report["status"] != "INCOMPLETE_OBSERVED_PREFIX":
        raise ValueError("Partial plots require an explicitly incomplete prefix report")
    start, end = report["steps"]
    steps = list(range(start, end + 1))
    reference = read_rows(args.root / "runs" / report["reference"], 0)
    candidate = read_rows(args.root / "runs" / report["candidate"], 0)
    if args.raw_components:
        for key, scale in ((LOSSES[2], 0.3), (LOSSES[3], 0.0001)):
            for values in (reference, candidate):
                for row in values.values():
                    row[key] /= scale
            report["step_errors"][key] = [value / scale for value in report["step_errors"][key]]
            report["errors"][key]["max_abs"] /= scale
        if any(report["errors"][key]["max_abs"] > report["tolerance_absolute"] for key in LOSSES):
            report["status"] = "FAIL"
    plt.rcParams.update({"font.family": "Microsoft YaHei" if args.chinese else "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "svg.fonttype": "none"})
    figure, axes = plt.subplots(2, 4, figsize=(17.8, 7.8), gridspec_kw={"height_ratios": [1.45, 1]})
    figure.patch.set_facecolor("#F7F9FC")
    names = ("Total loss", "LM loss", "MTP loss (weighted 0.3)", "Aux loss (weighted 0.0001)")
    if args.raw_components:
        names = ("Total loss (weighted sum)", "LM loss", "MTP loss (reconstructed)", "Aux loss (reconstructed)")
    if args.chinese:
        names = ("总损失（加权和）", "LM 损失", "MTP 损失（权重 0.3）", "Aux 损失（权重 0.0001）")
        if args.raw_components:
            names = ("总损失（加权和）", "LM 损失", "MTP 损失（按系数还原）", "Aux 损失（按系数还原）")
    for column, (key, name) in enumerate(zip(LOSSES, names)):
        top, bottom = axes[:, column]
        top.plot(steps, [reference[step][key] for step in steps], color="#215C9C", lw=1.7,
                 label=args.reference_label)
        top.plot(steps, [candidate[step][key] for step in steps], color="#D77B22", lw=1.1,
                 ls="--", label="本次验证" if args.chinese else "Candidate")
        top.set_title(name, loc="left", pad=11)
        top.set_ylabel("损失" if args.chinese else "Loss")
        if column == 0:
            top.legend(frameon=False, loc="upper right", fontsize=9)
        differences = report["step_errors"][key]
        bottom.plot(steps, differences, color="#15755D", lw=1.4)
        bottom.axhline(report["tolerance_absolute"], color="#B22B3C", lw=1.1, ls="--")
        peak = report["errors"][key]["max_abs"]
        error_title = f"全部 8 rank 最大绝对差：{peak:.7g}" if args.chinese else f"Max |error| across 8 ranks: {peak:.7g}"
        bottom.set_title(error_title, loc="left", fontsize=10)
        bottom.set_ylim(-0.00025, max(0.0056, peak * 1.12))
        bottom.set_xlabel("优化器步数" if args.chinese else "Optimizer step")
        bottom.set_ylabel("绝对误差" if args.chinese else "Absolute error")
        for axis in (top, bottom):
            axis.set_facecolor("white")
            axis.grid(alpha=0.2, lw=0.6)
            axis.set_xlim(start, end)
            axis.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
    color = "#15755D" if report["status"] == "PASS" else "#B22B3C"
    figure.text(0.05, 0.955, args.title, ha="left", fontsize=19, weight="bold", color="#192B40")
    subtitle = (f"CANN 9.2.0-beta.2  |  {args.source_label}  |  {end-start+1} steps x 8 ranks  |  "
                f"input mismatches: {report['input_mismatch_count']}  |  absolute gate: 0.005")
    if args.chinese:
        subtitle = (f"CANN 9.2.0-beta.2  |  {args.source_label}  |  {end-start+1} 步 × 8 rank  |  "
                    f"输入不一致：{report['input_mismatch_count']}  |  绝对误差门槛：0.005")
    figure.text(0.05, 0.917, subtitle, ha="left", fontsize=10.5, color="#526276")
    verdict = ("通过" if report["status"] == "PASS" else "未通过") if args.chinese else report["status"]
    if args.partial_observation:
        verdict = "仅阶段证据" if args.chinese else "Partial observation"
        color = "#9B5D00"
    figure.text(0.95, 0.951, verdict, ha="right", fontsize=18, weight="bold", color=color)
    footer = (f"Reference: {report['reference']}\nCandidate: {report['candidate']}  |  "
              "Dashed red = acceptance threshold; upper curves show rank 0, errors cover every rank.")
    if args.chinese:
        footer = (f"参考：{report['reference']}\n验证：{report['candidate']}\n"
                  "红虚线为验收门槛；上排曲线展示 rank0，下排逐步误差覆盖全部 rank。")
    if args.partial_observation:
        footer += (" 未完成200步，不能作为最终验收。" if args.chinese else
                   " Incomplete 200-step run; not final acceptance.")
    if args.raw_components:
        footer += ("\nMTP / Aux 由加权日志分别除以 0.3 / 0.0001 还原，并非额外采集的加权前张量。" if args.chinese else
                   "\nMTP / Aux are reconstructed from weighted logs using 0.3 / 0.0001, not separately captured tensors.")
    figure.text(0.05, 0.027, footer, ha="left", fontsize=9, color="#526276", linespacing=1.6)
    figure.subplots_adjust(left=0.055, right=0.975, top=0.84, bottom=0.21, wspace=0.46, hspace=0.42)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output.with_suffix(".png"), dpi=180, facecolor=figure.get_facecolor())
    svg_path = args.output.with_suffix(".svg")
    figure.savefig(svg_path, facecolor=figure.get_facecolor())
    svg_text = "\n".join(line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines()) + "\n"
    svg_path.write_bytes(svg_text.encode("utf-8"))
    plt.close(figure)


if __name__ == "__main__":
    main()
