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


"""Rebuild the bounded same129 four-arm report from pinned local evidence or portable figure data."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
from typing import Any

import matplotlib
matplotlib.use("Agg")
from matplotlib import colors, pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = "544526f7db8342c607244fd41009b2a74adb84c8"
IMAGE = "sha256:5f9e677f439d11ab4e7c888f54b142335a27eee7411cbd2ce894a4798a70146d"
OWNER = "01a13aa4-a553-7e22-a1c1-ebd16b083bb9"
SCALARS = {"lm": "training/foundation_loss/lm", "mtp": "training/foundation_loss/mtp",
           "aux": "training/foundation_loss/aux", "total": "training/total_loss",
           "norm": "training/grad_norm", "lr": "training/lr"}
COUNTERS = ("data/step_tokens", "data/consumed_tokens", "data/step_samples", "data/consumed_samples")
LOSS_COEFFICIENTS = {"mtp": 0.3, "aux": 0.0001}
ARM_KEYS = ("off_fsdp", "off_ep4", "on_fsdp", "on_ep4")
PAIR_KEYS = ("off_pair", "on_pair", "fsdp_flag", "ep4_flag")
PAIR_LABELS = ("未设：EP4 − FSDP", "设1：EP4 − FSDP", "FSDP：设1 − 未设", "EP4：设1 − 未设")
STAGES = ("G0.all", "G1.grad", "U0.weight", "U2.weight")
STAGE_LABELS = ("裁剪前 G0", "裁剪后 G1", "优化器后权重 U0", "QK阶段后权重 U2")
TARGETS = {"D": (5120, "D｜layer0 input_layernorm，全5120"),
           "E": (7864320, "E0｜layer1 expert0 down_proj，5120×1536"),
           "Q": (294912, "Q｜layer0 q_b_proj，仅前192×1536")}
OUTPUT_NAMES = ("figure-data.json", "machine-summary.json", "execution-provenance.json",
                "loss-four-arms.png", "loss-four-arms.svg", "native-boundaries.png", "native-boundaries.svg")


def require(condition: bool, message: str) -> None:
    """Reject an incomplete or ambiguous source chain without filling missing numeric values."""
    if not condition:
        raise ValueError(message)


def local_path(name: str) -> Path:
    """Keep every source reference within this task root."""
    value = PurePosixPath(name.replace("\\", "/"))
    require(not value.is_absolute() and ".." not in value.parts, "Unsafe relative source path")
    path = ROOT.joinpath(*value.parts).resolve(strict=True)
    require(path.is_relative_to(ROOT.resolve()) and path.is_file(), "Source escaped task root")
    return path


def snapshot(binding: dict, sources: dict[str, dict]) -> bytes:
    """Read and hash one bounded text artifact, retaining only its identity in the report."""
    path = local_path(binding["path"])
    require(path.stat().st_size <= 4 * 1024**2, "Large native blobs must remain local, not enter report memory")
    raw = path.read_bytes()
    require(("bytes" not in binding or len(raw) == binding["bytes"])
            and hashlib.sha256(raw).hexdigest() == binding["sha256"],
            "Pinned text artifact changed: " + binding["path"])
    sources[binding["path"]] = {**binding, "bytes": len(raw)}
    return raw


def read_json(binding: dict, sources: dict[str, dict]) -> Any:
    """Parse only the exact byte snapshot that passed SHA and size checks."""
    return json.loads(snapshot(binding, sources))


def compare_config(left: Any, right: Any, path: str = "") -> list[dict]:
    """List actual resolved-recipe differences instead of hiding arbitrary parallel fields."""
    if type(left) is dict and type(right) is dict and set(left) == set(right):
        return [row for key in sorted(left) for row in compare_config(left[key], right[key], path + "/" + key)]
    if type(left) is list and type(right) is list and len(left) == len(right):
        return [row for index, (first, second) in enumerate(zip(left, right))
                for row in compare_config(first, second, path + "/" + str(index))]
    return [] if type(left) is type(right) and left == right else [{"path": path, "left": left, "right": right}]


def collect_arm(plan: dict, sources: dict[str, dict]) -> tuple[dict, dict, dict]:
    """Bind one exact run, its remote-raw scheduling receipt, scalar scope and retained native manifest."""
    case = plan["case"]
    manifest = read_json(plan["manifest"], sources)
    receipt = read_json(plan["receipt"], sources)
    collection = read_json(plan["collection"], sources)
    verification = read_json(collection["durable_verification"], sources)
    require(manifest["case"] == case and receipt["case"] == case and verification["case"] == case, "Case differs")
    for item in (manifest, receipt, verification):
        require(item["host"].removesuffix("-root") == "129" and item["source"] == SOURCE
                and item["owner"] == OWNER, "Wrong host/source/owner")
    require(receipt["attempted"] is True and receipt["released"] is True
            and type(receipt["exit_code"]) is int and receipt["exit_code"] == 0
            and receipt["container_stopped"] is True and receipt["remaining_owned_pids"] == []
            and receipt["devices"] == list(range(8)) and receipt["image"] == IMAGE, "Incomplete released receipt")
    cid = receipt["container_id"]
    require(re.fullmatch(r"[0-9a-f]{64}", cid) is not None and manifest["container_id"] == cid,
            "Missing complete container identity")
    require(verification["container_absence"]["container_id"] == cid
            and verification["container_absence"]["absent"] is True
            and verification["all_file_SHA_match"] is True
            and verification["manifest_sha256"] == plan["manifest"]["sha256"], "Collection/absence chain differs")
    require(manifest["released"] is True and manifest["allow_incomplete"] is False
            and verification["both_duplicate_archives_removed"] is True
            and type(collection["exit"]) is int and collection["exit"] == 0, "Incomplete collection")
    entries = {row["path"]: row for row in manifest["files"]}
    mappings = {row["remote_path"]: row for row in verification["remote_to_local_files"]}
    require(len(entries) == len(manifest["files"]) == 111 and set(entries) == set(mappings), "Incomplete manifest")
    retained = []
    for name, entry in entries.items():
        mapping = mappings[name]
        require(mapping["sha256"] == entry["sha256"] and mapping["bytes"] == entry["bytes"], "Mapping SHA differs")
        if name.endswith(".bin"):
            require(re.fullmatch("runs/" + re.escape(case) + r"/ep-native-rank[0-7]\.bin", name) is not None,
                    "Unexpected native blob")
            require(local_path(mapping["local_path"]).stat().st_size == entry["bytes"],
                    "Retained native file absent or truncated")
            retained.append(dict(entry))
        else:
            snapshot({"path": mapping["local_path"], "sha256": entry["sha256"], "bytes": entry["bytes"]}, sources)
    require(len(retained) == 8, "Native file count differs")
    remote_receipt = entries["scheduling/" + case + ".receipt.json"]
    raw_receipt = read_json({**remote_receipt, "path": mappings[remote_receipt["path"]]["local_path"]}, sources)
    require(json.dumps(receipt, sort_keys=True) == json.dumps(raw_receipt, sort_keys=True),
            "Canonical and collected remote receipt semantics differ")

    def member(name: str, json_value: bool = True) -> Any:
        binding = entries["runs/" + case + "/" + name]
        return read_json(binding, sources) if json_value else snapshot(binding, sources)

    require(member("EXIT_CODE", False).strip() == b"0"
            and member("source-commit.txt", False).decode().strip() == SOURCE,
            "Wrong source or exit marker")
    require(not member("source-status.txt", False).strip() and not member("source.patch", False).strip(),
            "Source checkout not clean")
    config = member("config.resolved.json")
    require(config["training"]["train_iters"] == 3 and config["training"]["lr_scheduler_iters"] == 1000
            and config["training"]["seed"] == 42 and config["training"]["global_batch_size"] == 8
            and config["training"]["micro_batch_size"] == 1
            and config["model"]["config"]["expert_fp32_weight_gradients"] is True
            and config["model"]["config"]["mtp_loss_factor"] == LOSS_COEFFICIENTS["mtp"]
            and config["model"]["config"]["moe_aux_loss_coeff"] == LOSS_COEFFICIENTS["aux"]
            and config["optimizer"]["muon_config"]["ns_dtype"] == "float32", "Numerical recipe changed")
    rows_by_rank, initials, flags = [], [], []
    source_proofs = []
    for rank in range(8):
        rows = [json.loads(line) for line in member(f"metrics-rank{rank}.jsonl", False).splitlines()]
        require([row["step"] for row in rows] == [1, 2, 3] and all(row["rank"] == rank for row in rows),
                "Incomplete or reordered scalar trajectory")
        require(all(type(row[key]) in (int, float) and math.isfinite(row[key]) for row in rows
                    for key in (*SCALARS.values(), *COUNTERS)), "Nonfinite scalar")
        start, complete = member(f"start-rank{rank}.json"), member(f"complete-rank{rank}.json")
        require(start["global_step"] == 0 and start["train_iters"] == 3 and start["micro_batches"] == 1
                and complete["global_step"] == 3, "Run window differs")
        initial = member(f"initial-state-rank{rank}.json")
        require(len(initial) == 48, "Initial parameter hash scope differs")
        initial_raw = json.dumps(initial, sort_keys=True).encode()
        initials.append({"count": len(initial), "canonical_sha256": hashlib.sha256(initial_raw).hexdigest()})
        runtime = member(f"runtime-flags-rank{rank}.json")
        require(all(runtime[phase]["environment"]["CLOSE_MATMUL_K_SHIFT"] == plan["expected_flag"]
                    for phase in ("before_trainer", "training")), "Actual flag differs")
        flags.append(runtime)
        metadata = json.loads(member(f"ep-firststep-rank{rank}.jsonl", False).splitlines()[0])
        require(metadata["kind"] == "metadata" and metadata["source_commit"] == SOURCE
                and metadata["rank"] == rank and metadata["schema"] == "ep-firststep-runtime-v4"
                and len(metadata["source_proof"]) == 17, "Native source scope differs")
        source_proofs.append({"product": metadata["source_proof"], "framework": metadata["framework_source_proof"]})
        rows_by_rank.append(rows)
    require(all(flags[rank] == flags[0] and source_proofs[rank] == source_proofs[0] for rank in range(8)),
            "Runtime/source proof differs across ranks")
    require(all(all(row[key] == rows_by_rank[0][step][key] for key in (*SCALARS.values(), *COUNTERS, "input_identity"))
                for rows in rows_by_rank for step, row in enumerate(rows)), "Rank-global logged values differ")
    result = {"case": case, "key": plan["key"], "scalars": {key: [row[value] for row in rows_by_rank[0]]
               for key, value in SCALARS.items()}, "counters": [{key: row[key] for key in COUNTERS}
               for row in rows_by_rank[0]], "inputs": [row["input_identity"] for row in rows_by_rank[0]],
              "initials": initials, "runtime": flags[0], "source_proofs": source_proofs[0],
              "rank_count": 8, "steps": [1, 2, 3], "retained_native": retained,
              "manifest": plan["manifest"], "receipt": plan["receipt"], "collection": plan["collection"]}
    safe_receipt = {"case": case, "source": SOURCE, "image": IMAGE, "host": "129", "exit_code": 0,
                    "released": True, "full_CID_owner_case_host_bound": True,
                    "container_id_sha256": hashlib.sha256(cid.encode()).hexdigest(), "collected_absent": True,
                    "canonical_receipt": plan["receipt"], "remote_raw_receipt": remote_receipt,
                    "manifest": plan["manifest"], "verification": collection["durable_verification"]}
    return result, config, safe_receipt


def collect_raw() -> dict:
    """Recompute scalars and identity links; native numeric statistics come from bound full31 comparisons/review."""
    sources: dict[str, dict] = {}
    plan = json.loads((HERE / "study-inputs.json").read_bytes())
    require(plan["source"] == SOURCE and [row["key"] for row in plan["arms"]] == list(ARM_KEYS), "Wrong study")
    review_plan = plan["scientific_review"]
    require("binding" in review_plan, "Full four-arm independent review is still pending")
    review = read_json(review_plan["binding"], sources)
    require(review["status"].startswith("PASS"), "Independent scientific review did not pass")
    read_json(plan["off_pair_review"], sources)
    snapshot(plan["scope_audit"], sources)
    snapshot(plan["norm_audit"], sources)
    arms, configs, receipts = {}, {}, {}
    for arm in plan["arms"]:
        result, config, receipt = collect_arm(arm, sources)
        arms[arm["key"]], configs[arm["key"]], receipts[arm["key"]] = result, config, receipt
    base = arms["off_fsdp"]
    for arm in arms.values():
        require(arm["inputs"] == base["inputs"] and arm["counters"] == base["counters"]
                and arm["initials"] == base["initials"] and arm["source_proofs"] == base["source_proofs"]
                and arm["scalars"]["lr"] == base["scalars"]["lr"], "Initial/input/LR/source differs")
        require(all(row["rank_count"] == 8 and row["micro_steps"] == 1 for row in arm["inputs"]), "Input scope differs")
    recipe_deltas = {}
    for first, second in (("off_fsdp", "on_fsdp"), ("off_ep4", "on_ep4")):
        delta = compare_config(configs[first], configs[second])
        require([row["path"] for row in delta] == ["/checkpoint/checkpoint_dir"], "More than output path changed")
        recipe_deltas[first + "_to_" + second] = delta
        old, new = copy.deepcopy(arms[first]["runtime"]), copy.deepcopy(arms[second]["runtime"])
        for phase in ("before_trainer", "training"):
            del old[phase]["environment"]["CLOSE_MATMUL_K_SHIFT"]
            del new[phase]["environment"]["CLOSE_MATMUL_K_SHIFT"]
        require(old == new, "Other captured runtime flags changed")
    pairs, role_keys = {}, None
    for item in plan["pairs"]:
        canonical = read_json(item["canonical"], sources)
        execution = read_json(item["execution"], sources)
        command = execution["command"]
        require(execution["exit"] == 0 and execution["report_sha256"] == item["canonical"]["sha256"],
                "Comparator execution/result SHA differs")
        for side in ("left", "right"):
            path = command[command.index("--" + side) + 1].replace("\\", "/")
            require(path.endswith("/runs/" + arms[item[side]]["case"]), "Canonical relation identity differs")
        require(canonical["status"] == "SELECTED_WINDOWS_COMPARED_NOT_CAUSAL"
                and len(canonical["comparisons"]) == 31, "Incomplete native comparison")
        keys = {(row["target"], row["stage"]) for row in canonical["comparisons"]}
        require(len(keys) == 31 and (role_keys is None or keys == role_keys), "Native role set differs")
        role_keys = keys
        for row in canonical["comparisons"]:
            require(row["numel"] == TARGETS[row["target"]][0] and row["left_finite_count"] == row["numel"]
                    and row["right_finite_count"] == row["numel"] and 0 <= row["nonzero_count"] <= row["numel"]
                    and all(math.isfinite(row[key]) and row[key] >= 0
                            for key in ("max_abs", "mse", "squared_error_sum")),
                    "Invalid native statistic")
        pairs[item["key"]] = {"left": item["left"], "right": item["right"], "canonical": item["canonical"],
                              "execution": item["execution"], "roles": canonical["comparisons"],
                              "numeric_nonzero_roles": sum(row["nonzero_count"] > 0
                                                           for row in canonical["comparisons"]),
                              "byte_mismatch_roles": sum(row["byte_mismatched_slabs"] > 0
                                                         for row in canonical["comparisons"])}
    require(tuple(pairs) == PAIR_KEYS, "Wrong four relations")
    return {"schema": "kshift129-four-arm-figure-data-v1", "source": SOURCE, "host": "129", "arms": arms,
            "pairs": pairs, "recipe_deltas": recipe_deltas, "execution_provenance": receipts,
            "scientific_review": {"binding": review_plan["binding"], "status": review["status"]},
            "bindings": sorted(sources.values(), key=lambda row: row["path"]),
            "boundaries": {"root_cause": "UNRESOLVED", "steps": 3, "ranks_per_run": 8,
                           "independent_repeat_count": 1, "all_model_coverage": False,
                           "native_numeric_recomputed_here": False,
                           "native_numeric_source": "Pinned full31 comparisons and independent raw-native review",
                           "global_input_hash": "Unordered component multiset, not source-DP/token UID proof",
                           "kernel_flag_consumption_verified": False,
                           "performance_sample": False, "acceptance200": "NOT_EVALUATED"}}


def validate_data(data: dict) -> None:
    """Validate portable numeric dimensions without claiming to reverify absent private raw files."""
    require(data["schema"] == "kshift129-four-arm-figure-data-v1" and set(data["arms"]) == set(ARM_KEYS)
            and set(data["pairs"]) == set(PAIR_KEYS), "Wrong portable study")
    for arm in data["arms"].values():
        require(arm["steps"] == [1, 2, 3] and arm["rank_count"] == 8
                and all(len(values) == 3 and all(math.isfinite(value) for value in values)
                        for values in arm["scalars"].values()), "Incomplete portable scalars")
    for pair in data["pairs"].values():
        require(len(pair["roles"]) == 31, "Incomplete portable native scope")


def summarize(data: dict) -> dict:
    """Compute only arithmetic differences of recorded scalar values and native report counts."""
    result = {"schema": "kshift129-four-arm-machine-summary-v1", "source": SOURCE,
              "scientific_review": data["scientific_review"], "boundaries": data["boundaries"],
              "logged_loss_coefficients": LOSS_COEFFICIENTS,
              "coefficient_recovery_scope": "Division rescales logged scalar differences, not pre-multiply tensors",
              "arms": {key: {"case": row["case"], "scalars": row["scalars"]} for key, row in data["arms"].items()},
              "relations": {}}
    for key, pair in data["pairs"].items():
        left, right = data["arms"][pair["left"]], data["arms"][pair["right"]]
        differences = {name: [second - first for first, second in zip(left["scalars"][name], right["scalars"][name])]
                       for name in SCALARS}
        result["relations"][key] = {"left": pair["left"], "right": pair["right"], "signed_scalar_delta": differences,
                                    "max_abs_scalar_delta": {name: max(map(abs, values))
                                                             for name, values in differences.items()},
                                    "coefficient_recovered_max_abs": {
                                        name: max(map(abs, differences[name])) / coefficient
                                        for name, coefficient in LOSS_COEFFICIENTS.items()},
                                    "numeric_nonzero_roles": pair["numeric_nonzero_roles"],
                                    "byte_mismatch_roles": pair["byte_mismatch_roles"],
                                    "canonical": pair["canonical"]}
    return result


def render(data: dict, output: Path) -> None:
    """Draw exact zeros as zero/gray; apply logarithmic color only to strictly positive native differences."""
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "DejaVu Sans"], "axes.unicode_minus": False,
                         "font.size": 10.5, "figure.facecolor": "white", "axes.spines.top": False,
                         "axes.spines.right": False, "svg.hashsalt": "kshift129-four-arm-v1"})
    steps = [1, 2, 3]
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.9), layout="constrained")
    labels = ("未设｜FSDP8", "未设｜EP4", "设1｜FSDP8", "设1｜EP4")
    for key, label, color, marker in zip(ARM_KEYS, labels, ("#315a98", "#bb3b38", "#24816c", "#b26d1d"),
                                        ("o", "x", "s", "+")):
        axes[0].plot(steps, data["arms"][key]["scalars"]["lm"], color=color, marker=marker, label=label)
    axes[0].set(xlabel="训练步", ylabel="主 LM loss", xticks=steps)
    axes[0].legend(frameon=False, fontsize=9)
    for name, label, color, offset in (("off_pair", "变量未设", "#315a98", -.13),
                                        ("on_pair", "变量设1", "#bb3b38", .13)):
        pair = data["pairs"][name]
        delta = np.abs(np.array(data["arms"][pair["right"]]["scalars"]["lm"])
                       - np.array(data["arms"][pair["left"]]["scalars"]["lm"]))
        positions = np.array(steps) + offset
        axes[1].bar(positions, delta, width=.24, color=color, label=label)
        for position, value in zip(positions, delta):
            axes[1].annotate(f"{value:.9f}" if value else "0", (position, value),
                             xytext=(0, 6), textcoords="offset points", ha="center", fontsize=8.8)
    maximum = max(abs(value) for key in ("off_pair", "on_pair")
                  for value in summarize(data)["relations"][key]["signed_scalar_delta"]["lm"])
    axes[1].set(xlabel="训练步", ylabel="同环境 EP4 与 FSDP 主 LM 绝对差", xticks=steps,
                ylim=(0, maximum * 1.28 if maximum else 1))
    axes[1].legend(frameon=False)
    figure.suptitle("129 同机四臂：CLOSE_MATMUL_K_SHIFT 未设 / 设1\n仅3步，每次8 ranks；不是8次独立重复或200步验收", fontsize=14)
    figure.savefig(output / "loss-four-arms.png", dpi=180)
    figure.savefig(output / "loss-four-arms.svg", metadata={"Date": None})
    plt.close(figure)

    rows = [(target, stage) for target in TARGETS for stage in STAGES]
    values = np.array([[next(row["max_abs"] for row in data["pairs"][pair]["roles"]
                            if (row["target"], row["stage"]) == (target, stage)) for pair in PAIR_KEYS]
                       for target, stage in rows])
    positive = values[values > 0]
    require(positive.size > 0, "This study has no positive native differences to color")
    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad("#edf0f2")
    figure, axis = plt.subplots(figsize=(12.8, 9.2), layout="constrained")
    image = axis.imshow(np.ma.masked_equal(values, 0), aspect="auto", cmap=cmap,
                        norm=colors.LogNorm(vmin=float(positive.min()), vmax=float(positive.max())))
    for row_index, (target, stage) in enumerate(rows):
        for column, pair in enumerate(PAIR_KEYS):
            item = next(row for row in data["pairs"][pair]["roles"]
                        if (row["target"], row["stage"]) == (target, stage))
            value = item["max_abs"]
            label = "0" if value == 0 else f"{value:.3g}"
            color = "white" if value > 0 and image.norm(value) > .68 else "#17212b"
            axis.text(column, row_index, f"{label}\n{item['nonzero_count']:,}/{item['numel']:,}",
                      ha="center", va="center", color=color, fontsize=9)
    axis.set(xticks=range(4), xticklabels=PAIR_LABELS,
             yticks=range(len(rows)), yticklabels=[target + " · " + STAGE_LABELS[STAGES.index(stage)]
                                                  for target, stage in rows])
    axis.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    for boundary in (3.5, 7.5):
        axis.axhline(boundary, color="white", linewidth=3)
    colorbar = figure.colorbar(image, ax=axis, shrink=.75, pad=.02)
    colorbar.set_label("最大绝对差：仅正数采用对数颜色；灰格是精确数值0")
    figure.suptitle("第1步选定窗口：G0、裁剪后梯度与权重\n每格：max|Δ|；非零元素数 / 窗口总数；未重放计算", fontsize=14)
    axis.set_xlabel("D：全5120；E0：仅expert0全5120×1536；Q：仅前192×1536；G0为归约后、裁剪前", labelpad=12)
    figure.savefig(output / "native-boundaries.png", dpi=180)
    figure.savefig(output / "native-boundaries.svg", metadata={"Date": None})
    plt.close(figure)


def main() -> None:
    """Write new report outputs only; never overwrite evidence or previous generated figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-data", type=Path, help="Portable plot regeneration; no private raw verification")
    parser.add_argument("--output", type=Path, default=HERE)
    args = parser.parse_args()
    output = args.output.resolve()
    require(not any((output / name).exists() for name in OUTPUT_NAMES), "Refuse existing output artifact")
    data = json.loads(args.from_data.read_bytes()) if args.from_data else collect_raw()
    validate_data(data)
    output.mkdir(parents=True, exist_ok=True)
    render(data, output)
    for name, value in (("figure-data.json", data), ("machine-summary.json", summarize(data)),
                         ("execution-provenance.json", data["execution_provenance"])):
        with (output / name).open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
    print(json.dumps({"source_mode": "portable_data_only" if args.from_data else "pinned_local_text_and_reports",
                      "steps": 3, "runs": 4, "ranks_per_run": 8, "native_roles_per_relation": 31,
                      "full_native_numeric_recomputation": False, "acceptance200": "NOT_EVALUATED"}))


if __name__ == "__main__":
    main()
