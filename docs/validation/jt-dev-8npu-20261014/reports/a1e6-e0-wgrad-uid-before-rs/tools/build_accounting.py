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

"""Build the small, source-bound E0 accounting table without reading device tensors."""

import csv
import hashlib
import io
import json
import math
from pathlib import Path


REPORT = Path(__file__).resolve().parents[1]
ROOT = REPORT
EVIDENCE = ROOT / "evidence/a1e6-e0-wgrad-129-causal-v3"
REVIEW = EVIDENCE / "129-global-uid-native-independent-review-v1.json"
REVIEW_SHA = "03fb929fb7d241e5850499292f411a982c4dd5eda70eed9dbe6fed75a9b9411e"
LM = "training/foundation_loss/lm"
CASES = {
    "FSDP8": ("r129-a1e6-e0-wgrad-fsdp8-3-v3", "r129-a1e6-fsdp8-0-200-v3"),
    "EP4": ("r129-a1e6-e0-wgrad-ep4-3-v3", "r129-a1e6-ep4-200-v2"),
}


def binding(path: Path) -> dict:
    """Identify the complete local source bytes with a portable task-relative path."""
    raw = path.read_bytes()
    return {"path": path.relative_to(ROOT).as_posix(), "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest()}


def metrics(path: Path) -> dict:
    """Read one existing complete metric log, requiring unique steps."""
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if value["step"] in values:
            raise ValueError("Repeated metric step")
        values[value["step"]] = value
    return values


def save_new(path: Path, raw: bytes) -> None:
    """Refuse to replace any previous data artifact."""
    with path.open("xb") as handle:
        handle.write(raw)


def main() -> None:
    """Extract fair source-DP groups and the actual three-step main-LM window."""
    if binding(REVIEW)["sha256"] != REVIEW_SHA:
        raise ValueError("Frozen scientific review changed")
    review = json.loads(REVIEW.read_bytes())
    if review["status"] != "PASS_ACTUAL_UID_NATIVE_X_DY_AND_FAIR_SCOPE_LOCAL_ACCOUNTING_ONLY":
        raise ValueError("Scientific prerequisite is not passed")
    coverage = review["identity_and_coverage"]
    if coverage["UID_rows"] != 21714 or coverage["padding_rows_included"] != 16:
        raise ValueError("Selected UID coverage changed")
    if any(coverage["native_element_encoding_differences"].values()):
        raise ValueError("Original BF16 encodings differ")
    groups = []
    for group in review["fair_scope_accounting"]:
        reference = group["common_offline_products_fsum_f64"]
        fsdp = group["FSDP_native_partial_sum_f64"]
        ep = group["EP_native_local_GMM_post_f32"]
        if not group["exact_same_original_BF16_products"]:
            raise ValueError("Different UID/product domain")
        row = {**group, "FSDP_error_div8": (fsdp - reference) / 8,
               "EP_error_div8": (ep - reference) / 8,
               "EP_minus_FSDP_div8": (ep - fsdp) / 8,
               "FSDP_error_div8_units_1e9": (fsdp - reference) / 8 / 1e-9,
               "EP_error_div8_units_1e9": (ep - reference) / 8 / 1e-9}
        groups.append(row)
    sources = [binding(REVIEW)]
    references = {}
    for arm, (case, reference) in CASES.items():
        references[arm] = []
        for rank in range(8):
            actual_path = ROOT / "runs" / case / f"metrics-rank{rank}.jsonl"
            reference_path = ROOT / "runs" / reference / f"metrics-rank{rank}.jsonl"
            sources.extend((binding(actual_path), binding(reference_path)))
            actual, control = metrics(actual_path), metrics(reference_path)
            if set(actual) != {1, 2, 3} or set(control) != set(range(1, 201)):
                raise ValueError("Incomplete three-step or 200-step metric window")
            if any(actual[step]["rank"] != rank for step in actual):
                raise ValueError("Metric rank differs")
            references[arm].append((actual, control))
    steps = []
    for step in range(1, 4):
        values = {arm: [pair[0][step][LM] for pair in refs] for arm, refs in references.items()}
        residuals = {arm: max(abs(pair[0][step][LM] - pair[1][step][LM]) for pair in refs)
                     for arm, refs in references.items()}
        if any(residuals.values()):
            raise ValueError("Observer changes its own main-LM prefix")
        if any(not math.isfinite(value) for arm in values.values() for value in arm):
            raise ValueError("Nonfinite main-LM metric")
        row = {"step": step, "FSDP8_rank0_LM": values["FSDP8"][0], "EP4_rank0_LM": values["EP4"][0],
               "FSDP8_rank_range": [min(values["FSDP8"]), max(values["FSDP8"])],
               "EP4_rank_range": [min(values["EP4"]), max(values["EP4"])],
               "cross_topology_rankmax_abs": max(abs(a - b) for a, b in zip(values["FSDP8"], values["EP4"])),
               "own_FSDP8_200_prefix_rankmax_abs": residuals["FSDP8"],
               "own_EP4_200_prefix_rankmax_abs": residuals["EP4"]}
        steps.append(row)
    data = {
        "schema": "e0-before-rs-report-accounting-v1", "source": review["inputs"][1]["sha256"],
        "product_commit": "544526f7db8342c607244fd41009b2a74adb84c8", "host": "129",
        "target": "model.layers.1.mlp.experts.down_proj", "coordinate": {"expert": 0, "row": 306, "column": 58},
        "step": 1, "micro": 0, "UID_rows": 21714, "padding_rows": 16, "source_dp_slots": list(range(8)),
        "native_element_encoding_differences": coverage["native_element_encoding_differences"],
        "fair_source_DP_groups": groups, "main_LM_first3": steps,
        "own_prefix_comparisons_each_arm": 24,
        "units": {"native_byte_differences": "BF16 element encodings unequal count, not bytes",
                  "error": "signed offline error versus same-UID BF16 product f64 fsum, uniformly divided by8",
                  "plot_axis": "error / 1e-9"},
        "limits": ["Offline f64 sums are not NPU GMM/HCCL addition-order simulation.",
                   "Logged global scalars across8ranks are not8independent measurements.",
                   "Three steps do not accept200steps,full gradients,performance or a unique product/kernel cause."],
        "sources": sources,
    }
    # This SHA identifies the contract, not the product commit; keep both fields explicit.
    data["contract_sha256"] = data.pop("source")
    destination = REPORT / "data"
    destination.mkdir(exist_ok=True)
    save_new(destination / "accounting.json", (json.dumps(data, ensure_ascii=False, indent=2) + "\n").encode())
    output = io.StringIO(newline="")
    columns = ["source_dp_slots", "UIDs", "common_offline_products_fsum_f64", "FSDP_native_partial_sum_f64",
               "EP_native_local_GMM_post_f32", "EP_native_minus_FSDP_partial_sum_f64", "FSDP_error_div8",
               "EP_error_div8", "EP_minus_FSDP_div8"]
    writer = csv.DictWriter(output, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for group in groups:
        writer.writerow({key: group[key] for key in columns})
    save_new(destination / "accounting.csv", output.getvalue().encode())
    output = io.StringIO(newline="")
    columns = [key for key in steps[0] if not key.endswith("rank_range")]
    writer = csv.DictWriter(output, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows({key: row[key] for key in columns} for row in steps)
    save_new(destination / "main-lm-first3.csv", output.getvalue().encode())
    print(json.dumps({"status": "BUILT_REAL_DATA", "groups": len(groups), "steps": len(steps),
                      "maximum_cross_topology_LM": max(row["cross_topology_rankmax_abs"] for row in steps)}))


if __name__ == "__main__":
    main()
