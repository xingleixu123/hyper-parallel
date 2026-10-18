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

"""Summarize one completed pinned matrix run from existing all-rank audits only."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re


def main() -> None:
    """Write a new summary without modifying run evidence or invoking a device."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case")
    args = parser.parse_args()
    match = re.fullmatch(r"r129-a1e6-((?:tp[248])?(?:cp[24])?(?:ep[248])?)-200-v2", args.case)
    if not match or not match[1]:
        raise ValueError("Require one existing fixed 129 topology case")
    root = Path(__file__).resolve().parents[1]
    evidence, tag = root / "evidence", "a1e6-" + match[1] + "200-v2"
    strict_path, resource_path = evidence / (tag + "-strict.json"), evidence / (tag + "-resources.json")
    off_path = evidence / "a1e6-fsdp8-continuous-first200-v2-resources.json"
    strict, resource, off = [json.loads(path.read_bytes()) for path in (strict_path, resource_path, off_path)]
    if (strict["candidate"] != args.case or resource["case"] != args.case or strict["steps"] != [1, 200]
            or strict["ranks"] != 8 or strict["findings"] or strict["input_mismatch_count"]
            or strict["initial_mismatches"]):
        raise ValueError("Require complete200x8 fixed-recipe integrity with no input or initial mismatch")
    source = "544526f7db8342c607244fd41009b2a74adb84c8"
    if any(value["commit"] != source or value["patch_sha256"] != hashlib.sha256(b"").hexdigest()
           for value in strict["sources"].values()):
        raise ValueError("Require the unchanged clean fixed source")
    reference = strict["reference"]
    first = [json.loads((root / "runs" / case / "metrics-rank0.jsonl").read_text().splitlines()[0])
             for case in (reference, args.case)]
    lm = strict["errors"]["training/foundation_loss/lm"]
    paths = [strict_path, resource_path, evidence / ("129-" + tag + "-evidence-manifest.json"),
             evidence / "129-a1e6-fsdp8-200-v3-evidence-manifest.json", off_path]
    result = {
        "schema": tag + "-summary", "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": source, "candidate": args.case, "reference": reference, "steps": [1, 200], "ranks": 8,
        "main_lm_status": "PASS" if lm["max_abs"] <= 0.005 else "FAIL",
        "strict_all_component_status": strict["status"], "root_cause_status": "UNRESOLVED", "main_lm": lm,
        "main_lm_over_005_steps": [i + 1 for i, value in
                                   enumerate(strict["step_errors"]["training/foundation_loss/lm"]) if value > 0.005],
        "first_step_lm_abs": abs(first[0]["training/foundation_loss/lm"]
                                  - first[1]["training/foundation_loss/lm"]),
        "errors": strict["errors"], "unweighted_diagnostics": strict["unweighted_error_diagnostics"],
        "integrity": {"findings": [], "input_mismatch_count": 0, "cpu_initial_mismatches": []},
        "relative_to_off": {
            "median_time_increase_percent": 100 * (resource["median_step_seconds"] / off["median_step_seconds"] - 1),
            "supervised_throughput_change_percent": 100 * (
                resource["supervised_tokens_per_second"] / off["supervised_tokens_per_second"] - 1),
            "allocated_saved_gib": off["allocated_gib"] - resource["allocated_gib"],
            "allocated_saved_percent": 100 * (1 - resource["allocated_gib"] / off["allocated_gib"]),
            "reserved_saved_gib": off["reserved_gib"] - resource["reserved_gib"],
        },
        "inputs": {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        "limitations": [
            "Global scalar logs are not independent local losses; gradient norm is not tensor identity.",
            "CPU initial is after JT MLA replacement but before device conversion/TP/FSDP; not full NPU state.",
            "Single-run allrankMAX callback timing includes common evidence overhead; allocator peaks not physicalHBM.",
        ],
    }
    output = evidence / (tag + "-summary.json")
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({key: result[key] for key in (
        "candidate", "main_lm_status", "main_lm", "main_lm_over_005_steps", "first_step_lm_abs", "relative_to_off")}))


if __name__ == "__main__":
    main()
