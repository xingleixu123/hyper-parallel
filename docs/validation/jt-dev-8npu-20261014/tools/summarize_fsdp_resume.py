# Copyright 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Summarize a completed resume without substituting for an independent comparison."""

import argparse
import json
import math
import statistics
from pathlib import Path

from compare_runs import LOSSES, read_rows
from strict_compare_runs import _validate_output, _verify_manifest


def main() -> None:
    """Validate the recovered boundary and summarize all eight rank traces."""
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("resumed", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    _validate_output(args.output, args.baseline, args.resumed, (args.manifest,))
    receipt_path = args.resumed.parent.parent / "scheduling" / (args.resumed.name + ".receipt.json")
    receipt = json.loads(receipt_path.read_text())
    problems = _verify_manifest(args.resumed, args.manifest)
    if receipt.get("exit_code") != 0 or not receipt.get("released") or receipt.get("reason"):
        problems.append("Execution was not cleanly completed and released")
    all_rows = []
    boundaries = []
    fields = (*LOSSES, "training/grad_norm", "data/consumed_samples", "data/consumed_tokens")
    for rank in range(8):
        rows = read_rows(args.resumed, rank)
        all_rows.append(rows)
        start = json.loads((args.resumed / f"start-rank{rank}.json").read_text())
        complete = json.loads((args.resumed / f"complete-rank{rank}.json").read_text())
        if sorted(rows) != list(range(201, 401)) or start["global_step"] != 200 or complete["global_step"] != 400:
            raise ValueError(f"Incomplete resume boundary on rank {rank}")
        previous = read_rows(args.baseline, rank)[200]
        for step, row in sorted(rows.items()):
            if any(not math.isfinite(value) for value in row.values() if isinstance(value, (int, float))):
                problems.append(f"Nonfinite metric: rank {rank}, step {step}")
            identity = row["input_identity"]
            if row["rank"] != rank or identity["rank_count"] != 8:
                problems.append(f"Invalid rank identity: rank {rank}, step {step}")
            if identity["global_sha256"] != all_rows[0][step]["input_identity"]["global_sha256"]:
                problems.append(f"Cross-rank input identity differs: rank {rank}, step {step}")
            for suffix in ("samples", "tokens"):
                current = row[f"data/consumed_{suffix}"]
                increment = row[f"data/step_{suffix}"]
                if increment < 0 or current != previous[f"data/consumed_{suffix}"] + increment:
                    problems.append(f"Counter discontinuity: rank {rank}, step {step}, {suffix}")
            previous = row
        boundaries.append({"rank": rank, "first": {key: rows[201][key] for key in fields},
                           "last": {key: rows[400][key] for key in fields}})
    rows_flat = [row for rows in all_rows for row in rows.values()]
    report = {
        "case": args.resumed.name,
        "status": "INTEGRITY_FAILED" if problems else "RUN_COMPLETE_REFERENCE_COMPARISON_PENDING",
        "precision_verdict": "UNVERIFIED",
        "scope": "运行完整性与恢复边界核验；不是独立连续400步对照，也不证明性能收益",
        "problems": problems,
        "source": receipt["source"], "image": receipt["image"], "container_id": receipt["container_id"],
        "start_step": 200, "end_step": 400, "steps_per_rank": [len(rows) for rows in all_rows],
        "exit_code": receipt["exit_code"], "released": receipt["released"],
        "remaining_owned_pids": receipt["remaining_owned_pids"],
        "duration_seconds": receipt["duration_seconds"], "boundaries": boundaries,
        "max_rank_spread": {key: max(max(rows[step][key] for rows in all_rows)
                                     - min(rows[step][key] for rows in all_rows)
                                     for step in range(201, 401)) for key in fields},
        "median_step_seconds_rank0_excluding_first20": statistics.median(
            all_rows[0][step]["performance/step_time"] for step in range(221, 401)),
        "max_rank_step_median_seconds_excluding_first20": statistics.median(
            max(rows[step]["performance/step_time"] for rows in all_rows) for step in range(221, 401)),
        "peak_allocated_gib": max(row["memory/device_max_allocated_gb"] for row in rows_flat),
        "peak_reserved_gib": max(row["memory/device_max_reserved_gb"] for row in rows_flat),
    }
    args.output.write_bytes((json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode())
    print(json.dumps({key: value for key, value in report.items() if key != "boundaries"}, ensure_ascii=False))
    raise SystemExit(bool(problems))


if __name__ == "__main__":
    main()
