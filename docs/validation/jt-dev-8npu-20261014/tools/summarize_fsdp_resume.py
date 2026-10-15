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
import hashlib
import json
import math
import re
import statistics
from pathlib import Path

from compare_runs import LOSSES, read_rows
from strict_compare_runs import _execution, _recipe, _validate_output, _verify_manifest


def _verified_run(directory: Path, manifest: Path, first: int, last: int) -> tuple:
    """Bind a complete eight-rank run to its collection and execution identity."""
    problems = _verify_manifest(directory, manifest)
    if problems:
        raise ValueError("; ".join(problems))
    commit = (directory / "source-commit.txt").read_text().strip()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("A complete source SHA is required")
    source = {"commit": commit, "patch_sha256": hashlib.sha256((directory / "source.patch").read_bytes()).hexdigest()}
    _, problems = _execution(directory, source)
    if (directory / "EXIT_CODE").read_text().strip() != "0":
        problems.append("Recorded process exit was not zero")
    if problems:
        raise ValueError("; ".join(problems))
    rows = [read_rows(directory, rank) for rank in range(8)]
    for rank, values in enumerate(rows):
        start = json.loads((directory / f"start-rank{rank}.json").read_text())
        complete = json.loads((directory / f"complete-rank{rank}.json").read_text())
        if sorted(values) != list(range(first, last + 1)) or start["global_step"] != first - 1:
            raise ValueError(f"Incomplete run start or coverage: {directory.name}, rank {rank}")
        if complete["global_step"] != last:
            raise ValueError(f"Incomplete run completion: {directory.name}, rank {rank}")
    receipt = json.loads((directory.parent.parent / "scheduling" / (directory.name + ".receipt.json")).read_text())
    return rows, receipt, source, json.loads((directory / "config.resolved.json").read_text())


def summarize_resume(baseline: Path, resumed: Path, baseline_manifest: Path, resumed_manifest: Path) -> dict:
    """Validate both complete runs; keep precision unverified without an independent reference."""
    before, base_receipt, base_source, base_config = _verified_run(baseline, baseline_manifest, 1, 200)
    after, receipt, source, config = _verified_run(resumed, resumed_manifest, 201, 400)
    if baseline.resolve() == resumed.resolve() or base_receipt["container_id"] == receipt["container_id"]:
        raise ValueError("Separate baseline and resume executions are required")
    if base_source != source or base_receipt["image"] != receipt["image"] or _recipe(base_config) != _recipe(config):
        raise ValueError("Source, runtime image, or numerical recipe differs")
    for field in ("accelerator", "activation_checkpoint", "fsdp_config"):
        if base_config[field] != config[field]:
            raise ValueError(f"Resume configuration changed: {field}")
    for field in ("mtp_loss_factor", "moe_aux_loss_coeff"):
        coefficient = float(config["model"]["config"][field])
        if not math.isfinite(coefficient) or coefficient <= 0:
            raise ValueError("Loss coefficients must be positive and finite")
    problems = []
    all_rows = []
    boundaries = []
    fields = (*LOSSES, "training/grad_norm", "data/consumed_samples", "data/consumed_tokens")
    for rank in range(8):
        rows = after[rank]
        all_rows.append(rows)
        previous = {"data/consumed_samples": 0, "data/consumed_tokens": 0}
        chain = {**before[rank], **rows}
        for step, row in sorted(chain.items()):
            if any(not math.isfinite(value) for value in row.values() if isinstance(value, (int, float))):
                problems.append(f"Nonfinite metric: rank {rank}, step {step}")
            identity = row["input_identity"]
            if (row["rank"] != rank or identity["rank_count"] != 8
                    or not re.fullmatch(r"[0-9a-f]{64}", identity["global_sha256"])):
                problems.append(f"Invalid rank identity: rank {rank}, step {step}")
            reference_rows = before[0] if step <= 200 else after[0]
            if identity["global_sha256"] != reference_rows[step]["input_identity"]["global_sha256"]:
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
        "case": resumed.name,
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
    return report


def main() -> None:
    """Write a summary only outside the immutable input evidence."""
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("resumed", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    args = parser.parse_args()
    _validate_output(args.output, args.baseline, args.resumed, (args.baseline_manifest, args.manifest))
    report = summarize_resume(args.baseline, args.resumed, args.baseline_manifest, args.manifest)
    args.output.write_bytes((json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode())
    print(json.dumps({key: value for key, value in report.items() if key != "boundaries"}, ensure_ascii=False))
    raise SystemExit(bool(report["problems"]))


if __name__ == "__main__":
    main()
