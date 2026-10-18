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

"""Check an interrupted scalar prefix without a full-run verdict or resource comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


SCALARS = (
    "training/total_loss", "training/foundation_loss/lm", "training/foundation_loss/mtp",
    "training/foundation_loss/aux", "training/grad_norm",
)
COUNTERS = ("data/step_tokens", "data/consumed_tokens", "data/step_samples", "data/consumed_samples")
ABSENT_ARTIFACTS = (
    "EXIT_CODE", *(f"complete-rank{rank}.json" for rank in range(8)),
    *(f"final-memory-rank{rank}.json" for rank in range(8)),
)


def read_rank(directory: Path, rank: int) -> tuple[dict[int, dict[str, Any]], str]:
    """Read one byte snapshot, rejecting duplicates, foreign ranks, and nonpositive steps."""
    raw = (directory / f"metrics-rank{rank}.jsonl").read_bytes()
    records = [json.loads(line) for line in raw.splitlines()]
    if not records:
        raise ValueError(f"Empty metrics: {directory.name}, rank {rank}")
    rows = {}
    for row in records:
        step = row["step"]
        if type(step) is not int or step < 1 or row["rank"] != rank or step in rows:
            raise ValueError(f"Invalid/duplicate step or rank: {directory.name}, rank {rank}")
        for key in (*SCALARS, *COUNTERS, "training/lr"):
            value = row[key]
            if type(value) not in (float, int) or not math.isfinite(value):
                raise ValueError(f"Nonfinite/nonnumeric {key}: rank {rank}, step {step}")
        digest = row["input_identity"]["global_sha256"]
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"Invalid global input digest: rank {rank}, step {step}")
        rows[step] = row
    return rows, hashlib.sha256(raw).hexdigest()


def check_prefix(reference: Path, candidate: Path, last_step: int) -> dict[str, Any]:
    """Validate exactly eight interrupted prefixes and report only observed scalar differences.

    Args:
        reference: Directory containing at least the requested scalar prefix.
        candidate: Interrupted directory with no completion/exit/final-memory artifacts.
        last_step: Required contiguous prefix endpoint, strictly less than 200.

    Returns:
        Coverage, byte digests, scalar diagnostics, and explicit incomplete-run scope.

    Raises:
        ValueError: A complete-run marker, invalid record, or incorrect window is present.
    """
    if not 1 <= last_step < 200 or reference.resolve() == candidate.resolve():
        raise ValueError("An independent reference and a positive prefix shorter than 200 are required")
    present = [name for name in ABSENT_ARTIFACTS if (candidate / name).exists()]
    if present:
        raise ValueError(f"Interrupted-prefix checker requires absent end artifacts: {present}")
    steps = list(range(1, last_step + 1))
    errors = {key: {"max_abs": 0.0, "step": 1, "rank": 0} for key in SCALARS}
    coverage = {}
    source_hashes = {}
    input_mismatches = counter_mismatches = lr_mismatches = 0
    initial_mismatches = []
    reference_initial = json.loads((reference / "initial-state-rank0.json").read_bytes())
    for rank in range(8):
        expected, ref_sha = read_rank(reference, rank)
        actual, actual_sha = read_rank(candidate, rank)
        if sorted(actual) != steps or any(step not in expected for step in steps):
            raise ValueError(f"Incorrect prefix coverage: rank {rank}")
        coverage[str(rank)] = {"records": len(actual), "first_step": 1, "last_step": last_step, "contiguous": True}
        source_hashes[str(rank)] = {"reference": ref_sha, "candidate": actual_sha}
        for directory in (reference, candidate):
            if json.loads((directory / f"initial-state-rank{rank}.json").read_bytes()) != reference_initial:
                initial_mismatches.append([directory.name, rank])
        for step in steps:
            left, right = expected[step], actual[step]
            input_mismatches += left["input_identity"]["global_sha256"] != right["input_identity"]["global_sha256"]
            lr_mismatches += left["training/lr"] != right["training/lr"]
            counter_mismatches += sum(left[key] != right[key] for key in COUNTERS)
            for key, maximum in errors.items():
                difference = abs(left[key] - right[key])
                if not math.isfinite(difference):
                    raise ValueError(f"Nonfinite scalar difference: rank {rank}, step {step}, {key}")
                if difference > maximum["max_abs"]:
                    maximum.update(max_abs=difference, step=step, rank=rank)
    return {
        "schema": "interrupted-scalar-prefix-check-v1", "status": "INTERRUPTED_PREFIX_ONLY_NO_200_VERDICT",
        "reference": reference.name, "candidate": candidate.name, "steps": [1, last_step], "ranks": 8,
        "rank_coverage": coverage, "metrics_sha256": source_hashes, "scalar_errors": errors,
        "global_input_digest_mismatch_pairs": input_mismatches, "counter_mismatch_values": counter_mismatches,
        "LR_mismatch_pairs": lr_mismatches, "cpu_initial_mismatches": initial_mismatches,
        "observed_scalar_pairs_per_metric": 8 * last_step, "counter_pairs": 32 * last_step,
        "absent_raw_run_artifacts": list(ABSENT_ARTIFACTS), "main_lm_200_step_verdict": "NOT_EVALUATED_INCOMPLETE",
        "resource_comparison": "NOT_COMPUTED", "termination_cause": "NOT_INFERRED_FROM_METRICS",
        "scope": ("Logged scalar prefix and global input digests only; physical rank/microbatch identity is not "
                  "used to pair local tensors. No complete-run, gradient-tensor, or performance claim."),
    }


def main() -> None:
    """Write a fresh derived result outside either captured run directory."""
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--last-step", type=int, default=169)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if any(output == directory.resolve() or directory.resolve() in output.parents
           for directory in (args.reference, args.candidate)):
        raise ValueError("Derived output must be outside both captured run directories")
    result = check_prefix(args.reference, args.candidate, args.last_step)
    result["checker_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "steps", "ranks", "main_lm_200_step_verdict")}))


if __name__ == "__main__":
    main()
