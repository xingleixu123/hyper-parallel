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
"""Compare complete projection states; never add or equate different K contributions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from attention0_contract import EVENT_ORDER, SCHEMA, SOURCE_COMMIT, canonicalize_run, read_run


def compare_complete_states(reference: dict, candidate: dict) -> dict[str, Any]:
    """Compare already-validated canonical states by batch/FQN/stage/global coordinates.

    The numerical maximum covers only the saved deterministic sample grid.
    Complete block digests cover every true logical byte in their native dtype.
    Partial contributors are intentionally not compared across K partitions.
    """
    left, right = reference["complete"], candidate["complete"]
    if set(left) != set(right):
        raise ValueError("Runs do not contain the same actual batches and complete boundaries")
    results, first = [], None
    for module, stage in EVENT_ORDER:
        keys = sorted(key for key in left if key[1:] == (module, stage))
        if not keys:
            continue
        if len(keys) != 8:
            raise ValueError("Expected exactly eight logical states at each complete boundary")
        compared_blocks, different_blocks, compared_samples, different_samples = 0, 0, 0, 0
        grid_max, dtype_pairs, first_block = 0.0, set(), None
        unique_weight_tiles = set()
        for key in keys:
            first_state, second_state = left[key], right[key]
            if (first_state["logical_shape"] != second_state["logical_shape"]
                    or set(first_state["blocks"]) != set(second_state["blocks"])
                    or set(first_state["grid"]) != set(second_state["grid"])):
                raise ValueError("Global coordinate coverage differs between canonical runs")
            dtype_pairs.add((first_state["dtype"], second_state["dtype"]))
            for coordinates, digest in first_state["blocks"].items():
                compared_blocks += 1
                unique_weight_tiles.add(coordinates)
                if digest != second_state["blocks"][coordinates] or first_state["dtype"] != second_state["dtype"]:
                    different_blocks += 1
                    if first_block is None:
                        first_block = {"batch_sha256": key[0], "coordinates": list(coordinates)}
            for coordinates, value in first_state["grid"].items():
                compared_samples += 1
                difference = abs(value - second_state["grid"][coordinates])
                if difference and first_state["dtype"] == second_state["dtype"]:
                    row_index, column_index = coordinates
                    containing = [block for block in first_state["blocks"]
                                  if block[0] <= row_index < block[1] and block[2] <= column_index < block[3]]
                    if len(containing) != 1:
                        raise ValueError("Sample-grid coordinate does not belong to exactly one complete block")
                    block = containing[0]
                    if first_state["blocks"][block] == second_state["blocks"][block]:
                        raise ValueError("Identical native block bytes/dtype have contradictory numerical samples")
                different_samples += difference != 0
                grid_max = max(grid_max, difference)
        exact = different_blocks == 0
        row = {"module": module, "stage": stage, "logical_batches": len(keys),
               "status": "EXACT_NATIVE_BYTES" if exact else "NATIVE_BYTE_DIFFERENCE",
               "batch_block_comparisons": compared_blocks, "different_batch_blocks": different_blocks,
               "dtype_pairs": [list(pair) for pair in sorted(dtype_pairs)],
               "sample_grid_comparisons": compared_samples, "different_grid_samples": different_samples,
               "sample_grid_max_abs_difference": grid_max, "first_different_block": first_block}
        if stage == "weight":
            row["unique_logical_weight_tiles"] = len(unique_weight_tiles)
        results.append(row)
        if not exact and first is None:
            first = {"module": module, "stage": stage, "first_different_block": first_block}
    return {"schema": f"{SCHEMA}-comparison", "source_commit": SOURCE_COMMIT,
            "scope": "passive-use-point-projection-evidence", "logical_batches": 8,
            "reference_tp_size": reference["tp_size"], "candidate_tp_size": candidate["tp_size"],
            "status": "EXACT_COMPLETE_BOUNDARIES" if first is None else "COMPLETE_BOUNDARY_DIFFERENCES",
            "first_observed_complete_difference": first, "complete_boundaries": results,
            "o_proj_raw_contributors": {
                "status": "KEPT_SEPARATE_NOT_REDUCED_OR_PAIRED_AS_REPLICAS",
                "reference_states": len(reference["contributors"]),
                "candidate_states": len(candidate["contributors"]),
                "reference_k_partitions": sorted({tuple(key[-2:]) for key in reference["contributors"]}),
                "candidate_k_partitions": sorted({tuple(key[-2:]) for key in candidate["contributors"]})},
            "own_prefix_transparency": "REQUIRES_SEPARATE_THREE_STEP_SCALAR_INPUT_RUNTIME_REVIEW",
            "root_cause": "UNRESOLVED",
            "limits": ["Sample-grid numerical maxima are not full-tensor numerical maxima.",
                       "Native-byte differences locate observed boundaries and do not establish a causal kernel.",
                       "Raw o_proj contributions have different K partitions; "
                       "no offline sum or GEMM replay is performed."]}


def main() -> None:
    """Validate both eight-rank inputs and create a new comparison receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cpu-fixture", action="store_true",
                        help="Explicitly permit synthetic H128 fixtures; never native NPU evidence")
    args = parser.parse_args()
    reference_rows, reference_sha = read_run(args.reference)
    candidate_rows, candidate_sha = read_run(args.candidate)
    reference = canonicalize_run(reference_rows, cpu_fixture=args.cpu_fixture)
    candidate = canonicalize_run(candidate_rows, cpu_fixture=args.cpu_fixture)
    report = compare_complete_states(reference, candidate)
    report["scope"] = "cpu-fixture-only" if args.cpu_fixture else report["scope"]
    report["input_sha256"] = {"reference": reference_sha, "candidate": candidate_sha}
    report["tool_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"output": str(args.output), "status": report["status"],
                      "root_cause": report["root_cause"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
