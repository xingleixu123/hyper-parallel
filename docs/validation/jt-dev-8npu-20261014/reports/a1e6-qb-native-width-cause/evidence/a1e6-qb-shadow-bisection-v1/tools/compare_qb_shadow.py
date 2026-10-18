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

"""Match full native q_b shadow tiles to existing real FSDP/TP2 use-point evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from qb_shadow_contract import (
    MODULE, NATIVE_SHAPE, SCHEMA, SOURCE_COMMIT, TARGET_BATCH, frozen_contract,
    read_rank_files, validate_operand, validate_run,
)


def _blocks(identity: Mapping[str, Any]) -> dict[tuple[int, int, int, int], str]:
    keys = [tuple(block["coordinates"]) for block in identity["blocks"]]
    if len(keys) != len(set(keys)):
        raise ValueError("Repeated native identity tile")
    return {tuple(block["coordinates"]): block["sha256"] for block in identity["blocks"]}


def compare_to_existing(validated: Mapping[str, Any], reference: Mapping, candidate: Mapping, *,
                        cpu_fixture: bool = False) -> dict[str, Any]:
    """Compare digests only; old TP sparse samples cannot provide full numeric error."""
    target = validated["target"]
    batch = target["batch_sha256"]
    result = target["result"]
    base = {"schema": f"{SCHEMA}-comparison", "source_commit": SOURCE_COMMIT,
            "batch_sha256": batch, "module": MODULE, "stage": "output", "root_cause": "UNRESOLVED",
            "scope": "cpu-fixture-only" if cpu_fixture else "one-native-use-point-shadow",
            "own_three_step_prefix_transparency": "REQUIRES_SEPARATE_ALL_RANK_REVIEW",
            "full_TP_numeric_error": "NOT_AVAILABLE_FROM_OLD_HASH_AND_SPARSE_GRID_TRACE"}
    coherence = {}
    for stage, field in (("input", "input_native_identity"), ("weight", "weight_native_identity")):
        new = result[field]
        states = [old["complete"][(batch, MODULE, stage)] for old in (reference, candidate)]
        shape = list(new["shape"][-2:])
        blocks = _blocks(new)
        if any(state["logical_shape"] != shape or set(state["blocks"]) != set(blocks) for state in states):
            raise ValueError("Actual X/W logical coverage differs from old evidence")
        coherence[stage] = all(state["dtype"] == new["dtype"] and state["blocks"] == blocks for state in states)
    base["old_and_new_actual_X_W_native_equal"] = coherence
    if not all(coherence.values()):
        return {**base, "status": "ACTUAL_X_W_CHANGED_NO_N_INFERENCE"}
    full = result["calibration"]["operands"]["production"]
    old_full = reference["complete"][(batch, MODULE, "output")]
    full_blocks = _blocks(full)
    if set(old_full["blocks"]) != set(full_blocks):
        raise ValueError("Production output coverage differs from old FSDP")
    equal_original = old_full["dtype"] == full["dtype"] and old_full["blocks"] == full_blocks
    base["new_production_equals_old_FSDP_native"] = equal_original
    if not equal_original:
        return {**base, "status": "PRODUCTION_CHANGED_NO_N_INFERENCE"}
    if not validated["calibration_native_exact"]:
        return {**base, "status": "CALIBRATION_FAILED_NO_N_INFERENCE"}
    if not validated["half_repeats_native_exact"]:
        return {**base, "status": "HALF_REPEAT_UNSTABLE_NO_N_INFERENCE"}
    tp = candidate["complete"][(batch, MODULE, "output")]
    shape = tuple((full["logical_shape"][0], result["input_native_identity"]["shape"][-1],
                   full["logical_shape"][1]))
    if not cpu_fixture and (batch != TARGET_BATCH or shape != NATIVE_SHAPE):
        raise ValueError("Wrong native target/shape")
    comparisons = []
    merged = {"repeat1": {}, "repeat2": {}}
    for side in range(2):
        half = result["halves"][str(side)]
        begin, end = half["global_features"]
        for repeat in ("repeat1", "repeat2"):
            operand = half["operands"][repeat]
            blocks = validate_operand(operand, shape, begin, end, cpu_fixture=cpu_fixture)
            expected = {key: value for key, value in tp["blocks"].items() if begin <= key[2] < end}
            if set(expected) != set(blocks) or any(key in merged[repeat] for key in blocks):
                raise ValueError("Old TP/global N offset coverage is incomplete or overlapping")
            different = [key for key in blocks if blocks[key] != expected[key] or operand["dtype"] != tp["dtype"]]
            comparisons.append({"half": side, "repeat": repeat, "global_features": [begin, end],
                                "compared_native_blocks": len(blocks), "different_native_blocks": len(different),
                                "first_different_block": list(different[0]) if different else None})
            merged[repeat].update(blocks)
    if any(set(blocks) != set(tp["blocks"]) for blocks in merged.values()):
        raise ValueError("Two half shadows do not cover the complete old TP output")
    exact = all(row["different_native_blocks"] == 0 for row in comparisons)
    base["comparisons_to_old_TP2"] = comparisons
    base["full_element_shadow_vs_production"] = {
        side: half["comparisons"] for side, half in result["halves"].items()}
    base["status"] = ("N_WIDTH_SHADOW_REPRODUCES_OLD_TP2_NATIVE_BYTES" if exact else
                      "N_WIDTH_ALONE_DID_NOT_REPRODUCE_OLD_TP2_NATIVE_BYTES")
    base["limits"] = ["One actual batch and one projection only;"
                      "not an identified CANN kernel or repaired training run.",
                      "Full numeric metrics compare currently available shadow/production tensors.",
                      "Existing TP uses complete native hashes;its sparse grid isnot a full numeric maximum."]
    return base


def main() -> None:
    """Read fixed eight-rank sources and create a new, exclusive comparison receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shadow", type=Path, required=True)
    parser.add_argument("--reference-fsdp", type=Path, required=True)
    parser.add_argument("--existing-tp2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if any(output == folder.resolve() or folder.resolve() in output.parents
           for folder in (args.shadow, args.reference_fsdp, args.existing_tp2)):
        raise ValueError("Derived comparison must stay outside all three captured run directories")
    rows, shadow_sha = read_rank_files(args.shadow)
    validated = validate_run(rows)
    contract = frozen_contract()
    original, original_sha = contract.read_run(args.reference_fsdp)
    candidate, candidate_sha = contract.read_run(args.existing_tp2)
    reference = contract.canonicalize_run(original)
    candidate = contract.canonicalize_run(candidate)
    if reference["tp_size"] != 1 or candidate["tp_size"] != 2:
        raise ValueError("Expected original native FSDP8 and TP2 pair")
    result = compare_to_existing(validated, reference, candidate)
    result["input_sha256"] = {"shadow": shadow_sha, "reference_fsdp": original_sha, "existing_tp2": candidate_sha}
    result["tool_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({key: result[key] for key in ("status", "scope", "root_cause")}))


if __name__ == "__main__":
    main()
