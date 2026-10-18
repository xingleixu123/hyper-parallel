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

"""Fixed contracts and source loading for a new, FSDP-only q_b shadow case."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, Mapping


SCHEMA = "qb-shadow-v1"
SOURCE_COMMIT = "544526f7db8342c607244fd41009b2a74adb84c8"
MODULE = "model.layers.0.self_attn.q_b_proj"
TARGET_RANK = 2
TARGET_BATCH = "17c01af6623ce2bf41d8e786abba55318ea40f8595ba027303d4d797d8349113"
NATIVE_SHAPE = (4096, 1536, 24576)
CPU_SHAPE = (32, 8, 256)
TOKEN_BLOCK = 512
FEATURE_BLOCK = 128
WEIGHT_BLOCK = 128
WEIGHT_CPU_ROWS = 4096
FROZEN_FILES = {
    "attention0_contract": (
        "evidence/a1e6-attention0-forward-bisection-v1/tools/attention0_contract.py",
        "f3722477073c81c54844dafd18b2a5e2f08a9250757bec87cea50544cec33d4c",
    ),
    "native_attention0_trace": (
        "evidence/a1e6-attention0-forward-bisection-v1/tools/native_attention0_trace.py",
        "a60cd571e4f9fa10915994726c950499e817344a8ceabc9712cb4a45c766506f",
    ),
    "capture_attention0_forward_training": (
        "evidence/a1e6-attention0-forward-bisection-v1/tools/capture_attention0_forward_training.py",
        "03bdb08e4e358401bb0d0a82a8d94f017bfaa11090af2b00d5a2bb41e5f7d6e0",
    ),
    "native_b8f": (
        "evidence/a1e6-native-forward-bisection-v1/tools/native_forward_trace.py",
        "b8f16b43b9a8ce1b624d786979b986a4835f9d729aa08bf97fc1bbab17e81b6e",
    ),
}


def frozen_snapshots() -> dict[str, tuple[Path, bytes]]:
    """Verify every fixed dependency before executing any dependency module."""
    root = Path("/work") if os.name != "nt" else Path(__file__).resolve().parents[3]
    result = {}
    for name, (relative, digest) in FROZEN_FILES.items():
        path = root / relative
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError(f"Frozen dependency differs: {name}")
        result[name] = path, raw
    return result


def _load(name: str, path: Path, raw: bytes) -> ModuleType:
    existing = sys.modules.get(name)
    if existing is not None:
        if (Path(existing.__file__).resolve() != path.resolve()
                or getattr(existing, "_QB_SOURCE_SHA256", None) != hashlib.sha256(raw).hexdigest()):
            raise ValueError(f"Ambiguous frozen module: {name}")
        return existing
    module = ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    try:
        exec(compile(raw, str(path), "exec", dont_inherit=True), module.__dict__)
        module._QB_SOURCE_SHA256 = hashlib.sha256(raw).hexdigest()
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def frozen_contract() -> ModuleType:
    """Load the already-frozen pure JSON contract, without importing Torch."""
    snapshots = frozen_snapshots()
    return _load("attention0_contract", *snapshots["attention0_contract"])


def frozen_support(*, capture: bool = False) -> ModuleType:
    """Load pinned materialization/FIFO support and optionally native capture helpers."""
    snapshots = frozen_snapshots()
    _load("attention0_contract", *snapshots["attention0_contract"])
    support = _load("native_attention0_trace", *snapshots["native_attention0_trace"])
    if capture:
        return _load("capture_attention0_forward_training", *snapshots["capture_attention0_forward_training"])
    return support


def block_coordinates(rows: int, begin: int, end: int) -> list[tuple[int, int, int, int]]:
    """Return complete output tiles in the frozen attention0 global coordinate rule."""
    return [(row, min(row + TOKEN_BLOCK, rows), col, min(col + FEATURE_BLOCK, end))
            for row in range(0, rows, TOKEN_BLOCK) for col in range(begin, end, FEATURE_BLOCK)]


def validate_operand(value: Mapping[str, Any], shape: tuple[int, int, int], begin: int, end: int,
                     *, cpu_fixture: bool = False) -> dict[tuple[int, int, int, int], str]:
    """Require every global output tile exactly once, with a native dtype tag."""
    rows, _inner, width = shape
    if (value["logical_shape"] != [rows, width] or value["bounds"] != [0, rows, begin, end]
            or value["dtype"] != ("torch.float64" if cpu_fixture else "torch.bfloat16")):
        raise ValueError("Shadow output dtype, shape or global N offset differs")
    blocks = value["blocks"]
    keys = [tuple(block["coordinates"]) for block in blocks]
    if len(keys) != len(set(keys)) or set(keys) != set(block_coordinates(rows, begin, end)):
        raise ValueError("Incomplete, duplicate or misplaced native output tiles")
    if any(not isinstance(block["sha256"], str) or len(block["sha256"]) != 64
           or any(c not in "0123456789abcdef" for c in block["sha256"]) for block in blocks):
        raise ValueError("Invalid native tile digest")
    return {tuple(block["coordinates"]): block["sha256"] for block in blocks}


def validate_pair(value: Mapping[str, Any], numel: int, bounds: tuple[int, int, int, int]) -> None:
    """Validate all-element numerical counts independently of native byte equality."""
    integers = ("compared_numel", "left_finite", "right_finite", "both_finite", "nonzero_count")
    if any(type(value[key]) is not int for key in integers) or value["compared_numel"] != numel:
        raise ValueError("Invalid all-element pair coverage")
    if (not all(value[key] == numel for key in ("left_finite", "right_finite", "both_finite"))
            or value["finite"] is not True or not 0 <= value["nonzero_count"] <= numel):
        raise ValueError("Nonfinite or invalid all-element statistics")
    for key in ("max_abs", "squared_error_sum", "mse"):
        metric = value[key]
        if type(metric) not in (int, float) or not 0 <= metric < float("inf"):
            raise ValueError("Nonfinite or negative pair metric")
    if value["mse"] != value["squared_error_sum"] / numel:
        raise ValueError("MSE is not the full-element weighted mean")
    if value["nonzero_count"] == 0:
        if (any(value[key] != 0 for key in ("max_abs", "squared_error_sum", "mse"))
                or value["max_coordinate"] is not None or value["max_values"] is not None):
            raise ValueError("Zero unequal elements contradict nonzero maxima or coordinates")
    else:
        if any(value[key] <= 0 for key in ("max_abs", "squared_error_sum", "mse")):
            raise ValueError("Unequal finite values require positive full numeric metrics")
        coordinate, samples = value["max_coordinate"], value["max_values"]
        if (not isinstance(coordinate, list) or len(coordinate) != 2
                or any(type(item) is not int for item in coordinate)
                or not bounds[0] <= coordinate[0] < bounds[1] or not bounds[2] <= coordinate[1] < bounds[3]
                or not isinstance(samples, list) or len(samples) != 2
                or any(type(item) not in (int, float) or not math.isfinite(item) for item in samples)
                or abs(samples[0] - samples[1]) != value["max_abs"]):
            raise ValueError("Maximum coordinate/value does not prove the reported maximum")
        square = value["max_abs"]**2
        total = value["squared_error_sum"]
        if ((total < square and not math.isclose(total, square, rel_tol=1e-12))
                or (total > value["nonzero_count"] * square
                    and not math.isclose(total, value["nonzero_count"] * square, rel_tol=1e-12))):
            raise ValueError("Full squared error contradicts maximum/count bounds")


def _validate_native_pair(left: Mapping, right: Mapping, statistic: Mapping) -> None:
    different = sum(left[key] != right[key] for key in left)
    if different != statistic["different_native_blocks"]:
        raise ValueError("Native differing-block count contradicts complete digests")
    if not different and any(statistic[key] != 0 for key in ("nonzero_count", "max_abs", "squared_error_sum", "mse")):
        raise ValueError("Identical native bytes have contradictory full numeric statistics")


def validate_run(records: Mapping[int, list[dict]], *, cpu_fixture: bool = False) -> dict[str, Any]:
    """Require all eight three-step files and exactly one selected first-step shadow.

    This validates diagnostic completeness. It never claims NPU prefix transparency,
    an existing TP match, a repaired model, or a complete 200-step precision pass.
    """
    if set(records) != set(range(8)):
        raise ValueError("Exactly eight FSDP rank records are required")
    shape = CPU_SHAPE if cpu_fixture else NATIVE_SHAPE
    batches, target, fixture_identity = set(), None, None
    for rank, rows in records.items():
        metas = [row for row in rows if row["kind"] == "metadata"]
        if len(metas) != 1:
            raise ValueError("Missing/duplicate shadow metadata")
        meta = metas[0]
        topo = meta["topology"]
        if (meta["schema"] != SCHEMA or meta["source_commit"] != SOURCE_COMMIT or meta["rank"] != rank
                or meta["scope"] != ("cpu-fixture-only" if cpu_fixture else "native-candidate")
                or meta["shape"] != list(shape) or topo["rank"] != rank or topo["dp_rank"] != rank
                or topo["tp_size"] != 1 or topo["dp_size"] != 8 or topo["tp_rank"] != 0
                or topo["cp_size"] != 1 or topo["pp_size"] != 1 or topo["ep_size"] != 1
                or topo["sequence_parallel"] is not False or meta["target_rank"] != TARGET_RANK
                or meta["runtime_layout_proof"].get("verified") is not True
                or meta["frozen_inputs"] != {name: digest for name, (_path, digest) in FROZEN_FILES.items()}):
            raise ValueError("Mixed/unproven shadow runtime contract")
        if fixture_identity is None:
            fixture_identity = meta["target_batch"]
        if fixture_identity != meta["target_batch"]:
            raise ValueError("Mixed target identities")
        batch_rows = [row for row in rows if row["kind"] == "batch"]
        completed = [row for row in rows if row["kind"] == "step_complete"]
        runs = [row for row in rows if row["kind"] == "run_complete"]
        projections = [row for row in rows if row["kind"] == "projection"]
        if (len(batch_rows) != 1 or batch_rows[0]["micro"] != 0 or batch_rows[0]["step"] != 1
                or len(completed) != 1 or completed[0]["step"] != 1
                or completed[0]["micro_batches"] != 1 or len(runs) != 1 or runs[0]["completed_steps"] != 3
                or len(projections) != 2 or {row["stage"] for row in projections} != {"input", "output"}):
            raise ValueError("Missing/repeated first-step calls or three-step completion")
        batch = batch_rows[0]["batch_sha256"]
        if batch in batches or any(row["batch_sha256"] != batch for row in projections):
            raise ValueError("Ambiguous batch identity")
        if any(row["module"] != MODULE or row["step"] != 1 or row["micro"] != 0 for row in projections):
            raise ValueError("Projection FQN/stage is not bound to the actual first microbatch")
        batches.add(batch)
        results = [row for row in rows if row["kind"] == "shadow"]
        if rank == TARGET_RANK:
            if len(results) != 1 or batch != meta["target_batch"]:
                raise ValueError("The selected actual batch was not observed exactly once")
            target = results[0]
            if (target["batch_sha256"] != batch or target["step"] != 1 or target["micro"] != 0
                    or target["module"] != MODULE or target["stage"] != "output"):
                raise ValueError("Shadow result is not bound to the selected actual q_b output")
        elif results or batch == meta["target_batch"]:
            raise ValueError("Shadow ran on an unselected batch/rank")
        if not cpu_fixture and meta["target_batch"] != TARGET_BATCH:
            raise ValueError("Native target batch is not the fixed actual difference batch")
        if any(row.get("rank", rank) != rank for row in rows):
            raise ValueError("Mixed physical rank record")
        expected_status = results[0]["result"]["status"] if rank == TARGET_RANK else "NOT_SELECTED"
        if runs[0]["diagnostic_status"] != expected_status:
            raise ValueError("Completion diagnostic status contradicts the actual target result")
    if target is None:
        raise ValueError("Missing target shadow result")
    result = target["result"]
    for key in ("input_native_unchanged", "weight_native_unchanged", "production_native_unchanged",
                "rng_unchanged", "production_object_preserved", "requires_grad_unchanged",
                "tensor_versions_unchanged", "grad_enabled_unchanged", "module_weight_object_unchanged",
                "functional_linear_unchanged", "module_forward_unchanged"):
        if result["invariants"][key] is not True:
            raise ValueError(f"Shadow changed an original invariant: {key}")
    calibration = result["calibration"]
    if (set(calibration["operands"]) != {"production", "repeat"}
            or set(calibration["comparisons"]) != {"production_vs_repeat"}):
        raise ValueError("Unexpected calibration calls or operands")
    full = shape[0] * shape[2]
    validate_pair(calibration["comparisons"]["production_vs_repeat"], full, (0, shape[0], 0, shape[2]))
    left = validate_operand(calibration["operands"]["production"], shape, 0, shape[2], cpu_fixture=cpu_fixture)
    right = validate_operand(calibration["operands"]["repeat"], shape, 0, shape[2], cpu_fixture=cpu_fixture)
    original = result["production_native_identity_before_shadow"]
    if (result["production_identity_capture"] != "actual_output_hook_before_any_shadow_linear"
            or original["shape"] != [1, shape[0], shape[2]]
            or original["dtype"] != calibration["operands"]["production"]["dtype"]
            or original["blocks"] != calibration["operands"]["production"]["blocks"]):
        raise ValueError("Production before-shadow identity differs from the calibrated original output")
    _validate_native_pair(left, right, calibration["comparisons"]["production_vs_repeat"])
    exact = left == right
    if result["calibration_native_exact"] is not exact:
        raise ValueError("Calibration flag contradicts complete native tiles")
    if not exact:
        if result["halves"] or result["status"] != "CALIBRATION_FAILED_NO_N_INFERENCE":
            raise ValueError("Half-N ran despite failed native full-N calibration")
        return {"status": result["status"], "target": target, "calibration_native_exact": False}
    if set(result["halves"]) != {"0", "1"}:
        raise ValueError("Two global N halves are required after full-N calibration")
    stable = True
    for side in range(2):
        half = result["halves"][str(side)]
        begin, end = side * shape[2] // 2, (side + 1) * shape[2] // 2
        if half["global_features"] != [begin, end]:
            raise ValueError("Half-N global offsets differ")
        if set(half["comparisons"]) != {"production_vs_repeat1", "production_vs_repeat2", "repeat1_vs_repeat2"}:
            raise ValueError("Incomplete full-element half/repeat comparisons")
        for value in half["comparisons"].values():
            validate_pair(value, full // 2, (0, shape[0], begin, end))
        parts = {name: validate_operand(value, shape, begin, end, cpu_fixture=cpu_fixture)
                 for name, value in half["operands"].items()}
        if set(parts) != {"production", "repeat1", "repeat2"}:
            raise ValueError("Two native same-shape repeats and production slice are required")
        if parts["production"] != {key: digest for key, digest in left.items() if begin <= key[2] < end}:
            raise ValueError("Half production view differs from the original full output")
        for name, first, second in (("production_vs_repeat1", "production", "repeat1"),
                                    ("production_vs_repeat2", "production", "repeat2"),
                                    ("repeat1_vs_repeat2", "repeat1", "repeat2")):
            _validate_native_pair(parts[first], parts[second], half["comparisons"][name])
        stable &= parts["repeat1"] == parts["repeat2"]
    expected_status = "CORE_SHADOW_COMPLETE" if stable else "HALF_REPEAT_UNSTABLE_NO_N_INFERENCE"
    if result["status"] != expected_status:
        raise ValueError("Half repeat status contradicts native bytes")
    return {"status": result["status"], "target": target, "calibration_native_exact": True,
            "half_repeats_native_exact": stable}


def read_rank_files(directory: Path) -> tuple[dict[int, list[dict]], dict[str, str]]:
    """Hash and parse the same eight byte snapshots; never infer missing completion."""
    rows, digests = {}, {}
    for rank in range(8):
        path = directory / f"qb-shadow-rank{rank}.jsonl"
        raw = path.read_bytes()
        rows[rank] = [json.loads(line) for line in raw.splitlines()]
        digests[str(path)] = hashlib.sha256(raw).hexdigest()
    return rows, digests
