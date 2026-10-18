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

"""Separate contracts for a target-only, autograd-enabled q_b padding short arm."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import sys
from types import ModuleType
from typing import Any, Mapping


SCHEMA = "qb-padding-causal-v2"
SOURCE_COMMIT = "544526f7db8342c607244fd41009b2a74adb84c8"
MODULE = "model.layers.0.self_attn.q_b_proj"
OWNER = "model.layers.0"
TARGET_BATCH = "17c01af6623ce2bf41d8e786abba55318ea40f8595ba027303d4d797d8349113"
TARGET_RANKS = frozenset((4, 5))
NATIVE_SHAPE = (4096, 1536, 24576)
CPU_SHAPE = (32, 128, 256)
FROZEN_FILES = {
    "qb_shadow_contract": (
        "evidence/a1e6-qb-shadow-bisection-v1/tools/qb_shadow_contract.py",
        "9f67ef1cf3c8983f3113d9de0ba3a4c8bc7d0db067b42552c76a643d54206e7d"),
    "native_qb_shadow": (
        "evidence/a1e6-qb-shadow-bisection-v1/tools/native_qb_shadow.py",
        "4b43fe902981394687fe54fb10e41373d1b8259c6f85267b292fcef5272a8222"),
    "attention0_contract": (
        "evidence/a1e6-attention0-forward-bisection-v1/tools/attention0_contract.py",
        "f3722477073c81c54844dafd18b2a5e2f08a9250757bec87cea50544cec33d4c"),
    "native_attention0_trace": (
        "evidence/a1e6-attention0-forward-bisection-v1/tools/native_attention0_trace.py",
        "a60cd571e4f9fa10915994726c950499e817344a8ceabc9712cb4a45c766506f"),
    "capture_attention0_forward_training": (
        "evidence/a1e6-attention0-forward-bisection-v1/tools/capture_attention0_forward_training.py",
        "03bdb08e4e358401bb0d0a82a8d94f017bfaa11090af2b00d5a2bb41e5f7d6e0"),
    "native_b8f": (
        "evidence/a1e6-native-forward-bisection-v1/tools/native_forward_trace.py",
        "b8f16b43b9a8ce1b624d786979b986a4835f9d729aa08bf97fc1bbab17e81b6e"),
}
SOURCE_FILES = {
    "rewriter": ("hyper_parallel/distributed/_builder/forward_rewriter.py",
                 "696acee412fd606c3abfd977435bd11d13fe4fb9f39c017a63acaaf523acb853"),
    "dtensor": ("hyper_parallel/core/dtensor/dtensor.py",
                "3236f59a15283a4fabb1493fb9f7c38d90ce4f9f6b491345cbdda6c17ab94763"),
    "scheduler": ("hyper_parallel/core/fully_shard/hsdp_scheduler.py",
                  "920d3c8591e1a2e5b8cdd543851ebcad704c9d0137e12688dd38944499b0b19a"),
}
INVARIANTS = (
    "input_native_unchanged", "weight_native_unchanged", "original_native_unchanged", "rng_unchanged",
    "input_object_preserved", "weight_object_restored", "same_live_compute_weight", "versions_unchanged",
    "original_graph_preserved", "effective_graph_present", "grad_mode_unchanged", "forward_unchanged",
)


def task_root() -> Path:
    """Return the exact mounted task root, without an environment fallback."""
    return Path("/work") if os.name != "nt" else Path(__file__).resolve().parents[3]


def source_snapshot(name: str) -> tuple[Path, bytes]:
    """Read the fixed source file once, for code/API proof and CPU AST fixtures."""
    relative, digest = SOURCE_FILES[name]
    base = Path("/workspace/hyper-parallel") if os.name != "nt" else task_root() / "fixes/latest-jt-validation-f637"
    path = base / relative
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError(f"Fixed source changed: {name}")
    return path, raw


def _load(name: str, path: Path, raw: bytes) -> ModuleType:
    existing = sys.modules.get(name)
    digest = hashlib.sha256(raw).hexdigest()
    if existing is not None:
        if (Path(existing.__file__).resolve() != path.resolve()
                or getattr(existing, "_PADDING_SOURCE_SHA256", None) != digest):
            raise ValueError(f"Ambiguous frozen padding support: {name}")
        return existing
    module = ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    try:
        exec(compile(raw, str(path), "exec", dont_inherit=True), module.__dict__)
        module._PADDING_SOURCE_SHA256 = digest
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def frozen_support(*, tensor: bool = False, capture: bool = False) -> ModuleType:
    """Verify all transitive task inputs before executing the exact cached raw."""
    snapshots = {}
    for name, (relative, digest) in FROZEN_FILES.items():
        path = task_root() / relative
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError(f"Frozen support differs: {name}")
        snapshots[name] = path, raw
    contract = _load("qb_shadow_contract", *snapshots["qb_shadow_contract"])
    if capture:
        return contract.frozen_support(capture=True)
    return _load("native_qb_shadow", *snapshots["native_qb_shadow"]) if tensor else contract


def region(stage: str, topology: Mapping[str, Any], *, cpu_fixture: bool = False) -> dict[str, Any]:
    """Derive global rows/features from a proven TP coordinate, never rank parity."""
    rows, inner, width = CPU_SHAPE if cpu_fixture else NATIVE_SHAPE
    tp = topology["tp_rank"]
    if topology["tp_size"] != 2 or type(tp) is not int or tp not in (0, 1):
        raise ValueError("Only the exact two-peer q_b layout is supported")
    begin, end = tp * width // 2, (tp + 1) * width // 2
    if stage == "weight":
        return {"logical_shape": [width, inner], "bounds": [begin, end, 0, inner],
                "physical_ndim": 2, "semantics": "weight"}
    if stage not in ("input", "output"):
        raise ValueError("Unknown q_b stage")
    return {"logical_shape": [rows, inner if stage == "input" else width],
            "bounds": [0, rows, 0, inner] if stage == "input" else [0, rows, begin, end],
            "physical_ndim": 3, "semantics": "complete"}


def _blocks(value: Mapping[str, Any], expected: Mapping[str, Any]) -> dict[tuple, str]:
    if any(value[key] != expected[key] for key in ("logical_shape", "bounds", "semantics")):
        raise ValueError("Wrong canonical global region")
    rb, re_, cb, ce = expected["bounds"]
    size = 128 if expected["semantics"] == "weight" else 512
    keys = {(row, min(row + size, re_), col, min(col + 128, ce))
            for row in range(rb, re_, size) for col in range(cb, ce, 128)}
    actual = [tuple(item["coordinates"]) for item in value["blocks"]]
    if len(actual) != len(set(actual)) or set(actual) != keys:
        raise ValueError("Missing, duplicate or overlapping native tiles")
    if any(not re.fullmatch(r"[a-f0-9]{64}", item["sha256"]) for item in value["blocks"]):
        raise ValueError("Invalid native digest")
    return {tuple(item["coordinates"]): item["sha256"] for item in value["blocks"]}


def _physical(value: Mapping[str, Any], stage: str, shape: tuple[int, int, int], *, cpu_fixture: bool) -> None:
    rows, inner, width = shape
    expected = [width // 2, inner] if stage == "weight" else [1, rows, inner if stage == "input" else width // 2]
    physical = value["physical"]
    size = 8 if cpu_fixture else 2
    product = 1
    for extent in expected:
        product *= extent
    stride = [inner, 1] if stage == "weight" else [rows * expected[-1], expected[-1], 1]
    if (physical["shape"] != expected or physical["stride"] != stride or physical["contiguous"] is not True
            or physical["storage_offset"] != 0 or physical["storage_nbytes"] != product * size
            or physical["dtype"] != value["dtype"] or not physical["data_ptr"]
            or physical["data_ptr"] != physical["storage_data_ptr"]):
        raise ValueError("Incomplete or noncompact saved physical tensor proof")


def validate_forward_route(proof: Mapping[str, Any]) -> None:
    """Bind the JSON proof to one of the two strict original Linear routes."""
    source = proof["torch_linear_source_identity"]
    identity = proof["forward_identity"]
    if (proof.get("verified") is not True or proof["original_linear"] != "torch.nn.Linear.forward"
            or source["module"] != "torch.nn.modules.linear" or source["qualname"] != "Linear.forward"
            or source["file_identity_checked"] is not True or source["source_bytes_sha_checked"] is not False
            or not isinstance(source["source_file"], str) or not source["source_file"]
            or source["source_file"] != source["code_filename"]
            or not isinstance(identity, (list, tuple)) or len(identity) != 2
            or any(type(value) is not int or value <= 0 for value in identity)):
        raise ValueError("Incomplete original Linear function/source identity proof")
    if proof["route"] == "bare_nn_linear":
        if (any(proof.get(key) is not True for key in (
                "same_bound_module", "class_forward_unchanged", "plain_installed_weight"))
                or type(proof.get("installed_weight_object_id")) is not int
                or proof["installed_weight_object_id"] <= 0):
            raise ValueError("Unproven direct-bound original plain-weight Linear")
    elif proof["route"] == "fixed_single_local_param":
        if (proof.get("rewriter_sha256") != SOURCE_FILES["rewriter"][1]
                or proof.get("local_compute_active") is not True or proof.get("closure_module_is_actual") is not True):
            raise ValueError("Unproven fixed single local-parameter adapter")
    else:
        raise ValueError("Unsupported original forward route")


def validate_run(records: Mapping[int, list[dict]], *, cpu_fixture: bool = False) -> dict[str, Any]:
    """Validate first-step routing/coverage and merge the actual target's two peers."""
    if set(records) != set(range(8)):
        raise ValueError("Exactly eight rank files are required")
    support = frozen_support()
    target = None
    arm = None
    coordinates, groups, selected = set(), {}, []
    merged = {name: {"blocks": {}, "dtype": None} for name in ("input", "weight", "original", "effective")}
    shape = CPU_SHAPE if cpu_fixture else NATIVE_SHAPE
    for rank, rows in records.items():
        if any(row.get("rank") != rank or row["kind"] not in (
                "metadata", "batch", "call", "intervention", "step_complete", "run_complete") for row in rows):
            raise ValueError("Mixed rank or unexpected/incomplete evidence event")
        metas = [row for row in rows if row["kind"] == "metadata"]
        complete = [row for row in rows if row["kind"] == "run_complete"]
        done = [row for row in rows if row["kind"] == "step_complete"]
        if len(metas) != 1 or len(complete) != 1 or len(done) != 1:
            raise ValueError("Incomplete or duplicate lifecycle markers")
        meta = metas[0]
        topology = meta["topology"]
        integers = ("rank", "dp_rank", "tp_rank", "tp_size", "dp_size", "sequence_length", "micro_batch_size",
                    "hidden_size", "cp_size", "pp_size", "ep_size")
        if any(type(topology.get(key)) is not int for key in integers):
            raise ValueError("Malformed topology coordinate types")
        if (meta["schema"] != SCHEMA or meta["source_commit"] != SOURCE_COMMIT
                or meta["scope"] != ("cpu-fixture-only" if cpu_fixture else "native-candidate")
                or topology["rank"] != rank or topology["tp_size"] != 2 or topology["dp_size"] != 4
                or topology["sequence_parallel"] is not True or topology["cp_size"] != 1
                or topology["ep_size"] != 1 or topology["pp_size"] != 1
                or topology["sequence_length"] != 4096 or topology["micro_batch_size"] != 1
                or topology["hidden_size"] != (128 if cpu_fixture else 5120)
                or meta["frozen_inputs"] != {name: digest for name, (_path, digest) in FROZEN_FILES.items()}
                or meta["source_inputs"] != {name: digest for name, (_path, digest) in SOURCE_FILES.items()}
                or meta["arm"] not in ("T", "P") or meta["runtime_proof"].get("verified") is not True
                or complete[0]["completed_steps"] != 3 or done[0]["step"] != 1
                or done[0]["micro_batches"] != 2 or done[0]["unique_batches"] != 2):
            raise ValueError("Mixed/unproven padding scope")
        coordinate = topology["dp_rank"], topology["tp_rank"]
        if coordinate in coordinates or not 0 <= coordinate[0] < 4:
            raise ValueError("Duplicate/invalid DP/TP coordinate")
        coordinates.add(coordinate)
        if target is None:
            target, arm = meta["target_batch"], meta["arm"]
        if (meta["target_batch"], meta["arm"]) != (target, arm) or (not cpu_fixture and target != TARGET_BATCH):
            raise ValueError("Mixed target or active arm")
        batches = [row for row in rows if row["kind"] == "batch"]
        calls = [row for row in rows if row["kind"] == "call"]
        changes = [row for row in rows if row["kind"] == "intervention"]
        if (len(batches) != 2 or {row["micro"] for row in batches} != {0, 1}
                or len(calls) != 2 or {row["micro"] for row in calls} != {0, 1}):
            raise ValueError("Incomplete actual microbatch/call coverage")
        by_micro = {row["micro"]: row["batch_sha256"] for row in batches}
        for row in batches:
            if (row["step"] != 1 or row["dp"] != coordinate[0] or row["tp"] != coordinate[1]
                    or not re.fullmatch(r"[a-f0-9]{64}", row["batch_sha256"])):
                raise ValueError("Actual batch event disagrees with rank coordinates")
            groups.setdefault((coordinate[0], row["micro"]), {})[coordinate[1]] = row["batch_sha256"]
        for row in calls:
            if (row["batch_sha256"] != by_micro[row["micro"]] or row["module"] != MODULE
                    or row["step"] != 1 or row["intervened"] is not (row["batch_sha256"] == target)):
                raise ValueError("Call routing differs from actual batch identity")
        expected_count = 1 if rank in TARGET_RANKS else 0
        if len(changes) != expected_count or complete[0]["target_calls"] != expected_count:
            raise ValueError("Target must occur exactly once on each selected physical peer")
        for row in changes:
            if row["batch_sha256"] != target or row["step"] != 1 or row["micro"] != 0 or row["module"] != MODULE:
                raise ValueError("Wrong target use-point")
            result = row["result"]
            if result["return_contract"] != {
                    "original_type": "torch.Tensor", "effective_type": "torch.Tensor",
                    "same_shape_dtype_device": True, "parameter_restore_before_posthook": True,
                    "DTensor_rewrap": False}:
                raise ValueError("Unproven original/effective plain Tensor return contract")
            if any(result["invariants"].get(name) is not True for name in INVARIANTS):
                raise ValueError("Original graph/state invariant failed")
            weight_proof = result["compute_weight_proof"]
            validate_forward_route(weight_proof["actual_forward_route"])
            if (weight_proof.get("same_original_local_tensor") is not True or result["grad_enabled"] is not True
                    or weight_proof["at_pre"] != weight_proof["after_restore"]
                    or weight_proof["at_pre"]["requires_grad"] is not True
                    or weight_proof["actual_forward_route"]["verified"] is not True):
                raise ValueError("No original-use-point local Tensor identity proof")
            for name in merged:
                stage = name if name in ("input", "weight") else "output"
                item = result[name]
                expected = region(stage, topology, cpu_fixture=cpu_fixture)
                blocks = _blocks(item, expected)
                dtype = "torch.float64" if cpu_fixture else "torch.bfloat16"
                if item["dtype"] != dtype:
                    raise ValueError("Unknown native dtype")
                _physical(item, stage, shape, cpu_fixture=cpu_fixture)
                merged[name]["dtype"] = dtype
                merged[name]["logical_shape"] = expected["logical_shape"]
                for key, digest in blocks.items():
                    previous = merged[name]["blocks"].setdefault(key, digest)
                    if previous != digest:
                        raise ValueError("TP replica native bytes disagree")
            bounds = tuple(region("output", topology, cpu_fixture=cpu_fixture)["bounds"])
            stats = result["original_vs_effective"]
            support.validate_pair(stats, (bounds[1] - bounds[0]) * (bounds[3] - bounds[2]), bounds)
            support._validate_native_pair(  # pylint: disable=protected-access
                _blocks(result["original"], region("output", topology, cpu_fixture=cpu_fixture)),
                _blocks(result["effective"], region("output", topology, cpu_fixture=cpu_fixture)), stats)
            equal = result["original"]["blocks"] == result["effective"]["blocks"]
            expected_status = (("T_NATIVE_CONTROL_MATCH" if equal else "T_NATIVE_CONTROL_MISMATCH")
                               if arm == "T" else "P_ACTIVE_DIAGNOSTIC_ONLY")
            if result["original_equals_effective_native"] is not equal or result["status"] != expected_status:
                raise ValueError("Native control flag/status contradicts the complete tile identities")
            selected.append({"rank": rank, "tp_rank": coordinate[1], "result": result})
    identities = []
    for peers in groups.values():
        if set(peers) != {0, 1} or len(set(peers.values())) != 1:
            raise ValueError("TP peers disagree on logical batch")
        identities.append(next(iter(peers.values())))
    if len(identities) != 8 or len(set(identities)) != 8 or len(selected) != 2:
        raise ValueError("Incomplete/ambiguous eight logical batches")
    if {row["tp_rank"] for row in selected} != {0, 1}:
        raise ValueError("Both disjoint TP output halves are required")
    return {"arm": arm, "target_batch": target, "complete": merged, "selected": selected,
            "scope": "cpu-fixture-only" if cpu_fixture else "one-target-batch-active-short-arm"}


def read_run(directory: Path) -> tuple[dict[int, list[dict]], dict[str, str]]:
    """Hash exactly the byte snapshots used to parse the eight new files."""
    records, digests = {}, {}
    for rank in range(8):
        path = directory / f"qb-padding-rank{rank}.jsonl"
        raw = path.read_bytes()
        records[rank] = [json.loads(line) for line in raw.splitlines()]
        digests[str(path)] = hashlib.sha256(raw).hexdigest()
    return records, digests
