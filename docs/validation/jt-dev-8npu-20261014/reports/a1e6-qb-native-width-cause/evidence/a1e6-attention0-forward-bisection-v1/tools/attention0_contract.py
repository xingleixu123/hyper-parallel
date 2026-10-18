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
"""Explicit projection coordinates and Torch-free complete/Partial JSON merging."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping


SCHEMA = "attention0-forward-v1"
SOURCE_COMMIT = "544526f7db8342c607244fd41009b2a74adb84c8"
FROZEN_SHA256 = "b8f16b43b9a8ce1b624d786979b986a4835f9d729aa08bf97fc1bbab17e81b6e"
MODEL_SOURCE_SHA256 = "2ddd28910688710ea777f1724fb8aeb8821c6045a01f18f21eea4e92507fd906"
REWRITER_SHA256 = "696acee412fd606c3abfd977435bd11d13fe4fb9f39c017a63acaaf523acb853"
PARENT = "model.layers.0.self_attn"
NATIVE_SHAPES = {"linear_qkv": (2112, 5120), "q_b_proj": (24576, 1536),
                 "kv_b_proj": (32768, 512), "o_proj": (5120, 16384)}
CPU_SHAPES = {"linear_qkv": (192, 128), "q_b_proj": (256, 128),
              "kv_b_proj": (512, 64), "o_proj": (128, 256)}
EVENT_ORDER = tuple((f"{PARENT}.{name}", stage) for name in NATIVE_SHAPES
                    for stage in ("weight", "input", "output")) + ((PARENT, "output"),)
EVENT_CALLS = {f"{module}::{stage}": 1 for module, stage in EVENT_ORDER}
HASH_PATTERN = re.compile(r"[a-f0-9]{64}")


def projection_shapes(cpu_fixture: bool = False) -> dict[str, tuple[int, int]]:
    """Return an independent explicit native or small CPU-only shape contract."""
    return dict(CPU_SHAPES if cpu_fixture else NATIVE_SHAPES)


def expected_region(module: str, stage: str, topology: Mapping[str, Any], *,
                    cpu_fixture: bool = False) -> dict[str, Any]:
    """Derive only the pinned contiguous logical TP/SP ranges, never a rank pairing."""
    tp, size = topology["tp_rank"], topology["tp_size"]
    if size not in (1, 2) or not 0 <= tp < size:
        raise ValueError("Unsupported TP coordinate")
    shapes = projection_shapes(cpu_fixture)
    if module == PARENT and stage == "output":
        width = shapes["o_proj"][0]
        begin = tp * (4096 // size)
        return {"logical_shape": [4096, width], "bounds": [begin, begin + 4096 // size, 0, width],
                "semantics": "complete", "physical_ndim": 3}
    if not module.startswith(PARENT + ".") or module.rsplit(".", 1)[1] not in shapes:
        raise ValueError("Unknown projection module")
    name = module.rsplit(".", 1)[1]
    rows, columns = shapes[name]
    if stage == "weight":
        rb, re_, cb, ce = 0, rows, 0, columns
        if name in ("q_b_proj", "kv_b_proj"):
            rb, re_ = tp * (rows // size), (tp + 1) * (rows // size)
        elif name == "o_proj":
            cb, ce = tp * (columns // size), (tp + 1) * (columns // size)
        return {"logical_shape": [rows, columns], "bounds": [rb, re_, cb, ce],
                "semantics": "weight", "physical_ndim": 2}
    if stage not in ("input", "output"):
        raise ValueError("Unknown projection stage")
    width = columns if stage == "input" else rows
    tb, te, fb, fe = 0, 4096, 0, width
    semantics = "complete"
    if name == "linear_qkv":
        tb, te = tp * (4096 // size), (tp + 1) * (4096 // size)
    elif stage == "output" and name in ("q_b_proj", "kv_b_proj"):
        fb, fe = tp * (width // size), (tp + 1) * (width // size)
    elif name == "o_proj" and stage == "input":
        fb, fe = tp * (width // size), (tp + 1) * (width // size)
    elif name == "o_proj" and stage == "output":
        semantics = "k_partition_contribution"
    result = {"logical_shape": [4096, width], "bounds": [tb, te, fb, fe], "semantics": semantics,
              "physical_ndim": 4 if name == "kv_b_proj" else 3}
    if semantics == "k_partition_contribution":
        result["contributor_k"] = [tp * (columns // size), (tp + 1) * (columns // size)]
    return result


def tile_coordinates(bounds: list[int], tile: tuple[int, int]) -> set[tuple[int, int, int, int]]:
    """Enumerate every true edge-clipped block without padding."""
    rb, re_, cb, ce = bounds
    return {(row, min(row + tile[0], re_), col, min(col + tile[1], ce))
            for row in range(rb, re_, tile[0]) for col in range(cb, ce, tile[1])}


def sample_coordinates(begin: int, end: int, block: int, global_extent: int) -> list[int]:
    """Choose bounded deterministic positions anchored to one global logical axis."""
    stride = max(block, (global_extent // 8 // block) * block)
    selected = sorted(set(range(0, global_extent, stride)) | {global_extent - 1})
    return [position for position in selected if begin <= position < end]


def _check_topology(value: Mapping[str, Any], rank: int, cpu_fixture: bool) -> None:
    integer_fields = ("rank", "dp_rank", "tp_rank", "tp_size", "dp_size", "hidden_size", "sequence_length",
                      "cp_size", "pp_size", "ep_size", "micro_batch_size")
    if any(type(value.get(field)) is not int for field in integer_fields):
        raise ValueError("Malformed topology coordinates")
    if (value["rank"] != rank or value["tp_size"] not in (1, 2)
            or value["dp_size"] * value["tp_size"] != 8
            or not 0 <= value["dp_rank"] < value["dp_size"]
            or not 0 <= value["tp_rank"] < value["tp_size"]
            or value["hidden_size"] != (128 if cpu_fixture else 5120)
            or value["sequence_length"] != 4096
            or any(value[field] != 1 for field in ("cp_size", "pp_size", "ep_size", "micro_batch_size"))
            or value.get("sequence_parallel") is not (value["tp_size"] == 2)):
        raise ValueError("Unknown native topology/layout contract")


def _validate_summary(row: Mapping[str, Any], region: dict[str, Any]) -> None:
    for field in ("logical_shape", "bounds", "semantics"):
        if row.get(field) != region[field]:
            raise ValueError(f"Saved {field} differs from the actual coordinate contract")
    if row.get("contributor_k") != region.get("contributor_k"):
        raise ValueError("Wrong Partial K contributor")
    expected_ndim = region["physical_ndim"]
    rb, re_, cb, ce = region["bounds"]
    local_shape = ([re_ - rb, ce - cb] if expected_ndim == 2 else
                   [1, re_ - rb, ce - cb] if expected_ndim == 3 else [1, 1, re_ - rb, ce - cb])
    physical = row.get("physical", {})
    if (physical.get("shape") != local_shape or len(physical.get("stride", [])) != expected_ndim
            or any(type(item) is not int or item <= 0 for item in physical["stride"])
            or physical.get("dtype") not in ("torch.float32", "torch.bfloat16", "torch.float16", "torch.float64")):
        raise ValueError("Unknown physical tensor shape/stride/dtype")
    tile = (128, 128) if region["semantics"] == "weight" else (512, 128)
    blocks = row.get("blocks", [])
    keys = [tuple(block.get("coordinates", [])) for block in blocks]
    if len(keys) != len(set(keys)) or set(keys) != tile_coordinates(region["bounds"], tile):
        raise ValueError("Incomplete, duplicate or misplaced canonical blocks")
    if any(not HASH_PATTERN.fullmatch(str(block.get("sha256", ""))) for block in blocks):
        raise ValueError("Invalid block SHA")
    grid = row.get("grid", {})
    expected_rows = sample_coordinates(rb, re_, tile[0], region["logical_shape"][0])
    expected_cols = sample_coordinates(cb, ce, tile[1], region["logical_shape"][1])
    if grid.get("rows") != expected_rows or grid.get("columns") != expected_cols:
        raise ValueError("Incomplete numerical-grid coordinates")
    values = grid.get("values", [])
    if len(values) != len(expected_rows) or any(len(items) != len(expected_cols) for items in values):
        raise ValueError("Incomplete numerical-grid values")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item)
           for items in values for item in items):
        raise ValueError("Nonfinite numerical-grid values")


def _validate_runtime_proof(proof: Mapping[str, Any], tp_size: int, cpu_fixture: bool) -> None:
    if proof.get("verified") is not True:
        raise ValueError("Missing verified runtime layout contract")
    if cpu_fixture:
        if proof.get("scope") not in ("synthetic-json-fixture", "cpu-toy-not-jt-not-a-collective"):
            raise ValueError("CPU fixture must not claim native runtime evidence")
        return
    if (proof.get("source_commit") != SOURCE_COMMIT or proof.get("model_source_sha256") != MODEL_SOURCE_SHA256
            or proof.get("rewriter_sha256") != REWRITER_SHA256
            or proof.get("kv_b_physical_contract") != "[1,1,4096,feature]"
            or proof.get("o_proj_output") != "K-partition contribution"):
        raise ValueError("Native runtime proof is not bound to the fixed sources and boundary contract")
    plans = proof.get("compiled_tp_plans", {})
    if tp_size == 1:
        if plans != {PARENT: None}:
            raise ValueError("Unknown baseline compiled-boundary metadata")
        return
    shard = {"kind": "shard", "dim": 1}
    identity = {"src": shard, "dst": shard}
    expected = {PARENT: {"input": [identity], "output": [
        {"src": {"kind": "partial", "reduce_op": "sum"}, "dst": shard}]}}
    for suffix in ("q_a_layernorm", "kv_a_layernorm", "key_rope_gather"):
        expected[f"{PARENT}.{suffix}"] = {"input": [identity], "output": [
            {"src": shard, "dst": {"kind": "replicate"}}]}
    if plans != expected:
        raise ValueError("Saved actual TP/SP boundary plans differ from the exact contract")


def _validate_materialization(row: Mapping[str, Any], cpu_fixture: bool) -> None:
    proof = row.get("materialization", {})
    if (proof.get("verified") is not True or proof.get("state") != "UNSHARDED"
            or proof.get("owner_is_shard") is not False or proof.get("pending_handle") is not False
            or proof.get("installed_storage_equal") is not True or proof.get("physical") != row["physical"]):
        raise ValueError("Weight has no completed, installed, use-point materialization proof")
    if not cpu_fixture and row["physical"]["dtype"] != "torch.bfloat16":
        raise ValueError("Actual native materialized weight is not the fixed BF16 compute contract")


def canonicalize_run(rank_records: Mapping[int, Iterable[dict[str, Any]]], *,
                     cpu_fixture: bool = False) -> dict[str, Any]:
    """Merge complete states by actual batch, keeping o_proj contributors separate.

    No arithmetic is performed on Partial projections. Repeated W tiles across
    all first-step batches must retain their exact identity within each run.
    """
    if set(rank_records) != set(range(8)):
        raise ValueError("Exactly eight physical rank files are required")
    complete, contributors, batches, coordinates, weight_identity = {}, {}, {}, set(), {}
    weight_dtypes = {}
    weight_grid_identity = {}
    reference = None
    for rank, source_rows in rank_records.items():
        rows = list(source_rows)
        if not rows or rows[0].get("kind") != "metadata":
            raise ValueError("Metadata must be first")
        meta, topology = rows[0], rows[0].get("topology", {})
        _check_topology(topology, rank, cpu_fixture)
        tp_size = topology["tp_size"]
        if (meta.get("schema") != SCHEMA or meta.get("source_commit") != SOURCE_COMMIT
                or meta.get("frozen_source_sha256") != FROZEN_SHA256
                or meta.get("scope") != ("cpu-fixture" if cpu_fixture else "native-candidate")
                or meta.get("event_order") != [list(key) for key in EVENT_ORDER]
                or meta.get("projection_shapes") != {
                    key: list(value) for key, value in projection_shapes(cpu_fixture).items()}):
            raise ValueError("Schema, source or runtime layout proof mismatch")
        _validate_runtime_proof(meta.get("runtime_layout_proof", {}), tp_size, cpu_fixture)
        if reference is not None and reference != tp_size:
            raise ValueError("Mixed topologies")
        reference = tp_size
        coordinate = topology["dp_rank"], topology["tp_rank"]
        if coordinate in coordinates:
            raise ValueError("Duplicate DP/TP coordinates")
        coordinates.add(coordinate)
        expected_kinds = ["metadata"] + (["batch"] + ["boundary"] * len(EVENT_ORDER)) * tp_size
        expected_kinds += ["step_complete", "run_complete"]
        if [row.get("kind") for row in rows] != expected_kinds:
            raise ValueError("Incomplete, duplicate or reordered boundary lifecycle")
        for row in rows[1:]:
            if type(row.get("rank")) is not int or row["rank"] != rank:
                raise ValueError("Physical rank mismatch")
        for micro in range(tp_size):
            start = 1 + micro * (len(EVENT_ORDER) + 1)
            batch = rows[start]
            identity = batch.get("batch_sha256", "")
            if not HASH_PATTERN.fullmatch(str(identity)):
                raise ValueError("Invalid actual batch SHA")
            common = {"step": 1, "micro": micro, "dp": topology["dp_rank"],
                      "tp": topology["tp_rank"], "batch_sha256": identity}
            if any(batch.get(key) != value for key, value in common.items()):
                raise ValueError("Batch identity/consumption mismatch")
            batches.setdefault((topology["dp_rank"], micro), {})[topology["tp_rank"]] = identity
            selected = rows[start + 1:start + 1 + len(EVENT_ORDER)]
            for row, (module, stage) in zip(selected, EVENT_ORDER, strict=True):
                if ((row.get("module"), row.get("stage"), row.get("call")) != (module, stage, 0)
                        or any(row.get(key) != value for key, value in common.items())):
                    raise ValueError("Boundary identity or module/stage order mismatch")
                region = expected_region(module, stage, topology, cpu_fixture=cpu_fixture)
                _validate_summary(row, region)
                dtype = row["physical"]["dtype"]
                key = identity, module, stage
                bank = complete
                if region["semantics"] == "k_partition_contribution":
                    key += tuple(region["contributor_k"])
                    bank = contributors
                merged = bank.setdefault(key, {"dtype": dtype, "logical_shape": region["logical_shape"],
                                               "blocks": {}, "grid": {}})
                if merged["dtype"] != dtype:
                    raise ValueError("Replica dtype mismatch")
                if stage == "weight":
                    _validate_materialization(row, cpu_fixture)
                if stage == "weight" and weight_dtypes.setdefault(module, dtype) != dtype:
                    raise ValueError("Materialized weight dtype changed across copies")
                for block in row["blocks"]:
                    block_key = tuple(block["coordinates"])
                    if merged["blocks"].setdefault(block_key, block["sha256"]) != block["sha256"]:
                        raise ValueError("Replica/TP overlap block mismatch")
                    if stage == "weight":
                        weight_key = module, block_key, dtype
                        if weight_identity.setdefault(weight_key, block["sha256"]) != block["sha256"]:
                            raise ValueError("Materialized weight changed across DP/microbatch copies")
                grid = row["grid"]
                for global_row, values in zip(grid["rows"], grid["values"], strict=True):
                    for col, value in zip(grid["columns"], values, strict=True):
                        if merged["grid"].setdefault((global_row, col), value) != value:
                            raise ValueError("Replica numerical-grid mismatch")
                        if stage == "weight":
                            weight_key = module, global_row, col, dtype
                            if weight_grid_identity.setdefault(weight_key, value) != value:
                                raise ValueError("Materialized weight grid changed across DP/microbatch copies")
        if (rows[-2].get("step") != 1 or rows[-2].get("micro_batches") != tp_size
                or rows[-2].get("unique_batches") != tp_size or rows[-1].get("completed_steps") != 3):
            raise ValueError("Incomplete three-step run")
    identities = []
    for peers in batches.values():
        if set(peers) != set(range(reference)) or len(set(peers.values())) != 1:
            raise ValueError("TP peers disagree on actual batch identity")
        identities.append(next(iter(peers.values())))
    if len(identities) != 8 or len(set(identities)) != 8:
        raise ValueError("Repeated or missing logical input batch")
    if len(complete) != 8 * (len(EVENT_ORDER) - 1) or len(contributors) != 8 * reference:
        raise ValueError("Incomplete logical projection/contributor states")
    for bank in (complete, contributors):
        for key, value in bank.items():
            rows, columns = value["logical_shape"]
            tile = (128, 128) if key[2] == "weight" else (512, 128)
            if set(value["blocks"]) != tile_coordinates([0, rows, 0, columns], tile):
                raise ValueError("Missing full logical tile coverage")
    return {"complete": complete, "contributors": contributors, "logical_batches": 8,
            "tp_size": reference, "weight_tile_identity": weight_identity}


def read_run(directory: Path) -> tuple[dict[int, list[dict]], dict[str, str]]:
    """Read exactly eight new-schema files without importing a tensor framework."""
    rows, bindings = {}, {}
    for rank in range(8):
        path = directory / f"attention0-forward-rank{rank}.jsonl"
        payload = path.read_bytes()
        rows[rank] = [json.loads(line) for line in payload.splitlines()]
        bindings[str(path)] = hashlib.sha256(payload).hexdigest()
    return rows, bindings
