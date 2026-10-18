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

"""Compare saved native-forward JSON using the pinned recorder's own merger.

No Torch import, tensor replay, distributed operation or device access occurs.
Hidden observations cover optimizer step one. The three-step check establishes
observer transparency against each topology's own original 200-step prefix.
"""

import argparse
import ast
from collections import Counter
import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping
import unittest
from unittest import mock


OBSERVER_SHA256 = "b8f16b43b9a8ce1b624d786979b986a4835f9d729aa08bf97fc1bbab17e81b6e"
SOURCE_COMMIT = "544526f7db8342c607244fd41009b2a74adb84c8"
PURE_NAMES = ("TOKEN_BLOCK", "FEATURE_BLOCK", "WATCHED_CALLS", "Topology", "canonicalize_run")
CORE_METRICS = (
    "training/total_loss", "training/foundation_loss/lm", "training/foundation_loss/mtp",
    "training/foundation_loss/aux", "training/grad_norm", "training/lr", "data/step_tokens",
    "data/step_samples", "data/consumed_tokens", "data/consumed_samples",
)
BACKBONE_ORDER = (("model.embed_tokens", 0), ("model.layers.0", 0), ("model.layers.1", 0), ("model.norm", 0))


def _select_pure_nodes(tree: ast.Module) -> ast.Module:
    """Select only the five approved definitions; never execute the module imports."""
    selected = {}
    for node in tree.body:
        name = None
        if isinstance(node, ast.Assign):
            targets = {part.id for target in node.targets for part in ast.walk(target) if isinstance(part, ast.Name)}
            if targets & set(PURE_NAMES):
                if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                    raise ValueError("Unexpected assignment to an approved recorder symbol")
                name = node.targets[0].id
                if name not in PURE_NAMES[:3]:
                    raise ValueError("Recorder class/function was replaced by an assignment")
                ast.literal_eval(node.value)
        elif getattr(node, "name", None) in PURE_NAMES:
            name = node.name
            expected_type = ast.ClassDef if name == "Topology" else ast.FunctionDef
            if name not in PURE_NAMES[3:] or type(node) is not expected_type:
                raise ValueError("Unexpected node for an approved recorder symbol")
        if name is not None:
            if name in selected:
                raise ValueError("Duplicate approved recorder definition")
            selected[name] = copy.deepcopy(node)
    if set(selected) != set(PURE_NAMES):
        raise ValueError("Missing approved recorder definition")
    return ast.fix_missing_locations(ast.Module(body=[selected[name] for name in PURE_NAMES], type_ignores=[]))


def load_canonicalizer() -> dict[str, Any]:
    """Load exact pure definitions from the fixed sibling recorder, rejecting another SHA."""
    path = Path(__file__).with_name("native_forward_trace.py")
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != OBSERVER_SHA256:
        raise ValueError("Recorder source differs from the approved b8f fingerprint")
    namespace = {"__name__": __name__, "Counter": Counter, "dataclass": dataclass,
                 "Any": Any, "Iterable": Iterable, "Mapping": Mapping}
    selected = _select_pure_nodes(ast.parse(source, filename=str(path)))
    exec(compile(selected, str(path), "exec", dont_inherit=True), namespace)
    return {name: namespace[name] for name in PURE_NAMES}


def _finite_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _read_json(path: Path, bindings: dict[str, str]) -> Any:
    payload = path.read_bytes()
    bindings[str(path)] = hashlib.sha256(payload).hexdigest()
    return json.loads(payload)


def _read_jsonl(path: Path, bindings: dict[str, str]) -> list[dict]:
    payload = path.read_bytes()
    bindings[str(path)] = hashlib.sha256(payload).hexdigest()
    return [json.loads(line) for line in payload.splitlines()]


def _metric_rows(records: Mapping[int, list[dict]], count: int) -> dict[int, dict[int, dict]]:
    if set(records) != set(range(8)):
        raise ValueError("Exactly eight metric rank files are required")
    result = {}
    for rank, rows in records.items():
        if [row.get("step") for row in rows] != list(range(1, count + 1)):
            raise ValueError("Metric steps must be unique, ordered and complete")
        for row in rows:
            if type(row.get("step")) is not int or type(row.get("rank")) is not int or row["rank"] != rank:
                raise ValueError("Metric rank label differs from its file")
            if any(not _finite_number(row.get(metric)) for metric in CORE_METRICS):
                raise ValueError("Missing or nonfinite loss, gradient, LR or counter metric")
            if any(row[metric] < 0 or not float(row[metric]).is_integer()
                   for metric in CORE_METRICS if metric.startswith("data/")):
                raise ValueError("Data counters must be nonnegative integers")
            identity = row.get("input_identity", {})
            if (identity.get("rank_count") != 8 or type(identity.get("micro_steps")) is not int
                    or identity["micro_steps"] not in (1, 2)
                    or not re.fullmatch(r"[a-f0-9]{64}", identity.get("global_sha256", ""))):
                raise ValueError("Invalid global input identity")
        result[rank] = {row["step"]: row for row in rows}
    return result


def check_transparency(observed: Mapping[int, list[dict]], original: Mapping[int, list[dict]]) -> dict:
    """Require exact three-step scalars/inputs against the same topology's 200-step run.

    Timing and memory are intentionally excluded. This comparison uses matching
    ranks only within the same topology; cross-topology hidden matching never
    uses rank or microbatch indices.
    """
    actual, reference = _metric_rows(observed, 3), _metric_rows(original, 200)
    additional = {key for records in (observed, original) for rows in records.values() for row in rows[:3]
                  for key in row if key.startswith("training/") and ("loss" in key or "grad" in key)}
    metrics = list(CORE_METRICS) + sorted(additional - set(CORE_METRICS))
    errors = {}
    for metric in metrics:
        differences = []
        for step in range(1, 4):
            for rank in range(8):
                left, right = reference[rank][step].get(metric), actual[rank][step].get(metric)
                if not _finite_number(left) or not _finite_number(right):
                    raise ValueError("Incomplete additional loss/gradient evidence")
                differences.append((abs(right - left), step, rank))
        peak = max(differences, key=lambda value: value[0])
        errors[metric] = {"pairs": 24, "max_abs": peak[0], "step": peak[1], "rank": peak[2],
                          "mismatching_pairs": sum(value[0] != 0 for value in differences)}
    input_mismatches = sum(actual[rank][step]["input_identity"] != reference[rank][step]["input_identity"]
                           for rank in range(8) for step in range(1, 4))
    passed = input_mismatches == 0 and all(value["max_abs"] == 0 for value in errors.values())
    return {"status": "PASS" if passed else "FAIL", "exact_equality_required": True,
            "steps": [1, 3], "ranks": 8, "metrics": errors, "input_identity_mismatch_pairs": input_mismatches,
            "excluded": ["performance/*", "memory/*"], "scope": "Each topology against its own original prefix"}


def _canonicalize(records: Mapping[int, list[dict]], core: dict[str, Any]) -> tuple[dict, dict]:
    for rows in records.values():
        for row in rows:
            if row.get("kind") == "metadata":
                if (row.get("schema") != "native-forward-v1" or row.get("token_block") != core["TOKEN_BLOCK"]
                        or row.get("feature_block") != core["FEATURE_BLOCK"]):
                    raise ValueError("Recorder metadata contract differs")
            if "batch_sha256" in row and not re.fullmatch(r"[a-f0-9]{64}", row["batch_sha256"]):
                raise ValueError("Malformed logical batch SHA")
            if row.get("kind") != "hidden":
                continue
            for block in row["blocks"]:
                if not re.fullmatch(r"[a-f0-9]{64}", block["sha256"]):
                    raise ValueError("Malformed canonical block SHA")
            grid = row["grid"]
            for axis in ("tokens", "features"):
                if grid[axis] != sorted(set(grid[axis])) or any(type(value) is not int for value in grid[axis]):
                    raise ValueError("Grid coordinates must be unique sorted integers")
            if any(not _finite_number(value) for values in grid["values"] for value in values):
                raise ValueError("Nonfinite sampled-grid value")
    merged = core["canonicalize_run"](records)
    metadata = next(row for row in records[0] if row["kind"] == "metadata")
    topology = metadata["topology"]
    hidden = topology["hidden_size"]
    stride = max(core["FEATURE_BLOCK"], (hidden // 8 // core["FEATURE_BLOCK"]) * core["FEATURE_BLOCK"])
    tokens = set(range(0, 4096, core["TOKEN_BLOCK"])) | {4095}
    features = set(range(0, hidden, stride)) | {hidden - 1}
    expected_grid = {(token, feature) for token in tokens for feature in features}
    for value in merged.values():
        if set(value["grid"]) != expected_grid:
            raise ValueError("Incomplete or out-of-range saved numerical grid")
    return merged, topology


def compare_record_sets(reference: Mapping[int, list[dict]], candidate: Mapping[int, list[dict]],
                        core: dict[str, Any]) -> dict:
    """Compare complete canonical blocks and sampled grids by batch/module/call identity."""
    left, left_topology = _canonicalize(reference, core)
    right, right_topology = _canonicalize(candidate, core)
    if (left_topology["tp_size"], right_topology["tp_size"]) != (1, 2):
        raise ValueError("Require DP8/TP1 reference and DP4/TP2 candidate")
    if set(left) != set(right):
        raise ValueError("Logical batch/module/call identities differ; rank pairing is forbidden")
    boundaries = {}
    for key in sorted(left):
        batch, module, call = key
        a, b = left[key], right[key]
        if set(a["blocks"]) != set(b["blocks"]) or set(a["grid"]) != set(b["grid"]):
            raise ValueError("Canonical block or grid coordinates are incomparable")
        item = boundaries.setdefault((module, call), {
            "module": module, "call": call, "logical_batches": 0, "complete_block_pairs": 0,
            "complete_block_mismatches": 0, "sampled_grid_pairs": 0, "sampled_grid_max_abs": 0.0,
            "sampled_grid_max_location": None, "dtype_mismatch_batches": 0, "differing_batch_sha256": [],
        })
        block_mismatches = sum(a["blocks"][block] != b["blocks"][block] for block in a["blocks"])
        grid_errors = [(abs(a["grid"][point] - b["grid"][point]), point) for point in sorted(a["grid"])]
        peak, point = max(grid_errors, key=lambda value: value[0])
        dtype_differs = a["dtype"] != b["dtype"]
        if peak != 0 and block_mismatches == 0 and not dtype_differs:
            raise ValueError("Saved grid differs although every corresponding full-byte block hash agrees")
        item["logical_batches"] += 1
        item["complete_block_pairs"] += len(a["blocks"])
        item["complete_block_mismatches"] += block_mismatches
        item["sampled_grid_pairs"] += len(a["grid"])
        item["dtype_mismatch_batches"] += int(dtype_differs)
        if peak > item["sampled_grid_max_abs"]:
            item["sampled_grid_max_abs"] = peak
            item["sampled_grid_max_location"] = {"batch_sha256": batch, "token": point[0], "feature": point[1]}
        if block_mismatches or peak or dtype_differs:
            item["differing_batch_sha256"].append(batch)
    ordered = [boundaries[(module, call)] for module, calls in core["WATCHED_CALLS"].items() for call in range(calls)]
    first = next((boundaries[key] for key in BACKBONE_ORDER if boundaries[key]["differing_batch_sha256"]), None)
    return {"logical_batches": len({key[0] for key in left}), "hidden_observation_steps": [1, 1],
            "reference_topology": left_topology, "candidate_topology": right_topology,
            "matching_key": ["batch_sha256", "module", "call", "canonical_token_feature_coordinates"],
            "boundaries": ordered, "complete_block_pairs": sum(v["complete_block_pairs"] for v in ordered),
            "complete_block_mismatches": sum(v["complete_block_mismatches"] for v in ordered),
            "backbone_boundary_order": [{"module": module, "call": call} for module, call in BACKBONE_ORDER],
            "first_observed_backbone_difference": first,
            "sampled_grid_is_full_tensor_numeric_max": False,
            "root_cause_claimed": False,
            "scope": "Complete byte-block hashes; numerical maxima only over the recorder's small saved grid"}


def _load_directory(directory: Path, count: int, bindings: dict[str, str]) -> dict:
    source = directory / "source-commit.txt"
    payload = source.read_bytes()
    bindings[str(source)] = hashlib.sha256(payload).hexdigest()
    patch = directory / "source.patch"
    patch_bytes = patch.read_bytes()
    bindings[str(patch)] = hashlib.sha256(patch_bytes).hexdigest()
    if payload.decode().strip() != SOURCE_COMMIT or patch_bytes:
        raise ValueError("Diagnostic and original runs must use the fixed clean source")
    for name, expected in (("source-status.txt", ""), ("EXIT_CODE", "0")):
        path = directory / name
        data = path.read_bytes()
        bindings[str(path)] = hashlib.sha256(data).hexdigest()
        if data.decode().strip() != expected:
            raise ValueError("Run source is dirty or the recorded process did not exit zero")
    if count == 3:
        path = directory / "native_forward_trace.py"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        bindings[str(path)] = digest
        if digest != OBSERVER_SHA256:
            raise ValueError("Captured diagnostic recorder differs from the approved source")
    metrics = {rank: _read_jsonl(directory / f"metrics-rank{rank}.jsonl", bindings) for rank in range(8)}
    _metric_rows(metrics, count)
    for rank in range(8):
        start = _read_json(directory / f"start-rank{rank}.json", bindings)
        complete = _read_json(directory / f"complete-rank{rank}.json", bindings)
        if (type(start.get("global_step")) is not int or start["global_step"] != 0
                or type(complete.get("global_step")) is not int or complete["global_step"] != count):
            raise ValueError("Run did not start at zero and complete its declared window")
        if start.get("micro_batches") != metrics[rank][0]["input_identity"]["micro_steps"]:
            raise ValueError("Run start and metric microbatch counts disagree")
    return {"metrics": metrics,
            "runtime_flags": [_read_json(directory / f"runtime-flags-rank{rank}.json", bindings) for rank in range(8)]}


def compare_directories(reference: Path, candidate: Path, reference_original: Path, candidate_original: Path) -> dict:
    """Read saved runs only; execution receipt/collection audits remain separate evidence."""
    paths = [path.resolve() for path in (reference, candidate, reference_original, candidate_original)]
    if len(set(paths)) != 4:
        raise ValueError("Four independent run directories are required")
    core = load_canonicalizer()
    bindings = {}
    runs = [_load_directory(path, count, bindings) for path, count in zip(paths, (3, 3, 200, 200), strict=True)]
    transparency = {}
    for label, observed, original in (("reference", runs[0], runs[2]), ("candidate", runs[1], runs[3])):
        value = check_transparency(observed["metrics"], original["metrics"])
        value["runtime_flags_exact_match"] = observed["runtime_flags"] == original["runtime_flags"]
        if not value["runtime_flags_exact_match"]:
            value["status"] = "FAIL"
        transparency[label] = value
    traces = [{rank: _read_jsonl(path / f"native-forward-rank{rank}.jsonl", bindings) for rank in range(8)}
              for path in paths[:2]]
    compared = compare_record_sets(traces[0], traces[1], core)
    for index, key in enumerate(("reference_topology", "candidate_topology")):
        expected = compared[key]["tp_size"]
        if any(row["input_identity"]["micro_steps"] != expected
               for rows in runs[index]["metrics"].values() for row in rows):
            raise ValueError("Hidden topology contradicts the captured microbatch count")
    transparent = all(value["status"] == "PASS" for value in transparency.values())
    for path, expected in bindings.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError("An input changed during the offline comparison")
    return {"schema": "native-forward-offline-comparison-v1", "observer_source_sha256": OBSERVER_SHA256,
            "canonicalizer": "Exact AST-selected b8f constants, Topology and canonicalize_run; no Torch import",
            "transparency": transparency, "eligible_for_boundary_interpretation": transparent,
            "status": "OBSERVED_BOUNDARY_DIAGNOSTIC" if transparent else "OBSERVER_TRANSPARENCY_FAIL",
            "comparison": compared, "input_sha256": bindings,
            "limitations": ["Saved JSON comparison alone does not audit NPU execution or collection manifests.",
                            "Hidden tensors were observed only in optimizer step1; transparency covers steps1-3.",
                            "Sampled-grid max_abs is not the maximum over every tensor element.",
                            "First observed coarse boundary difference is not a localized function or root cause."]}


def _fixture_records(core: dict[str, Any], tp_size: int) -> dict[int, list[dict]]:
    """Create explicitly synthetic JSON fixtures with deliberately permuted logical batches."""
    result = {}
    permutation = (7, 3, 5, 1, 6, 2, 4, 0)
    for rank in range(8):
        dp, tp = divmod(rank, tp_size)
        topology = core["Topology"](rank, dp, tp, tp_size, 8 // tp_size, True, 128)
        rows = [{"kind": "metadata", "schema": "native-forward-v1", "topology": asdict(topology),
                 "watched_calls_per_micro": core["WATCHED_CALLS"], "token_block": 512, "feature_block": 128}]
        for micro in range(tp_size):
            logical = dp if tp_size == 1 else permutation[dp * 2 + micro]
            batch = hashlib.sha256(f"CPU_FIXTURE_batch{logical}".encode()).hexdigest()
            identity = {"step": 1, "rank": rank, "micro": micro, "dp": dp, "tp": tp, "batch_sha256": batch}
            rows.append({"kind": "batch", **identity})
            for module, calls in core["WATCHED_CALLS"].items():
                begin, end = ((tp * 2048, (tp + 1) * 2048)
                              if tp_size == 2 and module != "model.embed_tokens" else (0, 4096))
                tokens = [token for token in list(range(0, 4096, 512)) + [4095] if begin <= token < end]
                for call in range(calls):
                    blocks = [{"tokens": [token, token + 512], "features": [0, 128],
                               "sha256": hashlib.sha256(f"{batch}|{module}|{call}|{token}".encode()).hexdigest()}
                              for token in range(begin, end, 512)]
                    rows.append({"kind": "hidden", **identity, "module": module, "call": call, "stage": "output",
                                 "dtype": "torch.bfloat16", "shape": [1, end - begin, 128],
                                 "global_shape": [1, 4096, 128],
                                 "sequence_offset": begin, "mapping": "full_sequence" if end - begin == 4096 else
                                 "contiguous_tp_sequence_shard", "blocks": blocks,
                                 "grid": {"tokens": tokens, "features": [0, 127],
                                          "values": [[float(logical + call + token), float(logical + token + 127)]
                                                     for token in tokens]}})
        rows.extend([{"kind": "step_complete", "rank": rank, "step": 1, "micro_batches": tp_size,
                      "unique_batches": tp_size}, {"kind": "run_complete", "rank": rank, "completed_steps": 3}])
        result[rank] = json.loads(json.dumps(rows))
    return result


def _fixture_metrics(count: int, micro_steps: int) -> dict[int, list[dict]]:
    return {rank: [{"step": step, "rank": rank, **{metric: float(step) for metric in CORE_METRICS},
                    "input_identity": {"global_sha256": hashlib.sha256(str(step).encode()).hexdigest(),
                                       "rank_count": 8, "micro_steps": micro_steps},
                    "performance/step_time": 1.0, "memory/device_max_allocated_gb": 1.0}
                   for step in range(1, count + 1)] for rank in range(8)}


def _fixture_directories(core: dict[str, Any]) -> tuple[list[Path], dict[Path, bytes]]:
    """Model a complete four-directory input tree in memory without creating any path."""
    paths = [(Path(__file__).parent / "__CPU_JSON_FIXTURE_ONLY__" / name).resolve()
             for name in ("reference3", "candidate3", "reference200", "candidate200")]
    observer = Path(__file__).with_name("native_forward_trace.py").read_bytes()
    files = {}
    for index, directory in enumerate(paths):
        count = 3 if index < 2 else 200
        micro = 1 if index % 2 == 0 else 2
        files[directory / "source-commit.txt"] = (SOURCE_COMMIT + "\n").encode()
        files[directory / "source.patch"] = b""
        files[directory / "source-status.txt"] = b""
        files[directory / "EXIT_CODE"] = b"0\n"
        metrics = _fixture_metrics(count, micro)
        traces = _fixture_records(core, micro) if count == 3 else None
        if count == 3:
            files[directory / "native_forward_trace.py"] = observer
        for rank in range(8):
            files[directory / f"metrics-rank{rank}.jsonl"] = (
                "\n".join(json.dumps(row) for row in metrics[rank]) + "\n").encode()
            for prefix, value in (("start", {"global_step": 0, "micro_batches": micro}),
                                  ("complete", {"global_step": count}),
                                  ("runtime-flags", {"CPU_FIXTURE_ONLY": True})):
                files[directory / f"{prefix}-rank{rank}.json"] = json.dumps(value).encode()
            if traces is not None:
                files[directory / f"native-forward-rank{rank}.jsonl"] = (
                    "\n".join(json.dumps(row) for row in traces[rank]) + "\n").encode()
    return paths, files


class PureJsonTests(unittest.TestCase):
    """Only synthetic JSON fixtures; these tests cannot establish any NPU result."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.core = load_canonicalizer()

    def test_source_sha_is_required(self) -> None:
        with mock.patch.object(Path, "read_bytes", return_value=b"import torch\n"):
            with self.assertRaises(ValueError):
                load_canonicalizer()

    def test_selection_rejects_missing_duplicate_and_wrong_definition_nodes(self) -> None:
        source = Path(__file__).with_name("native_forward_trace.py").read_bytes()
        tree = ast.parse(source)
        selected = _select_pure_nodes(tree)
        self.assertEqual(len(selected.body), 5)
        for altered in (ast.Module(body=selected.body[:-1], type_ignores=[]),
                        ast.Module(body=selected.body + [selected.body[0]], type_ignores=[]),
                        ast.Module(body=[ast.parse("Topology = 0").body[0]] + selected.body[1:], type_ignores=[])):
            with self.assertRaises(ValueError):
                _select_pure_nodes(altered)

    def test_permuted_batches_and_sequence_shards_match_without_rank_pairing(self) -> None:
        result = compare_record_sets(_fixture_records(self.core, 1), _fixture_records(self.core, 2), self.core)
        self.assertEqual(result["logical_batches"], 8)
        self.assertEqual(result["complete_block_mismatches"], 0)
        self.assertIsNone(result["first_observed_backbone_difference"])
        self.assertFalse(result["sampled_grid_is_full_tensor_numeric_max"])

    def test_complete_hash_mismatch_can_exist_outside_the_sampled_grid(self) -> None:
        left, right = _fixture_records(self.core, 1), _fixture_records(self.core, 2)
        row = next(value for value in right[0] if value.get("module") == "model.layers.0")
        row["blocks"][0]["sha256"] = "f" * 64
        result = compare_record_sets(left, right, self.core)
        first = result["first_observed_backbone_difference"]
        self.assertEqual(first["module"], "model.layers.0")
        self.assertEqual(first["complete_block_mismatches"], 1)
        self.assertEqual(first["sampled_grid_max_abs"], 0)
        self.assertFalse(result["root_cause_claimed"])

    def test_sampled_error_reports_coordinates_and_complete_block_count(self) -> None:
        left, right = _fixture_records(self.core, 1), _fixture_records(self.core, 2)
        row = next(value for value in right[0] if value.get("module") == "model.layers.1")
        row["blocks"][0]["sha256"] = "f" * 64
        row["grid"]["values"][0][0] += 0.25
        result = compare_record_sets(left, right, self.core)
        self.assertEqual(result["first_observed_backbone_difference"]["sampled_grid_max_abs"], 0.25)
        self.assertEqual(result["first_observed_backbone_difference"]["complete_block_mismatches"], 1)

    def test_missing_rank_block_event_and_repeated_batch_are_rejected(self) -> None:
        left = _fixture_records(self.core, 1)
        for kind in ("rank", "block", "event", "batch"):
            right = _fixture_records(self.core, 2)
            if kind == "rank":
                right.pop(7)
            elif kind == "block":
                next(row for row in right[0] if row.get("kind") == "hidden")["blocks"].pop()
            elif kind == "event":
                right[0].pop(next(index for index, row in enumerate(right[0]) if row.get("kind") == "hidden"))
            else:
                first = next(row["batch_sha256"] for row in right[0]
                             if row.get("kind") == "batch" and row["micro"] == 0)
                for rank in (0, 1):
                    for row in right[rank]:
                        if row.get("micro") == 1:
                            row["batch_sha256"] = first
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                compare_record_sets(left, right, self.core)

    def test_incomplete_grid_nonfinite_grid_and_malformed_sha_reject(self) -> None:
        left = _fixture_records(self.core, 1)
        for kind in ("grid", "nonfinite", "sha"):
            right = _fixture_records(self.core, 2)
            row = next(value for value in right[0] if value.get("module") == "model.layers.0")
            if kind == "grid":
                row["grid"]["tokens"].pop()
                row["grid"]["values"].pop()
            elif kind == "nonfinite":
                row["grid"]["values"][0][0] = float("nan")
            else:
                row["blocks"][0]["sha256"] = "short"
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                compare_record_sets(left, right, self.core)

    def test_grid_disagreement_with_all_equal_hashes_is_rejected(self) -> None:
        left, right = _fixture_records(self.core, 1), _fixture_records(self.core, 2)
        row = next(value for value in right[0] if value.get("module") == "model.layers.0")
        row["grid"]["values"][0][0] += 1
        with self.assertRaises(ValueError):
            compare_record_sets(left, right, self.core)

    def test_transparency_excludes_only_timing_memory_and_requires_exact_core(self) -> None:
        original, observed = _fixture_metrics(200, 2), _fixture_metrics(3, 2)
        observed[0][0]["performance/step_time"] = 999
        observed[0][0]["memory/device_max_allocated_gb"] = 999
        self.assertEqual(check_transparency(observed, original)["status"], "PASS")
        for field in CORE_METRICS:
            value = copy.deepcopy(observed)
            value[7][2][field] += 1 if field.startswith("data/") else 0.000001
            with self.subTest(field=field):
                self.assertEqual(check_transparency(value, original)["status"], "FAIL")
        value = copy.deepcopy(observed)
        value[0][0]["input_identity"]["global_sha256"] = "f" * 64
        self.assertEqual(check_transparency(value, original)["status"], "FAIL")

    def test_missing_steps_and_extra_loss_are_not_silently_ignored(self) -> None:
        original, observed = _fixture_metrics(200, 1), _fixture_metrics(3, 1)
        observed[0].pop()
        with self.assertRaises(ValueError):
            check_transparency(observed, original)
        observed = _fixture_metrics(3, 1)
        for records in (original, observed):
            for rows in records.values():
                for row in rows:
                    row["training/foundation_loss/other"] = 1.0
        observed[0][0]["training/foundation_loss/other"] = 1.1
        self.assertEqual(check_transparency(observed, original)["status"], "FAIL")

    def test_four_directory_pipeline_is_pure_json_and_transparency_failure_is_ineligible(self) -> None:
        paths, files = _fixture_directories(self.core)
        actual_read = Path.read_bytes

        def fixture_read(path: Path) -> bytes:
            return files[path] if path in files else actual_read(path)

        with mock.patch.object(Path, "read_bytes", fixture_read):
            result = compare_directories(*paths)
        self.assertTrue(result["eligible_for_boundary_interpretation"])
        self.assertEqual(result["comparison"]["complete_block_mismatches"], 0)
        path = paths[1] / "metrics-rank0.jsonl"
        rows = [json.loads(row) for row in files[path].splitlines()]
        rows[0]["training/foundation_loss/lm"] += 0.001
        files[path] = ("\n".join(json.dumps(row) for row in rows) + "\n").encode()
        with mock.patch.object(Path, "read_bytes", fixture_read):
            result = compare_directories(*paths)
        self.assertFalse(result["eligible_for_boundary_interpretation"])
        self.assertEqual(result["status"], "OBSERVER_TRANSPARENCY_FAIL")

    def test_failed_exit_or_mismatched_captured_observer_rejects(self) -> None:
        actual_read = Path.read_bytes
        for kind in ("exit", "observer"):
            paths, files = _fixture_directories(self.core)
            files[paths[0] / ("EXIT_CODE" if kind == "exit" else "native_forward_trace.py")] = b"wrong"

            def fixture_read(path: Path) -> bytes:
                return files[path] if path in files else actual_read(path)

            with self.subTest(kind=kind), mock.patch.object(Path, "read_bytes", fixture_read):
                with self.assertRaises(ValueError):
                    compare_directories(*paths)


def main() -> None:
    """Run pure JSON tests or create one exclusive offline comparison report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--reference-original", type=Path)
    parser.add_argument("--candidate-original", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    directories = [args.reference, args.candidate, args.reference_original, args.candidate_original]
    if args.self_test:
        if any(value is not None for value in directories + [args.output]):
            parser.error("--self-test accepts no run or output paths")
        torch_before = "torch" in sys.modules
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(PureJsonTests)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        loaded_torch = "torch" in sys.modules and not torch_before
        print(json.dumps({"scope": "Synthetic pure-JSON CPU fixtures only; no NPU result", "tests": result.testsRun,
                          "passed": result.wasSuccessful(), "torch_imported": loaded_torch,
                          "observer_sha256": OBSERVER_SHA256}))
        raise SystemExit(0 if result.wasSuccessful() and not loaded_torch else 1)
    if any(value is None for value in directories) or args.output is None:
        parser.error("Provide four run directories and --output, or use --self-test")
    target = args.output.resolve()
    if any(target == path.resolve() or path.resolve() in target.parents for path in directories):
        raise ValueError("Output must be outside every immutable run directory")
    result = compare_directories(*directories)
    with target.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "eligible_for_boundary_interpretation":
                      result["eligible_for_boundary_interpretation"], "output": str(target)}))


if __name__ == "__main__":
    main()
