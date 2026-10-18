# Copyright 2026
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

"""Verify the portable evidence without importing Torch or requiring retained native binaries."""

import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import struct


HERE = Path(__file__).resolve().parent
CASES = ("r125-a1e6-ep-stepbind-fsdp8-3-v1", "r125-a1e6-ep-stepbind-ep4-3-v1")
PAIR_PATH = "evidence/a1e6-ep-firststep-runtime-probe-v4/125-v4-fsdp-vs-ep4-native-comparison-v1.json"
PAIR_SHA = "18254a7b45a934e88577fe5d806059f874fb541fc5ca68b177dc56bace06da8d"
SOURCE = "544526f7db8342c607244fd41009b2a74adb84c8"
SCALARS = ("training/total_loss", "training/grad_norm", "training/lr", "training/foundation_loss/aux",
           "training/foundation_loss/lm", "training/foundation_loss/mtp")
COUNTERS = ("data/step_tokens", "data/consumed_tokens", "data/step_samples", "data/consumed_samples")
TARGETS = {"D": (5120, 1, None, 12), "E": (5120, 1536, 0, 9), "Q": (192, 1536, None, 10)}


def require(condition: bool, message: str) -> None:
    """Reject inconsistent or incomplete public evidence."""
    if not condition:
        raise ValueError(message)


def path_for(relative: str) -> Path:
    """Resolve one package-relative file, refusing redirects or traversal."""
    name = PurePosixPath(relative)
    require(not name.is_absolute() and ".." not in name.parts and "\\" not in relative, "Unsafe relative path")
    path = HERE.joinpath(*name.parts)
    require(path.resolve() == path and path.is_file() and not path.is_symlink(), "Missing/redirected public file")
    return path


def fingerprint(path: Path) -> tuple[int, str]:
    """Hash bounded metadata and figures in one-MiB chunks."""
    require(path.stat().st_size <= 2 * 1024**2, "Unexpected large public file")
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024**2)
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def load(relative: str) -> object:
    """Read one adjacent JSON object."""
    return json.loads(path_for(relative).read_bytes())


def verify_manifest() -> int:
    """Verify every published byte and reject added files or native binary payloads."""
    manifest = load("manifest.json")
    expected = set()
    total = 0
    for row in manifest["files"]:
        name = row["path"]
        require(name not in expected and name != "manifest.json", "Duplicate/self manifest row")
        require(not name.endswith(".bin"), "Retained native binaries must not be published")
        expected.add(name)
        actual = fingerprint(path_for(name))
        require(actual == (row["bytes"], row["sha256"]), "Public byte/SHA mismatch: " + name)
        total += actual[0]
    actual_files = {item.relative_to(HERE).as_posix() for item in HERE.rglob("*") if item.is_file()}
    require(actual_files == expected | {"manifest.json"}, "Unexpected or missing file")
    require(total <= 16 * 1024**2, "Public package budget exceeded")
    return len(expected)


def verify_arm(case: str, label: str, pins: dict) -> dict:
    """Check raw scalar/input identity and the full declared selected-role slab catalog.

    Native SHA values are declarations in the raw JSONL, not rehashed native bytes.
    The original full binary verification is documented by the independent review.
    """
    index = load("evidence/retained-" + label + "-manifest-index.json")
    require(index["case"] == case and index["original_file_count"] == 111, "Wrong retained source index")
    original = {item["path"]: item for item in index["files"]}
    for name in index["public_raw_files"]:
        relative = "runs/" + case + "/" + name
        require(fingerprint(path_for(relative)) == (original[relative]["bytes"], original[relative]["sha256"]),
                "Raw file differs from the original run manifest")
    require(path_for("runs/" + case + "/EXIT_CODE").read_bytes() == b"0\n", "Run did not exit0")
    require(path_for("runs/" + case + "/source-commit.txt").read_text().strip() == SOURCE, "Wrong source")
    metrics, initial, runtime, catalog, norms = [], [], [], {}, []
    for rank in range(8):
        prefix = "runs/" + case + "/"
        rows = [json.loads(line) for line in path_for(prefix + f"metrics-rank{rank}.jsonl").read_bytes().splitlines()]
        require([row["step"] for row in rows] == [1, 2, 3], "Expected exactly three ordered steps")
        require(all(row["rank"] == rank for row in rows), "Metric rank mismatch")
        require(all(math.isfinite(row[key]) for row in rows for key in SCALARS + COUNTERS), "Nonfinite scalar")
        require(all(row["input_identity"]["rank_count"] == 8 for row in rows), "Wrong input rank coverage")
        metrics.append(rows)
        initial.append(load(prefix + f"initial-state-rank{rank}.json"))
        runtime.append(load(prefix + f"runtime-flags-rank{rank}.json"))
        require(load(prefix + f"complete-rank{rank}.json")["global_step"] == 3, "Missing complete step3")
        event_path = path_for(prefix + f"ep-firststep-rank{rank}.jsonl")
        events = [json.loads(line) for line in event_path.read_bytes().splitlines()]
        headers = [row for row in events if row["kind"] == "metadata"]
        done = [row for row in events if row["kind"] == "run_complete"]
        require(len(headers) == len(done) == 1, "Missing/duplicate metadata or completion")
        require(headers[0]["rank"] == rank and headers[0]["source_commit"] == SOURCE, "Wrong metadata identity")
        require(headers[0]["schema"] == "ep-firststep-runtime-v4", "Wrong observer schema")
        require(headers[0]["source_proof"] == pins["HP_sources"], "HP source pin mismatch")
        require(headers[0]["framework_source_proof"] == pins["installed_framework_sources"], "Framework pin mismatch")
        require(done[0]["steps"] == 3 and done[0]["wrappers_remaining"] == 0, "Incomplete observer cleanup")
        require(len(done[0]["role_status"]) == 31 and set(done[0]["role_status"].values()) == {"COMPARABLE"},
                "Incomplete selected31 scope")
        norm = [row for row in events if row["kind"] == "total_norm"]
        clip = [row for row in events if row["kind"] == "clip_input"]
        require(len(norm) == len(clip) == 1, "Missing/duplicate actual norm/clip boundary")
        require(norm[0]["value"]["native_hex"] == clip[0]["total_norm"]["native_hex"], "Different clip norm")
        require(clip[0]["max_norm"] == 1 and clip[0]["effective_foreach"] is False, "Changed clipping recipe")
        norms.append(norm[0]["value"]["native_hex"])
        for row in events:
            if row["kind"] != "native":
                continue
            require(row["rank"] == rank and row["target"] in TARGETS, "Native catalog identity mismatch")
            native = row["native"]
            height, width, expert, _ = TARGETS[row["target"]]
            start, stop = row["rows"]
            require(row["expert"] == expert and row["cols"] == [0, width], "Wrong selected region")
            require(start % 128 == 0 and stop == min(start + 128, height) and 0 <= start < stop <= height,
                    "Wrong/overlapping global slab geometry")
            require(native["dtype"] == "torch.float32" and native["byte_order"] == "little", "Wrong native dtype")
            require(native["bytes"] == (stop - start) * width * 4, "Wrong native byte extent")
            require(native["file"] == f"ep-native-rank{rank}.bin", "Wrong physical native owner")
            raw_name = prefix + native["file"]
            require(type(native["offset"]) is int and 0 <= native["offset"] <= original[raw_name]["bytes"]
                    - native["bytes"], "Native reference escapes retained file")
            require(re.fullmatch(r"[a-f0-9]{64}", native["sha256"]) is not None, "Invalid declared native SHA")
            key = (row["target"], row["stage"], expert, start, stop, 0, width)
            require(key not in catalog or catalog[key] == native["sha256"], "Replica native SHA conflict")
            catalog[key] = native["sha256"]
    require(all(value == initial[0] for value in initial), "CPU initial-state hashes differ within run")
    require(all(value == runtime[0] for value in runtime), "Runtime flags differ within run")
    require(len(set(norms)) == 1, "Norm differs across physical rank logs")
    for rank_rows in metrics[1:]:
        for first, second in zip(metrics[0], rank_rows):
            require(all(first[key] == second[key] for key in SCALARS + COUNTERS + ("input_identity",)),
                    "Global scalar/input/counter differs across rank logs")
    return {"metrics": metrics, "initial": initial[0], "runtime": runtime[0], "catalog": catalog,
            "norm_hex": norms[0], "norm": struct.unpack("<f", bytes.fromhex(norms[0]))[0]}


def main() -> None:
    """Verify bytes, selected catalogs and raw metric figures without any device execution."""
    files = verify_manifest()
    require(fingerprint(path_for(PAIR_PATH))[1] == PAIR_SHA, "Changed canonical comparison")
    pair = load(PAIR_PATH)
    require(pair["status"] == "SELECTED_WINDOWS_COMPARED_NOT_CAUSAL", "Changed canonical scope")
    require(len(pair["comparisons"]) == 31, "Incomplete canonical31")
    pins = load("source-pins.json")
    require(pins["source_commit"] == SOURCE and len(pins["HP_sources"]) == 17, "Incomplete source pins")
    first, second = [verify_arm(case, label, pins) for case, label in zip(CASES, ("fsdp", "ep4"))]
    require(first["initial"] == second["initial"], "CPU initial-state hashes differ between arms")
    require(first["runtime"] == second["runtime"], "Runtime flags differ between arms")
    require(set(first["catalog"]) == set(second["catalog"]), "Different selected logical coverage")
    for record in pair["comparisons"]:
        target, stage = record["target"], record["stage"]
        height, width, expert, _ = TARGETS[target]
        keys = {(target, stage, expert, start, min(start + 128, height), 0, width)
                for start in range(0, height, 128)}
        actual = {key for key in first["catalog"] if key[:2] == (target, stage)}
        require(actual == keys and record["numel"] == height * width, "Incomplete selected-role coverage")
        mismatch = sum(first["catalog"][key] != second["catalog"][key] for key in keys)
        require(mismatch == record["byte_mismatched_slabs"], "Declared slab mismatch differs from canonical")
    for target, (_, _, _, expected) in TARGETS.items():
        require(sum(row["target"] == target for row in pair["comparisons"]) == expected, "Wrong role count")
    for left, right in zip(first["metrics"], second["metrics"]):
        for a, b in zip(left, right):
            require(all(a[key] == b[key] for key in COUNTERS + ("input_identity",)),
                    "Cross-arm input/count mismatch")
    data = load("figure-data.json")
    require(data["all31_native_roles"] == pair["comparisons"], "Figure roles differ from canonical")
    for arm, label in zip((first, second), ("fsdp", "ep4")):
        values = [row["training/foundation_loss/lm"] for row in arm["metrics"][0]]
        require(values == data["main_lm"][label], "Figure loss differs from raw metrics")
    for binding in data["bindings"]:
        path = binding["path"].replace("\\", "/")
        require(fingerprint(path_for(path)) == (binding["bytes"], binding["sha256"]), "Figure input binding differs")
    delta = [abs(a - b) for a, b in zip(data["main_lm"]["fsdp"], data["main_lm"]["ep4"])]
    require(delta == data["main_lm"]["max_abs_per_step"], "Figure differences are inconsistent")
    print(json.dumps({"status": "PASS_PORTABLE_METADATA_AND_SCALARS_ONLY", "manifest_files": files,
                      "runs": 2, "ranks_per_run": 8, "steps": 3, "metric_rank_steps": 48,
                      "selected_roles": 31, "unique_declared_slabs_per_run": len(first["catalog"]),
                      "main_lm_max_abs": max(delta), "actual_norms": [first["norm"], second["norm"]],
                      "native_binary_bytes_rehashed": 0, "native_numeric_max_MSE_recomputed": False,
                      "native_numeric_validation": ("Original independent full-binary reviews, "
                                                    "SHA-bound public extracts"),
                      "input_identity_scope": "Global unordered component fingerprints; not per-token DP-slot UID",
                      "acceptance_200_step": "NOT_PROVIDED"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
