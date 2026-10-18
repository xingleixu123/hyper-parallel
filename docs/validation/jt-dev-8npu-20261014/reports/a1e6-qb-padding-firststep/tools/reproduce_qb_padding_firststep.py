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

"""Reproduce the published one-batch q_b result offline, using only Python's stdlib.

The frozen source-values file is a historical author snapshot. This program reads
its numeric/scientific fields explicitly; its historical review_status is not a
current approval. Current scientific and publication states are checked separately.
No Torch module, training entry, SSH command, container, or accelerator is invoked.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import sys
from types import ModuleType
from typing import Any, Callable, Mapping


SOURCE = "544526f7db8342c607244fd41009b2a74adb84c8"
VALUES_SHA = "1ceb050cb0c49817eeb0234a60d76ed628323318f5984c4290053cb2c11daeff"
PAIR_SHA = "f7a2bdd342474820dcfb34e7448f29d263a6e839dd7194ba2152cdcd7bbbe7ed"
REVIEW_SHA = "4f5d682077a5afe40de9517ce0c989946596a6f95fc9642709f946b119e8c247"
NEIGHBOR_MANIFEST_SHA = "5498d76b34c860c17d136a0b4be952a71fa0900dc5ce95e77fd1b566ee84cd3d"
TOOL_DIR = "evidence/a1e6-qb-padding-causal-v2/tools"
CASES = {
    "T": "r129-a1e6-qb-padding-t-tp2-3-v2",
    "P": "r129-a1e6-qb-padding-p-tp2-3-v2",
    "TP": "r129-a1e6-attention0-forward-tp2-3-v1",
    "F": "r129-a1e6-attention0-forward-fsdp8-3-v1",
    "TP200": "r129-a1e6-tp2-200-v2",
    "F200": "r129-a1e6-fsdp8-0-200-v3",
}
NUMERIC_SNAPSHOT_FIELDS = (
    "original_comparison_fields", "target", "native_runtime_selected_records",
    "first_three_step_logged_scalars", "recipe", "execution_summary",
)


def require(condition: bool, message: str) -> None:
    """Fail explicitly when published evidence is incomplete or contradictory."""
    if not condition:
        raise ValueError(message)


def sha_file(path: Path) -> str:
    """Hash a file in bounded chunks without retaining its complete contents."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    """Read a JSON evidence record; parsing is separate from its manifest identity."""
    return json.loads(path.read_bytes())


def safe_member(root: Path, relative: str) -> Path:
    """Resolve a manifest member strictly inside its own bundle."""
    pure = PurePosixPath(relative)
    require(bool(pure.parts) and not pure.is_absolute() and ".." not in pure.parts
            and "\\" not in relative and ":" not in relative, "Unsafe manifest path")
    path = root.joinpath(*pure.parts)
    require(path.is_file() and not path.is_symlink() and root.resolve() in path.resolve().parents,
            "Missing or indirect manifest file: " + relative)
    return path


def verify_manifest(root: Path, expected_sha: str | None = None) -> dict[str, Any]:
    """Verify exact file inventory, bytes, and SHA for a self-contained bundle."""
    manifest_path = root / "manifest.json"
    if expected_sha is not None:
        require(sha_file(manifest_path) == expected_sha, "The reference bundle manifest changed")
    manifest = read_json(manifest_path)
    paths = [item["path"] for item in manifest["files"]]
    require(len(paths) == len(set(paths)) and "manifest.json" not in paths, "Duplicate manifest entries")
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    require(actual == set(paths) | {"manifest.json"}, "Manifest and actual file inventory differ")
    total = 0
    for item in manifest["files"]:
        path = safe_member(root, item["path"])
        require(type(item["bytes"]) is int and path.stat().st_size == item["bytes"],
                "Wrong byte count: " + item["path"])
        require(sha_file(path) == item["sha256"], "Wrong file SHA: " + item["path"])
        total += item["bytes"]
    return {"entries": len(paths), "listed_bytes": total, "manifest_sha256": sha_file(manifest_path)}


def load_verified(name: str, path: Path, expected_sha: str) -> ModuleType:
    """Execute the very same verified raw snapshot, rejecting an ambiguous module."""
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected_sha, "Frozen executable input changed: " + name)
    require(name not in sys.modules, "An unexpected preloaded module shadows " + name)
    module = ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    try:
        exec(compile(raw, str(path), "exec", dont_inherit=True), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def load_contracts(bundle: Path, neighbor: Path) -> tuple[ModuleType, ModuleType, ModuleType]:
    """Bind frozen pure validators to explicit published paths without changing their bytes.

    Only the filesystem resolver is adapted in memory. All six original support
    inputs are SHA-checked, but only the stdlib shadow/attention JSON contracts are
    executed. Requests for tensor or capture execution are rejected.
    """
    padding = load_verified("qb_padding_contract", bundle / TOOL_DIR / "qb_padding_contract.py",
                            "9a2b6ad90e15d4d2f86c8553397fcad339f0993d547785488a723a0d99d71d7f")
    for name, (relative, digest) in padding.FROZEN_FILES.items():
        require(sha_file(safe_member(neighbor, relative)) == digest, "Frozen dependency changed: " + name)
    for name, (relative, digest) in padding.SOURCE_FILES.items():
        source_bundle = neighbor if name == "rewriter" else bundle
        require(sha_file(safe_member(source_bundle, "source/544526f7/" + relative)) == digest,
                "Fixed source anchor changed: " + name)
    shadow_relative, shadow_sha = padding.FROZEN_FILES["qb_shadow_contract"]
    shadow = load_verified("qb_shadow_contract", neighbor / shadow_relative, shadow_sha)
    attention_relative, attention_sha = padding.FROZEN_FILES["attention0_contract"]
    attention = load_verified("attention0_contract", neighbor / attention_relative, attention_sha)

    def published_pure_support(*, tensor: bool = False, capture: bool = False) -> ModuleType:
        if tensor or capture:
            raise ValueError("Offline reproduction never executes tensor/capture modules")
        return shadow

    padding.frozen_support = published_pure_support
    compare = load_verified("compare_qb_padding", bundle / TOOL_DIR / "compare_qb_padding.py",
                            "de840922c4e9ee4353bf7a4afa04aa63640a6df2047cd72be404f6e82ed9bafa")
    require(not any(name == "torch" or name.startswith(("torch.", "torch_npu")) for name in sys.modules),
            "The offline stdlib contract unexpectedly imported a tensor backend")
    return padding, attention, compare


def verify_review_state(bundle: Path) -> tuple[dict, dict]:
    """Keep the historical numeric snapshot and current review state distinct."""
    require(sha_file(bundle / "source-values.json") == VALUES_SHA, "Frozen numeric snapshot changed")
    values = read_json(bundle / "source-values.json")
    state = read_json(bundle / "review-status.json")
    review = read_json(bundle / "evidence/paired-independent-review.json")
    require(values["source_comparison_sha256"] == PAIR_SHA, "The numeric snapshot has a different source")
    require(state["frozen_values_original_sha256"] == VALUES_SHA
            and state["source_comparison_sha256"] == PAIR_SHA
            and state["paired_scientific_review"] == "PASS"
            and state["paired_scientific_review_original_sha256"] == REVIEW_SHA
            and state["publication_independent_review"] == "PENDING"
            and state["root_cause"] == "UNRESOLVED" and state["training200_acceptance"] is False,
            "Current review state is incomplete or overstates acceptance")
    require(review["status"] == state["paired_scientific_review_status"] and not review["findings"]
            and review["public_extract"]["original_sha256"] == REVIEW_SHA,
            "The public scientific review extract does not bind the current PASS")
    return values, state


def runtime_snapshot(run: Mapping[str, Any]) -> list[dict]:
    """Project only the fields retained by the immutable numeric snapshot."""
    result = []
    for row in run["selected"]:
        item = row["result"]
        result.append({
            "rank": row["rank"], "global_features": item["global_features"],
            "runtime_full_numeric": item["original_vs_effective"],
            "all_invariants_true": all(item["invariants"].values()),
            "actual_forward_route": item["compute_weight_proof"]["actual_forward_route"]["route"],
            "live_weight_shape": item["compute_weight_proof"]["at_pre"]["physical"]["shape"],
            "live_weight_stride": item["compute_weight_proof"]["at_pre"]["physical"]["stride"],
            "compute_weight_same_object": item["compute_weight_proof"]["same_original_local_tensor"],
            "explicit_forward_extra_tensor_bytes_bound": item["explicit_forward_extra_tensor_bytes_bound"],
            "backward_and_backend_workspace_bound": item["backward_and_backend_workspace_bound"],
        })
    return result


def verify_numeric_recipe(values: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    """Read the numeric snapshot recipe from actual resolved fields, not defaults."""
    training, model = config["training"], config["model"]["config"]
    precision = config["fsdp_config"]["mix_precision"]
    recipe = {
        "seed": training["seed"], "dataloader_seed": config["dataloader"]["seed"],
        "global_batch_size": training["global_batch_size"], "micro_batch_size": training["micro_batch_size"],
        "sequence_length": 4096, "lr": config["optimizer"]["muon_config"]["lr"],
        "scheduler_iters": training["lr_scheduler_iters"], "param_dtype": precision["param_dtype"],
        "reduce_dtype": precision["reduce_dtype"], "mtp_loss_factor": model["mtp_loss_factor"],
        "moe_aux_loss_coeff": model["moe_aux_loss_coeff"], "recompute_mode": config["activation_checkpoint"]["mode"],
        "TP": 2, "CP": 1, "EP": 1, "PP": 1, "train_iters": training["train_iters"],
        "save_ckpt": config["checkpoint"]["save_ckpt"],
    }
    # Sequence/topology values are independently validated in all eight trace metadata records.
    require(recipe == values["recipe"], "The frozen numeric recipe differs from the actual resolved config")
    require(config["optimizer"]["muon_config"]["ns_dtype"] == "float32"
            and model["expert_fp32_weight_gradients"] is True,
            "Existing FP32 NS/expert-wgrad recipe changed")


def reproduce(bundle: Path) -> tuple[dict[str, Any], dict, ModuleType]:
    """Reconstruct complete logical native maps and first-step LM from captured evidence."""
    neighbor = bundle.parent / "a1e6-qb-native-width-cause"
    inventories = {"this_bundle": verify_manifest(bundle),
                   "reference_bundle": verify_manifest(neighbor, NEIGHBOR_MANIFEST_SHA)}
    values, state = verify_review_state(bundle)
    padding, attention, compare = load_contracts(bundle, neighbor)
    records, native = {}, {}
    for arm in ("T", "P"):
        records[arm], _digests = padding.read_run(bundle / "runs" / CASES[arm])
        native[arm] = padding.validate_run(records[arm])
    old = {}
    for label in ("TP", "F"):
        rows, _digests = attention.read_run(neighbor / "runs" / CASES[label])
        old[label] = attention.canonicalize_run(rows)
    require(old["TP"]["tp_size"] == 2 and old["F"]["tp_size"] == 1, "Wrong old native topologies")
    report = compare.compare_native(native["T"], native["P"], old["TP"], old["F"])
    scalars, states = {}, {}
    for label in CASES:
        directory = (bundle if label in ("T", "P") else neighbor) / "runs" / CASES[label]
        scalars[label], _digests, states[label] = compare.scalar_prefix(directory)
        expected_steps = 200 if label.endswith("200") else 3
        for rank in range(8):
            require(read_json(directory / f"complete-rank{rank}.json")["global_step"] == expected_steps,
                    "Recorded completion differs from the bounded run")
            require(len((directory / f"metrics-rank{rank}.jsonl").read_bytes().splitlines()) == expected_steps,
                    "The published scalar run is incomplete")
    t_checks = {label: compare.prefix_transparency(scalars["T"], scalars[label], states["T"], states[label])
                for label in ("TP", "TP200")}
    require(all(item["passed"] for item in t_checks.values()), "T did not preserve both original TP prefixes")
    for rank in range(8):
        for step in (1, 2, 3):
            require(all(scalars["F"][rank][step][key] == scalars["F200"][rank][step][key]
                        for key in (*compare.METRICS, "input_identity")),
                    "The old native FSDP scalar prefix differs from the baseline")
        for prefix in ("initial-state", "runtime-flags"):
            name = f"{prefix}-rank{rank}.json"
            require(states["F"][name] == states["F200"][name], "The FSDP reference initial/runtime state differs")
    report["own_T_three_step_transparency"] = t_checks["TP"]
    p_check = compare.prefix_transparency(scalars["P"], scalars["TP"], states["P"], states["TP"],
                                         ("training/lr", *compare.METRICS[6:]))
    require(p_check["passed"], "P's recipe/initial/runtime/input/counter control failed")
    report["P_recipe_initial_runtime_input_counter_coherence"] = p_check
    report["first_step_forward_interpretation_eligible"] = (
        report["native_T_control_passed"] and t_checks["TP"]["passed"] and p_check["passed"])
    lm = "training/foundation_loss/lm"
    report["first_step_main_LM_by_rank"] = []
    for rank in range(8):
        lf, lt, lp = (scalars[label][rank][1][lm] for label in ("F", "T", "P"))
        report["first_step_main_LM_by_rank"].append({
            "rank": rank, "L_F": lf, "L_T": lt, "L_P": lp,
            "delta_T": lt - lf, "delta_P": lp - lf, "P_minus_T": lp - lt})
    report["no_contribution_percentage_or_batch_extrapolation"] = True
    saved = read_json(bundle / "evidence/paired-comparison.json")
    require(report == saved == values["original_comparison_fields"],
            "Recomputed native/scalar comparison differs from the frozen scientific record")
    for arm in ("T", "P"):
        verify_numeric_recipe(values, states[arm]["config.resolved.json"])
        require(runtime_snapshot(native[arm]) == values["native_runtime_selected_records"][arm],
                "The frozen runtime statistics differ from the actual captured use-point")
    target = values["target"]
    require(target["batch_sha256"] == report["target_batch"] and target["module"] == padding.MODULE
            and target["step"] == 1 and target["micro"] == 0 and target["TP_ranks"] == [4, 5]
            and target["old_FSDP_rank_for_logical_batch"] == 2
            and target["logical_shapes"] == {"input": [4096, 1536], "weight": [24576, 1536],
                                                "output": [4096, 24576]}
            and target["complete_native_blocks"] == {"input": 96, "weight": 2304, "output": 1536},
            "The frozen target description differs from the complete runtime contract")
    execution = read_json(bundle / "evidence/execution-safe-summary.json")
    require(execution["numeric_snapshot_execution_summary"] == values["execution_summary"],
            "The safe release summary does not bind the frozen execution fields")
    for arm in ("T", "P"):
        require(execution["reviewed_release"][arm]["exit_code"] == 0
                and execution["reviewed_release"][arm]["released"] is True
                and execution["reviewed_release"][arm]["duration_seconds"]
                == values["execution_summary"][f"{arm}_duration_seconds"], "Release receipt extract is inconsistent")
    for label in ("F", "T", "P"):
        projected = {str(rank): [dict(scalars[label][rank][step]) for step in (1, 2, 3)] for rank in range(8)}
        expected = values["first_three_step_logged_scalars"][label]
        # The snapshot keeps exactly the listed scalar/counter/input fields, not the full callback row.
        projected = {rank: [{key: row[key] for key in expected[rank][index]} for index, row in enumerate(rows)]
                     for rank, rows in projected.items()}
        require(projected == expected, "A frozen three-step scalar excerpt differs from the captured metrics")
    first = report["first_step_main_LM_by_rank"][0]
    require(all({key: value for key, value in row.items() if key != "rank"}
                == {key: value for key, value in first.items() if key != "rank"}
                for row in report["first_step_main_LM_by_rank"]), "Global LM rank copies disagree")
    tile_counts = {name: len(item["blocks"]) for name, item in native["P"]["complete"].items()}
    require(tile_counts == {"input": 96, "weight": 2304, "original": 1536, "effective": 1536},
            "Unexpected target native coverage")
    require(report["status"] == "PAD_TARGET_NATIVE_MATCHES_OLD_FSDP"
            and all(report["native_conditions"].values()), "The complete native restoration did not pass")
    output = {
        "schema": "qb-padding-public-offline-reproduction-v1", "status": "PASS",
        "source_commit": SOURCE, "inventories": inventories,
        "target_batch": report["target_batch"], "target_native_tile_counts": tile_counts,
        "native_conditions": report["native_conditions"], "T_prefix_controls": t_checks,
        "P_recipe_initial_runtime_input_counter_coherence": p_check,
        "first_step_main_LM": first,
        "P_original_vs_effective_full_numeric": report["P_original_vs_effective_full_numeric"],
        "numeric_snapshot_fields_read": list(NUMERIC_SNAPSHOT_FIELDS),
        "historical_snapshot_review_status_used_as_current_approval": False,
        "paired_scientific_review": state["paired_scientific_review"],
        "publication_independent_review": state["publication_independent_review"],
        "numeric_proof_boundary": "Runtime full-element statistics are checked for consistency and binding; "
                                  "complete hashes cannot reconstruct tensor values or recompute their MSE.",
        "eight_rank_LM_copies_are_independent_samples": False,
        "root_cause": "UNRESOLVED", "training200_acceptance": False, "torch_imported": False,
    }
    return output, records, padding


def run_self_tests(records: dict, padding: ModuleType) -> list[str]:
    """Reject malformed evidence with pure JSON mutations; never alter captured files."""
    completed = []

    def rejected(name: str, mutation: Callable[[dict], None]) -> None:
        candidate = copy.deepcopy(records["P"])
        mutation(candidate)
        try:
            padding.validate_run(candidate)
        except (ValueError, KeyError, TypeError):
            completed.append(name)
        else:
            raise RuntimeError("Negative evidence was accepted: " + name)

    def result(rows: dict) -> dict:
        return next(row["result"] for row in rows[4] if row["kind"] == "intervention")

    rejected("missing_rank", lambda rows: rows.pop(7))
    rejected("duplicate_completion", lambda rows: rows[0].append(copy.deepcopy(rows[0][-1])))
    rejected("missing_completion", lambda rows: rows[0].pop())
    rejected("unverified_runtime", lambda rows: rows[0][0]["runtime_proof"].update(verified=False))
    rejected("wrong_source", lambda rows: rows[0][0].update(source_commit="0" * 40))
    rejected("wrong_target_batch", lambda rows: rows[0][0].update(target_batch="0" * 64))
    rejected("missing_native_tile", lambda rows: result(rows)["effective"]["blocks"].pop())
    rejected("duplicate_native_tile", lambda rows: result(rows)["effective"]["blocks"].append(
        copy.deepcopy(result(rows)["effective"]["blocks"][0])))
    rejected("wrong_global_offset", lambda rows: result(rows)["effective"]["bounds"].__setitem__(2, 128))
    rejected("wrong_native_dtype", lambda rows: result(rows)["effective"].update(dtype="torch.float32"))
    rejected("nonfinite_statistics", lambda rows: result(rows)["original_vs_effective"].update(max_abs=math.nan))
    rejected("inconsistent_nonzero_statistics", lambda rows: result(rows)["original_vs_effective"].update(
        max_abs=0.0, squared_error_sum=0.0, mse=0.0))
    rejected("incomplete_numeric_count", lambda rows: result(rows)["original_vs_effective"].update(
        compared_numel=1))
    rejected("out_of_bounds_maximum", lambda rows: result(rows)["original_vs_effective"].update(
        max_coordinate=[4096, 0]))
    rejected("unknown_forward_route", lambda rows: result(rows)["compute_weight_proof"][
        "actual_forward_route"].update(route="unknown_wrapper"))
    rejected("unproven_DT_return", lambda rows: result(rows)["return_contract"].update(DTensor_rewrap=True))
    rejected("different_live_weight", lambda rows: result(rows)["compute_weight_proof"]["after_restore"].update(
        local_tensor_id=1))
    rejected("graph_invariant_failed", lambda rows: result(rows)["invariants"].update(effective_graph_present=False))
    rejected("noncompact_storage", lambda rows: result(rows)["effective"]["physical"].update(storage_offset=1))
    rejected("unverified_original_source_identity", lambda rows: result(rows)["compute_weight_proof"][
        "actual_forward_route"]["torch_linear_source_identity"].update(file_identity_checked=False))
    return completed


def main() -> None:
    """Print a bounded scientific receipt; keep the candidate bundle read-only."""
    sys.dont_write_bytecode = True
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="Also reject twenty malformed JSON examples")
    args = parser.parse_args()
    bundle = Path(__file__).resolve().parents[1]
    output, records, padding = reproduce(bundle)
    if args.self_test:
        output["negative_tests_passed"] = run_self_tests(records, padding)
        output["negative_test_count"] = len(output["negative_tests_passed"])
    print(json.dumps(output, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
