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

"""Compare active target native tiles and first-update-free LM deltas, without replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from qb_padding_contract import MODULE, SCHEMA, SOURCE_COMMIT, frozen_support, read_run, validate_run


METRICS = ("training/total_loss", "training/foundation_loss/lm", "training/foundation_loss/mtp",
           "training/foundation_loss/aux", "training/grad_norm", "training/lr", "data/step_tokens",
           "data/consumed_tokens", "data/step_samples", "data/consumed_samples")
RECIPE_FIELDS = ("model", "optimizer", "lr_scheduler", "loss_fn", "model_init_dtype", "training",
                 "accelerator", "fsdp_config",
                 "mixed_precision", "dataset", "dataloader", "packed_sequence", "plan_overrides",
                 "activation_checkpoint", "activation_swap", "compile", "checkpoint", "debug",
                 "model_integration", "memory", "wandb", "profiler", "magi", "peft")


def compare_native(control: Mapping[str, Any], padded: Mapping[str, Any], old_tp: Mapping,
                   old_fsdp: Mapping) -> dict[str, Any]:
    """Require a calibrated T before interpreting P's one-batch native output."""
    if (control["arm"] != "T" or padded["arm"] != "P"
            or control["target_batch"] != padded["target_batch"] or control["scope"] != padded["scope"]):
        raise ValueError("Need separate same-target T and P arms")
    batch = control["target_batch"]
    conditions = {}

    def same(new: Mapping, old: Mapping) -> bool:
        if new["logical_shape"] != old["logical_shape"] or set(new["blocks"]) != set(old["blocks"]):
            raise ValueError("Incomplete/incompatible complete native logical coverage")
        return new["dtype"] == old["dtype"] and new["blocks"] == old["blocks"]

    for label, run in (("T", control), ("P", padded)):
        for name in ("input", "weight"):
            conditions[f"{label}_{name}_equals_old_TP2"] = same(
                run["complete"][name], old_tp["complete"][(batch, MODULE, name)])
            conditions[f"{label}_{name}_equals_old_FSDP"] = same(
                run["complete"][name], old_fsdp["complete"][(batch, MODULE, name)])
        conditions[f"{label}_original_equals_old_TP2"] = same(
            run["complete"]["original"], old_tp["complete"][(batch, MODULE, "output")])
    conditions["T_effective_equals_old_TP2"] = same(
        control["complete"]["effective"], old_tp["complete"][(batch, MODULE, "output")])
    conditions["P_effective_equals_old_FSDP"] = same(
        padded["complete"]["effective"], old_fsdp["complete"][(batch, MODULE, "output")])
    calibrated = all(value for key, value in conditions.items() if key != "P_effective_equals_old_FSDP")
    status = ("T_OR_INPUT_CONTROL_FAILED_NO_P_INFERENCE" if not calibrated else
              "PAD_TARGET_NATIVE_MATCHES_OLD_FSDP" if conditions["P_effective_equals_old_FSDP"] else
              "PAD_TARGET_DID_NOT_RESTORE_OLD_FSDP_BYTES")
    return {"schema": SCHEMA + "-comparison", "source_commit": SOURCE_COMMIT, "status": status,
            "scope": control["scope"], "module": MODULE, "target_batch": batch,
            "native_conditions": conditions, "native_T_control_passed": calibrated,
            "own_T_three_step_transparency": "REQUIRES_COMPLETE_SCALAR_STATE_PREFIX_CHECK",
            "P_original_vs_effective_full_numeric": [row["result"]["original_vs_effective"]
                                                    for row in padded["selected"]],
            "root_cause": "UNRESOLVED", "training_repair_claimed": False,
            "limits": ["One selected logical batch, both TP peers, layer0 q_b only.",
                       "P changes backward shapes as well; only first-step forward LM has causal scope.",
                       "Padding changes allocation/layout and contents; no kernel identity or unique cause inferred."]}


def scalar_prefix(directory: Path) -> tuple[list[dict], dict[str, str], dict[str, Any]]:
    """Read exact first-three-step scalars and state identities; never invent completion."""
    rows, digests, states = [], {}, {}
    for rank in range(8):
        path = directory / f"metrics-rank{rank}.jsonl"
        raw = path.read_bytes()
        parsed = [json.loads(line) for line in raw.splitlines()]
        by_step = {row["step"]: row for row in parsed}
        if len(parsed) != len(by_step) or not {1, 2, 3}.issubset(by_step):
            raise ValueError("Missing/repeated first-three-step metrics")
        if any(row["rank"] != rank or any(not math.isfinite(row[key]) for key in METRICS)
               for row in parsed):
            raise ValueError("Wrong rank or nonfinite metric")
        rows.append({step: by_step[step] for step in (1, 2, 3)})
        digests[str(path)] = hashlib.sha256(raw).hexdigest()
        for name in (f"initial-state-rank{rank}.json", f"runtime-flags-rank{rank}.json",
                     f"start-rank{rank}.json", f"complete-rank{rank}.json"):
            raw = (directory / name).read_bytes()
            digests[str(directory / name)] = hashlib.sha256(raw).hexdigest()
            value = json.loads(raw)
            if name.startswith(("initial-state", "runtime-flags")):
                states[name] = value
            if name.startswith("start-") and value["global_step"] != 0:
                raise ValueError("First-step arm must start at zero")
            if name.startswith("complete-") and value["global_step"] < 3:
                raise ValueError("Incomplete three-step execution")
    for name in ("EXIT_CODE", "source-commit.txt", "source-status.txt", "source.patch", "config.resolved.json"):
        raw = (directory / name).read_bytes()
        digests[str(directory / name)] = hashlib.sha256(raw).hexdigest()
        if name == "EXIT_CODE" and raw.strip() != b"0":
            raise ValueError("The captured run did not exit successfully")
        if name == "source-commit.txt" and raw.decode().strip() != SOURCE_COMMIT:
            raise ValueError("The actual source is not fixed544")
        if name in ("source-status.txt", "source.patch") and raw:
            raise ValueError("The captured training checkout was not clean")
        if name == "config.resolved.json":
            states[name] = json.loads(raw)
    return rows, digests, states


def prefix_transparency(actual: list[dict], old: list[dict], states: Mapping, old_states: Mapping,
                        metrics: tuple[str, ...] = METRICS) -> dict[str, Any]:
    """Require T's complete3x8 scalars/counters/input and CPU/runtime state equality."""
    different = {key: 0 for key in (*metrics, "input_identity")}
    for rank in range(8):
        for step in (1, 2, 3):
            for key in different:
                different[key] += actual[rank][step][key] != old[rank][step][key]
    state_differences = []
    for rank in range(8):
        for prefix in ("initial-state", "runtime-flags"):
            name = f"{prefix}-rank{rank}.json"
            if states[name] != old_states[name]:
                state_differences.append(name)
    for name in RECIPE_FIELDS:
        actual_recipe, old_recipe = states["config.resolved.json"][name], old_states["config.resolved.json"][name]
        if name == "training":
            if actual_recipe["train_iters"] != 3 or old_recipe["train_iters"] < 3:
                state_differences.append("bounded_training_window")
            actual_recipe = {key: value for key, value in actual_recipe.items() if key != "train_iters"}
            old_recipe = {key: value for key, value in old_recipe.items() if key != "train_iters"}
        elif name == "checkpoint":
            actual_recipe = {key: value for key, value in actual_recipe.items() if key != "checkpoint_dir"}
            old_recipe = {key: value for key, value in old_recipe.items() if key != "checkpoint_dir"}
        if actual_recipe != old_recipe:
            state_differences.append("numeric_recipe:" + name)
    return {"passed": not any(different.values()) and not state_differences,
            "three_step_8rank_pairs_per_metric": 24, "mismatches": different,
            "CPU_initial_runtime_state_differences": state_differences}


def main() -> None:
    """Create a separate diagnostic receipt; T must pass before P is interpreted."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--padded", type=Path, required=True)
    parser.add_argument("--old-tp2", type=Path, required=True)
    parser.add_argument("--old-fsdp", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for directory in (args.control, args.padded, args.old_tp2, args.old_fsdp):
        if args.output.resolve() == directory.resolve() or directory.resolve() in args.output.resolve().parents:
            raise ValueError("Derived output must be outside all four captured run directories")
    control_rows, control_sha = read_run(args.control)
    padded_rows, padded_sha = read_run(args.padded)
    control, padded = validate_run(control_rows), validate_run(padded_rows)
    old_contract = frozen_support().frozen_contract()
    old_tp_rows, old_tp_sha = old_contract.read_run(args.old_tp2)
    old_fsdp_rows, old_fsdp_sha = old_contract.read_run(args.old_fsdp)
    old_tp, old_fsdp = old_contract.canonicalize_run(old_tp_rows), old_contract.canonicalize_run(old_fsdp_rows)
    if old_tp["tp_size"] != 2 or old_fsdp["tp_size"] != 1:
        raise ValueError("The old sources must be the accepted native TP2/FSDP attention pair")
    report = compare_native(control, padded, old_tp, old_fsdp)
    scalars, hashes, states = {}, {}, {}
    for label, directory in (("T", args.control), ("P", args.padded), ("TP", args.old_tp2), ("F", args.old_fsdp)):
        scalars[label], hashes[label], states[label] = scalar_prefix(directory)
    transparency = prefix_transparency(scalars["T"], scalars["TP"], states["T"], states["TP"])
    report["own_T_three_step_transparency"] = transparency
    coherence = prefix_transparency(scalars["P"], scalars["TP"], states["P"], states["TP"],
                                    ("training/lr", *METRICS[6:]))
    report["P_recipe_initial_runtime_input_counter_coherence"] = coherence
    eligible = report["native_T_control_passed"] and transparency["passed"] and coherence["passed"]
    report["first_step_forward_interpretation_eligible"] = eligible
    report["first_step_main_LM_by_rank"] = []
    key = "training/foundation_loss/lm"
    for rank in range(8):
        lf, lt, lp = (scalars[name][rank][1][key] for name in ("F", "T", "P"))
        report["first_step_main_LM_by_rank"].append(
            {"rank": rank, "L_F": lf, "L_T": lt, "L_P": lp, "delta_T": lt - lf,
             "delta_P": lp - lf, "P_minus_T": lp - lt})
    if not eligible:
        report["status"] = "CONTROL_OR_P_COHERENCE_FAILED_NO_P_INFERENCE"
    report["input_sha256"] = {"T": control_sha, "P": padded_sha, "old_TP": old_tp_sha,
                              "old_FSDP": old_fsdp_sha, "scalar_state_sources": hashes}
    report["no_contribution_percentage_or_batch_extrapolation"] = True
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": report["status"], "root_cause": "UNRESOLVED", "eligible": eligible}))


if __name__ == "__main__":
    main()
