"""Audit complete, all-rank training traces using absolute per-step error gates."""

import argparse
import json
import math
import statistics
from pathlib import Path


LOSSES = ("training/total_loss", "training/foundation_loss/lm",
          "training/foundation_loss/mtp", "training/foundation_loss/aux")


def read_rows(directory, rank):
    rows = [json.loads(line) for line in (directory / f"metrics-rank{rank}.jsonl").read_text().splitlines()]
    if len({row["step"] for row in rows}) != len(rows):
        raise ValueError(f"Duplicate steps in {directory}, rank {rank}")
    return {row["step"]: row for row in rows}


def measurement_context(reference, candidate):
    """Require matching lightweight observations before comparing model performance."""
    modes = {}
    reasons = []
    for directory in (reference, candidate):
        config_path = directory / "config.resolved.json"
        config = json.loads(config_path.read_text()) if config_path.exists() else {}
        mode = config.get("model_integration", {}).get("mode", "unknown")
        modes[directory.name] = mode
        if mode != "build":
            reasons.append(f"{directory.name}: model_integration mode is {mode}")
        patterns = ("stage-profile-rank*.jsonl", "module-profile-rank*.jsonl", "expert-samples.jsonl",
                    "routing-rank*.jsonl", "ns-batches-rank*.jsonl", "ns-single-matrix-rank*.jsonl",
                    "activations-rank*.jsonl", "projection-digests-rank*.jsonl", "loss-gradients-rank*.jsonl",
                    "*-calls-rank*.jsonl", "stable-clip-rank*.jsonl")
        found = [pattern for pattern in patterns if any(directory.rglob(pattern))]
        if found:
            reasons.append(f"{directory.name}: diagnostic observations: {found}")
        if config.get("profiler", {}).get("enabled") or config.get("memory", {}).get("enable"):
            reasons.append(f"{directory.name}: configured profiler/memory capture is enabled")
    return {"model_integration_modes": modes,
            "eligible_for_matching_build_mode_performance_comparison": not reasons,
            "excluded_reasons": reasons,
            "common_observations": "Per-step scalar metrics and input identity capture, including NPU-to-CPU copies/hash work.",
            "scope": "Build-mode comparisons retain matching input evidence overhead; they are not raw uninstrumented Trainer latency. Numerical gates are independent."}


def audit(reference, candidate, start, end, tolerance=0.005):
    steps = list(range(start, end + 1))
    findings = []
    errors = {key: {"max_abs": 0.0, "step": start, "rank": 0} for key in (*LOSSES, "training/grad_norm")}
    input_mismatches = []
    initial_mismatches = []
    step_errors = {key: [0.0] * len(steps) for key in LOSSES}
    for directory in (reference, candidate):
        exit_path = directory / "EXIT_CODE"
        if not exit_path.exists() or exit_path.read_text().strip() != "0":
            findings.append(f"Run is not successfully complete: {directory.name}")
    reference_initial = json.loads((reference / "initial-state-rank0.json").read_text())
    candidate_times = []
    supervised_tokens = []
    maximum_memory = {"allocated_gib": 0.0, "reserved_gib": 0.0}
    for rank in range(8):
        expected, actual = read_rows(reference, rank), read_rows(candidate, rank)
        if any(step not in expected or step not in actual for step in steps):
            findings.append(f"Missing required steps at rank {rank}")
            continue
        if sorted(actual) != steps:
            findings.append(f"Candidate window differs from requested window at rank {rank}")
        for directory in (reference, candidate):
            if json.loads((directory / f"initial-state-rank{rank}.json").read_text()) != reference_initial:
                initial_mismatches.append([directory.name, rank])
            complete = directory / f"complete-rank{rank}.json"
            if not complete.exists() or json.loads(complete.read_text())["global_step"] < end:
                findings.append(f"Missing completion marker: {directory.name}, rank {rank}")
        initial_step = json.loads((candidate / f"start-rank{rank}.json").read_text())["global_step"]
        if initial_step != start - 1:
            findings.append(f"Candidate starts at {initial_step}, expected {start - 1}, rank {rank}")
        for index, step in enumerate(steps):
            a, b = expected[step], actual[step]
            if a["input_identity"]["global_sha256"] != b["input_identity"]["global_sha256"]:
                input_mismatches.append([rank, step])
            for key, record in errors.items():
                difference = abs(a[key] - b[key])
                if not math.isfinite(difference):
                    findings.append(f"Non-finite {key}, rank {rank}, step {step}")
                if difference > record["max_abs"]:
                    record.update(max_abs=difference, step=step, rank=rank)
                if key in step_errors:
                    step_errors[key][index] = max(step_errors[key][index], difference)
            if a["training/lr"] != b["training/lr"]:
                findings.append(f"Learning rate differs at step {step}, rank {rank}")
            maximum_memory["allocated_gib"] = max(maximum_memory["allocated_gib"], b["memory/device_max_allocated_gb"])
            maximum_memory["reserved_gib"] = max(maximum_memory["reserved_gib"], b["memory/device_max_reserved_gb"])
            if rank == 0 and index >= 20:
                candidate_times.append(b["performance/step_time"])
                supervised_tokens.append(b["data/step_tokens"])
    passed = (not findings and not input_mismatches and not initial_mismatches
              and all(errors[key]["max_abs"] <= tolerance for key in LOSSES))
    median_seconds = statistics.median(candidate_times) if candidate_times else None
    return {
        "reference": reference.name, "candidate": candidate.name, "steps": [start, end], "ranks": 8,
        "tolerance_absolute": tolerance, "status": "PASS" if passed else "FAIL", "findings": findings,
        "errors": errors, "input_mismatch_count": len(input_mismatches), "input_mismatch_examples": input_mismatches[:8],
        "initial_mismatches": initial_mismatches, "step_errors": step_errors,
        "measurement_context": measurement_context(reference, candidate),
        "loss_units": {"lm": "unweighted", "mtp": "weighted by 0.3", "aux": "weighted by 0.0001"},
        "unweighted_error_diagnostics": {
            "mtp": errors["training/foundation_loss/mtp"]["max_abs"] / 0.3,
            "aux": errors["training/foundation_loss/aux"]["max_abs"] / 0.0001,
        },
        "performance": {"warmup_steps_excluded": 20, "median_step_seconds": median_seconds,
                        "processed_tokens_per_second": 32768 / median_seconds if median_seconds else None,
                        "supervised_tokens_per_second": (sum(supervised_tokens) / sum(candidate_times)
                                                         if candidate_times else None),
                        **maximum_memory},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("start", type=int)
    parser.add_argument("end", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.reference, args.candidate, args.start, args.end)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "step_errors"}, indent=2))
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
