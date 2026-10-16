# Copyright 2026 Huawei Technologies Co., Ltd.
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

"""Apply the current total, raw-MTP and raw-Aux criteria to complete saved trajectories."""
import argparse
import hashlib
import json
from pathlib import Path

from compare_loss_components import read_metrics

ROOT = Path(__file__).resolve().parent
KEYS = ("total_loss", "mtp_raw", "aux_raw")
FIRST_TOLERANCES = {"total_loss": .0005, "mtp_raw": .001, "aux_raw": .001}


def main() -> None:
    """Reject missing steps, reference-value interventions and any failing component."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="hp-jte213-expertnorm328400")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    paths = {"hp": args.root / "runs" / args.run / "metrics.jsonl",
             "mf": args.root / "runs/mf-resume/metrics.jsonl"}
    metrics = {family: read_metrics(path, family, 1e-4, .3) for family, path in paths.items()}
    steps = list(range(201, 401))
    if any(set(values) != set(steps) for values in metrics.values()):
        raise ValueError("Exact complete201-400 window required")
    eligible = not any((path.parent / "DIAGNOSTIC_ONLY.json").exists() for path in paths.values())
    rows = [{"step": step, "errors": {key: abs(metrics["hp"][step][key] - metrics["mf"][step][key])
                                      for key in KEYS}} for step in steps]
    summary = {}
    for key in KEYS:
        values = [row["errors"][key] for row in rows]
        summary[key] = {"first": values[0], "max": max(values), "max_step": steps[values.index(max(values))],
                        "mean": sum(values) / 200, "last": values[-1],
                        "bad_steps": [step for step, value in zip(steps, values) if value > .005],
                        "first_tolerance": FIRST_TOLERANCES[key],
                        "passed": values[0] <= FIRST_TOLERANCES[key] and max(values) <= .005}
    passed = eligible and all(row["passed"] for row in summary.values())
    result = {"run": args.run, "criteria": {"first": FIRST_TOLERANCES, "every_step": .005, "metrics": KEYS},
              "eligible_for_acceptance": eligible, "numerical_passed": passed, "summary": summary,
              "source_sha256": {family: hashlib.sha256(path.read_bytes()).hexdigest()
                                for family, path in paths.items()}, "rows": rows,
              "scope": "Numerical gate only; full input, state and source fidelity must be audited separately."}
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({key: value for key, value in result.items() if key not in ["rows", "source_sha256"]}))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
