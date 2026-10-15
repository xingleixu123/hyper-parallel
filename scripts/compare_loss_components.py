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
"""Compare explicitly normalized MF/HP loss components over an exact step window."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def scalar(value: object) -> float:
    """Accept a finite scalar or singleton nested list, without averaging away differences."""
    while isinstance(value, list) and len(value) == 1:
        value = value[0]
    if isinstance(value, (list, dict, bool)) or value is None:
        raise ValueError(f"Expected scalar loss, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Nonfinite loss")
    return result


def read_metrics(path: Path, family: str, aux_coefficient: float, mtp_factor: float) -> dict:
    """Read the existing observation contract and preserve both raw and weighted losses."""
    records = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        step = int(record["step"])
        if step in records:
            raise ValueError(f"Duplicate {family} step {step}")
        component = record["components"]
        if family == "mf":
            total = scalar(record["outputs"][0])
            lm = scalar(component["lm_loss"])
            mtp = scalar(component["mtp_loss"])
            aux_raw = scalar(component["aux_loss"])
            aux = aux_raw * aux_coefficient
        else:
            total = scalar(record["loss"])
            lm = scalar(component["foundation_loss/lm"])
            mtp = scalar(component["foundation_loss/mtp"])
            aux = scalar(component["foundation_loss/aux"])
            aux_raw = aux / aux_coefficient
        recomposition = abs(total - (lm + mtp + aux))
        if recomposition > 2e-5:
            raise ValueError(f"{family}/{step}: component semantics inconsistent with total: {recomposition}")
        records[step] = {"total_loss": total, "lm_loss": lm, "mtp_weighted": mtp,
                         "mtp_raw": mtp / mtp_factor, "aux_weighted": aux, "aux_raw": aux_raw,
                         "recomposition_error": recomposition}
    return records


def main() -> None:
    """Fail on missing steps, invalid metrics, or any requested loss threshold violation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mf", type=Path, required=True)
    parser.add_argument("--hp", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--aux-coefficient", type=float, default=0.0001)
    parser.add_argument("--mtp-factor", type=float, default=0.3)
    parser.add_argument("--start", type=int, default=201)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--first-tolerance", type=float, default=0.001)
    parser.add_argument("--window-tolerance", type=float, default=0.005)
    args = parser.parse_args()
    if args.aux_coefficient <= 0 or args.mtp_factor <= 0 or args.steps < 1:
        raise ValueError("This acceptance requires active auxiliary and MTP objectives")
    mf = read_metrics(args.mf, "mf", args.aux_coefficient, args.mtp_factor)
    hp = read_metrics(args.hp, "hp", args.aux_coefficient, args.mtp_factor)
    steps = list(range(args.start, args.start + args.steps))
    if set(mf) != set(steps) or set(hp) != set(steps):
        raise ValueError(f"Expected exact window {steps[0]}-{steps[-1]}, got MF {len(mf)}, HP {len(hp)}")
    columns = ["total_loss", "lm_loss", "mtp_weighted", "mtp_raw", "aux_weighted", "aux_raw"]
    report = {"steps": steps, "first_tolerance": args.first_tolerance,
              "window_tolerance": args.window_tolerance, "aux_coefficient": args.aux_coefficient,
              "mtp_factor": args.mtp_factor, "losses": {},
              "source_sha256": {family: hashlib.sha256(path.read_bytes()).hexdigest()
                                for family, path in (("mf", args.mf), ("hp", args.hp))}}
    rows = []
    for name in columns:
        differences = [abs(hp[step][name] - mf[step][name]) for step in steps]
        report["losses"][name] = {
            "first": differences[0], "last": differences[-1], "max": max(differences),
            "max_step": steps[differences.index(max(differences))], "mean": sum(differences) / len(steps),
            "over_window_tolerance": sum(value > args.window_tolerance for value in differences),
            "passed": differences[0] <= args.first_tolerance and max(differences) <= args.window_tolerance,
        }
        rows.extend({"component": name, "step": step, "mf": mf[step][name], "hp": hp[step][name],
                     "absolute_error": difference} for step, difference in zip(steps, differences))
    report["numerical_criteria_passed"] = all(value["passed"] for value in report["losses"].values())
    diagnostic_markers = {side: path.parent / "DIAGNOSTIC_ONLY.json" for side, path in
                          [("mf", args.mf), ("hp", args.hp)]}
    interventions = {side: json.loads(path.read_text()) for side, path in diagnostic_markers.items()
                     if path.exists()}
    report["eligible_for_acceptance"] = not interventions
    report["passed"] = report["numerical_criteria_passed"] and report["eligible_for_acceptance"]
    if interventions:
        report["diagnostic_interventions"] = interventions
        if "hp" in interventions:
            report["diagnostic_intervention"] = interventions["hp"]
    report["scope"] = "All raw and weighted component criteria; no claim about input/state equality."
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "component-comparison.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (args.output / "component-curves.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"passed": report["passed"], "losses": report["losses"]}, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
