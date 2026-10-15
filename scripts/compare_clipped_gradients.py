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

"""Compare actual clipped MF and HP gradients through the audited offline conversion mapping."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np


def main() -> None:
    """Report individual parameters, preserving absolute and relative error semantics."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--mf-run", default="mf-grad201-full")
    args = parser.parse_args()
    root = Path("/state/jt-recipe-20261014")
    source = root / args.mf_run / "gradients"
    target = root / args.run / "gradients"
    spec = importlib.util.spec_from_file_location(
        "conversion", "/alignment/phase12/hyper-parallel/scripts/checkpoint_conversion/convert_mf_ckpt_to_hp.py")
    conversion = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conversion)
    inputs = {}
    for entry in json.loads((source / "manifest.json").read_text()):
        files = list(source.glob(entry["prefix"] + "_*.npy"))
        if len(files) != 1:
            raise ValueError(f"Unexpected gradient dump count for {entry['name']}")
        inputs[entry["name"]] = np.load(files[0], mmap_mode="r")
    reference = conversion.convert_model(inputs)
    records = []
    for entry in json.loads((target / "manifest.json").read_text()):
        name = entry["name"]
        if name not in reference:
            raise ValueError("Missing MF gradient: " + name)
        left = reference[name].reshape(-1)
        right = np.load(target / entry["file"], mmap_mode="r").reshape(-1)
        if left.shape != right.shape:
            raise ValueError("Shape mismatch: " + name)
        sum_left = sum_right = sum_error = dot = absolute_sum = maximum = 0.0
        unequal = 0
        for start in range(0, left.size, 1_000_000):
            a = left[start:start + 1_000_000].astype(np.float64)
            b = right[start:start + 1_000_000].astype(np.float64)
            error = a - b
            sum_left += float(np.dot(a, a))
            sum_right += float(np.dot(b, b))
            sum_error += float(np.dot(error, error))
            dot += float(np.dot(a, b))
            absolute_sum += float(np.abs(error).sum())
            maximum = max(maximum, float(np.abs(error).max()))
            unequal += int(np.count_nonzero(error))
        records.append({"name": name, "numel": left.size, "relative_l2": (sum_error / max(sum_left, 1e-300)) ** 0.5,
                        "max": maximum, "mean": absolute_sum / left.size, "unequal": unequal,
                        "mf_l2": sum_left ** 0.5, "hp_l2": sum_right ** 0.5,
                        "cosine": dot / max((sum_left * sum_right) ** 0.5, 1e-300)})
    report = {"scope": "Actual post-clip leaf optimizer gradients; scope is exactly the recorded manifest.",
              "mf_run": args.mf_run, "hp_run": args.run, "records": records}
    output = Path("/alignment/phase12/evidence") / (args.run + "-comparison.json")
    output.write_text(json.dumps(report, indent=2))
    for row in sorted(records, key=lambda row: row["relative_l2"], reverse=True):
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
