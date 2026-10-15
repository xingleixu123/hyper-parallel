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

"""Compare observed MF and HP model input values with explicit label masking."""
import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    """Verify every paired batch; partial inspection is never final acceptance."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mf", type=Path, required=True)
    parser.add_argument("--hp", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    mf_files = {path.name for path in args.mf.glob("*.npz")}
    hp_files = {path.name for path in args.hp.glob("*.npz")}
    expected = {f"{step:06d}.npz" for step in range(200, 400)}
    complete = mf_files == hp_files == expected
    if not args.partial and not complete:
        raise ValueError("Expected exactly the batches for steps 201 through 400")
    rows = []
    for filename in sorted(mf_files & hp_files):
        with np.load(args.mf / filename) as mf, np.load(args.hp / filename) as hp:
            targets = mf["labels"].copy()
            targets[mf["loss_mask"] == 0] = -100
            checks = {
                "tokens": np.array_equal(mf["input_ids"], hp["input_ids"]),
                "targets_after_mask": np.array_equal(targets, hp["shift_labels"]),
                "boundaries": np.array_equal(np.unique(mf["actual_seq_len"][mf["actual_seq_len"] > 0]),
                                             hp["actual_seq_len"].reshape(-1)),
                "position_ids_supplied": np.array_equal(mf["position_ids"].reshape(-1),
                                                       hp["position_ids"].reshape(-1)),
            }
            rows.append({"step": int(filename[:-4]) + 1, **checks})
    report = {"complete": complete, "batches": len(rows), "all_supplied_inputs_equal":
              bool(rows) and all(all(value for key, value in row.items() if key != "step") for row in rows),
              "note": "Equal supplied position_ids do not prove both model implementations use them.",
              "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}))
    if not report["all_supplied_inputs_equal"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
