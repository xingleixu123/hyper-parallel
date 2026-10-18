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
"""Reproduce public native-byte comparisons and three-step prefixes, without Torch."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ATTENTION = ROOT / "evidence/a1e6-attention0-forward-bisection-v1"
SHADOW = ROOT / "evidence/a1e6-qb-shadow-bisection-v1"
SOURCE = "544526f7db8342c607244fd41009b2a74adb84c8"
BASELINE = "r129-a1e6-fsdp8-0-200-v3"
TP_CONTROL = "r129-a1e6-tp2-200-v2"
PAIRS = (
    ("r129-a1e6-attention0-forward-fsdp8-3-v1", BASELINE),
    ("r129-a1e6-attention0-forward-tp2-3-v1", TP_CONTROL),
    ("r129-a1e6-qb-shadow-fsdp8-3-v1", BASELINE),
)


def _load(name: str, path: Path) -> Any:
    """Import an already manifest-verified local module without running its CLI."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load {path.name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _json(path: Path) -> Any:
    return json.loads(path.read_bytes())


def _metrics(case: str) -> dict[int, list[dict]]:
    return {rank: [json.loads(line) for line in
                   (ROOT / "runs" / case / f"metrics-rank{rank}.jsonl").read_bytes().splitlines()]
            for rank in range(8)}


def _scientific(report: dict) -> dict:
    """Exclude only CLI input path bookkeeping and explicit public-extract metadata."""
    data = {key: value for key, value in report.items() if key not in
            {"input_sha256", "tool_sha256", "public_extract"}}
    # The frozen comparer uses tuples for K partitions; saved JSON uses arrays.
    return json.loads(json.dumps(data, allow_nan=False))


def main() -> None:
    """Check the complete public manifest and replay all native and prefix contracts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plots", action="store_true", help="Also replay both plots in RAM; needs Matplotlib/Pillow")
    args = parser.parse_args()
    if "torch" in sys.modules or "torch_npu" in sys.modules:
        raise RuntimeError("Use a fresh Python process without Torch")
    manifest = _json(ROOT / "manifest.json")
    files = manifest["files"]
    actual = {path.relative_to(ROOT).as_posix() for path in ROOT.rglob("*") if path.is_file()}
    if actual != {row["path"] for row in files} | {"manifest.json"}:
        raise ValueError("Public manifest does not cover the exact package")
    for row in files:
        path = (ROOT / row["path"]).resolve()
        if ROOT not in path.parents:
            raise ValueError("Manifest path escapes the package")
        raw = path.read_bytes()
        if len(raw) != row["bytes"] or hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise ValueError(f"Manifest identity mismatch: {row['path']}")
    sys.path.insert(0, str(SHADOW / "tools"))
    qb = _load("qb_shadow_contract", SHADOW / "tools/qb_shadow_contract.py")
    contract = qb.frozen_contract()
    native = _load("public_compare_qb", SHADOW / "tools/compare_qb_shadow.py")
    attention = _load("public_compare_attention", ATTENTION / "tools/compare_attention0_forward_runs.py")
    prefix = _load("public_compare_prefix", ROOT /
                   "evidence/a1e6-native-forward-bisection-v1/tools/compare_native_forward_runs.py")
    transparency = {}
    old_prefix = _json(ATTENTION / "own-prefix-root-review-v1.json")
    shadow_prefix = _json(SHADOW / "own-prefix-root-review-v1.json")
    for case, original in PAIRS:
        result = prefix.check_transparency(_metrics(case), _metrics(original))
        expected = (shadow_prefix["transparent"] if "qb-shadow" in case else
                    old_prefix["runs"]["tp2" if "-tp2-" in case else "fsdp8"]["transparent"])
        if result != expected or result["status"] != "PASS":
            raise ValueError(f"Three-step prefix mismatch: {case}")
        for rank in range(8):
            for name in ("initial-state", "runtime-flags"):
                if _json(ROOT / "runs" / case / f"{name}-rank{rank}.json") != _json(
                        ROOT / "runs" / original / f"{name}-rank{rank}.json"):
                    raise ValueError(f"Initial/runtime mismatch: {case}/{name}/{rank}")
            left = _json(ROOT / "runs" / case / f"start-rank{rank}.json")
            right = _json(ROOT / "runs" / original / f"start-rank{rank}.json")
            left.pop("train_iters")
            right.pop("train_iters")
            if left != right:
                raise ValueError(f"Start metadata mismatch: {case}/{rank}")
        transparency[case] = result
    for case in {name for pair in PAIRS for name in pair}:
        folder = ROOT / "runs" / case
        if ((folder / "source-commit.txt").read_text().strip() != SOURCE or
                (folder / "source-status.txt").read_bytes() or (folder / "source.patch").read_bytes() or
                (folder / "EXIT_CODE").read_text().strip() != "0"):
            raise ValueError(f"Source/exit identity differs: {case}")
        step = 3 if "-3-v1" in case else 200
        if any(_json(folder / f"complete-rank{rank}.json")["global_step"] != step for rank in range(8)):
            raise ValueError(f"Incomplete run: {case}")
    left_rows, _ = contract.read_run(ROOT / "runs" / PAIRS[0][0])
    right_rows, _ = contract.read_run(ROOT / "runs" / PAIRS[1][0])
    left, right = contract.canonicalize_run(left_rows), contract.canonicalize_run(right_rows)
    observed = attention.compare_complete_states(left, right)
    if _scientific(observed) != _scientific(_json(ATTENTION / "paired-comparison-v1.json")):
        raise ValueError("Attention complete-native comparison differs")
    rows, _ = qb.read_rank_files(ROOT / "runs" / PAIRS[2][0])
    validated = qb.validate_run(rows)
    result = native.compare_to_existing(validated, left, right)
    if _scientific(result) != _scientific(_json(SHADOW / "paired-comparison-v1.json")):
        raise ValueError("Width shadow comparison differs")
    if "torch" in sys.modules or "torch_npu" in sys.modules:
        raise RuntimeError("The offline comparison imported Torch")
    figure_result = None
    if args.plots:
        figures = _load("public_reproduce_figures", ROOT / "tools/reproduce_qb_native_width_figures.py")
        figure_result = figures.reproduce_figures(ROOT)
    print(json.dumps({"status": "PASS_PUBLIC_REPRODUCTION_AUTHOR_QA_NOT_200_ACCEPTANCE",
                      "manifest_entries": len(files), "own_prefix_pairs": 3,
                      "rank_step_pairs_each": 24, "initial_runtime_start_pairs_each": 8,
                      "attention_first_observed_difference": observed["first_observed_complete_difference"],
                      "shadow_status": result["status"], "root_cause": result["root_cause"],
                      "full_TP_numeric_error": result["full_TP_numeric_error"],
                      "figure_RAM_replays": figure_result,
                      "Torch_imported": False, "outputs_written": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
