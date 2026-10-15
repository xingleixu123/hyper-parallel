# Copyright 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Check rejection of incomplete and contradictory copies of the collected evidence."""

import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path
from unittest import mock

from plot_fsdp_resume_chain import main as plot_main
from summarize_fsdp_resume import summarize_resume


ROOT = Path(__file__).resolve().parents[1]
BASELINE = "r129-f637-fp32-fsdp8-0-200-r4"
RESUMED = "r129-f637-fp32-fsdp8-200-400-r2"
MANIFESTS = ("f637-fsdp8-r4-complete-evidence-manifest.json",
             "f637-fsdp8-resume-r2-complete-evidence-manifest.json")


def refresh_entry(root: Path, manifest: Path, changed: Path) -> None:
    """Keep byte hashes valid when testing semantic contradictions."""
    data = json.loads(manifest.read_text())
    entry = next(item for item in data["files"] if item["path"] == changed.relative_to(root).as_posix())
    contents = changed.read_bytes()
    entry.update(bytes=len(contents), sha256=hashlib.sha256(contents).hexdigest())
    manifest.write_bytes((json.dumps(data) + "\n").encode())


def main() -> None:
    """Run copied-evidence counterexamples, cleaning each temporary tree afterward."""
    cases = ("valid", "missing_baseline_complete", "changed_baseline_metric", "short_baseline",
             "source_mismatch", "patch_mismatch", "failed_exit", "image_mismatch",
             "recipe_mismatch", "plot_missing_rank", "plot_output_over_input")
    results = []
    for case in cases:
        with tempfile.TemporaryDirectory(prefix="jt-resume-evidence-") as temporary:
            root = Path(temporary)
            (root / "scheduling").mkdir()
            for name in (BASELINE, RESUMED):
                shutil.copytree(ROOT / "runs" / name, root / "runs" / name)
                shutil.copyfile(ROOT / "scheduling" / (name + ".receipt.json"),
                                root / "scheduling" / (name + ".receipt.json"))
            baseline, resumed = root / "runs" / BASELINE, root / "runs" / RESUMED
            manifests = [root / name for name in MANIFESTS]
            for source, target in zip(MANIFESTS, manifests):
                shutil.copyfile(ROOT / "evidence" / source, target)
            if case == "missing_baseline_complete":
                (baseline / "complete-rank7.json").unlink()
            elif case in ("changed_baseline_metric", "short_baseline"):
                path = baseline / "metrics-rank7.jsonl"
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                if case == "changed_baseline_metric":
                    rows[198]["training/foundation_loss/lm"] += 100
                else:
                    rows = rows[-1:]
                path.write_bytes(("\n".join(json.dumps(row) for row in rows) + "\n").encode())
                if case == "short_baseline":
                    refresh_entry(root, manifests[0], path)
            elif case in ("source_mismatch", "patch_mismatch", "failed_exit"):
                name, value = {"source_mismatch": ("source-commit.txt", "f" * 40 + "\n"),
                               "patch_mismatch": ("source.patch", "# source patch mismatch\n"),
                               "failed_exit": ("EXIT_CODE", "1\n")}[case]
                path = resumed / name
                path.write_bytes(value.encode())
                refresh_entry(root, manifests[1], path)
            elif case == "image_mismatch":
                path = root / "scheduling" / (RESUMED + ".receipt.json")
                data = json.loads(path.read_text())
                data["image"] = "sha256:" + "a" * 64
                path.write_bytes(json.dumps(data).encode())
                refresh_entry(root, manifests[1], path)
            elif case == "recipe_mismatch":
                path = resumed / "config.resolved.json"
                data = json.loads(path.read_text())
                data["model"]["config"]["mtp_loss_factor"] = 0.7
                path.write_bytes(json.dumps(data).encode())
                refresh_entry(root, manifests[1], path)
            elif case == "plot_missing_rank":
                (resumed / "metrics-rank7.jsonl").unlink()
            error = None
            try:
                if case.startswith("plot_"):
                    output = resumed / "bad-output" if case == "plot_output_over_input" else root / "plot"
                    arguments = ["plot", str(baseline), str(resumed), str(output),
                                 "--baseline-manifest", str(manifests[0]),
                                 "--resumed-manifest", str(manifests[1])]
                    with mock.patch.object(sys, "argv", arguments):
                        plot_main()
                else:
                    result = summarize_resume(baseline, resumed, *manifests)
                    if result["problems"]:
                        raise ValueError("; ".join(result["problems"]))
            except (ValueError, FileNotFoundError) as exc:
                error = str(exc)
            if (case == "valid") != (error is None):
                raise AssertionError(f"Unexpected outcome: {case}: {error}")
            results.append({"case": case, "passed": True, "rejected_reason": error})
    report = {"scope": "离线证据工具反例；不是NPU精度验收", "checks": len(results), "results": results}
    (ROOT / "evidence/resume-evidence-helper-validation.json").write_bytes(
        (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode())
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
