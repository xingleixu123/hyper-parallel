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

"""Audit two independent 200-step trajectories without promoting partial results."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

ROOT = Path("/home/xxl/jt-adamw-align-20261013/phase12")
CONTROLS = {
    'hp-cann92-jtf637-adamw-'+scope+'-archive': 'hp-jtf637-adamw-'+scope+'-archive206'
    for scope in ['embedding', 'all']
}
JOBS = list(CONTROLS)


def analyze(name: str) -> dict:
    """Require all restoration, source, input and loss-window evidence independently."""
    control = CONTROLS[name]
    for filename in ("RESTORE.json", "SOURCE_REVISION.json"):
        actual = json.loads((ROOT / "runs" / name / filename).read_text())
        reference = json.loads((ROOT / "runs" / control / filename).read_text())
        if actual != reference:
            raise ValueError("Controlled restoration or source differs: " + filename)
    base = Path("/alignment/phase12")
    commands = [
        ["compare_inputs.py", "--mf", str(base / "runs/mf-resume/batches"),
         "--hp", str(base / "runs" / name / "batches"),
         "--output", str(base / "evidence" / (name + "-inputs.json"))],
        ["compare_loss_components.py", "--mf", str(base / "runs/mf-resume/metrics.jsonl"),
         "--hp", str(base / "runs" / name / "metrics.jsonl"),
         "--output", str(base / "evidence" / (name + "-components"))],
    ]
    with (ROOT / "runs" / (name + "-analysis.log")).open("w") as log:
        for index, command in enumerate(commands):
            args = ["/workspace/venv/bin/python", str(base / "tools" / command[0]), *command[1:]]
            shell = "source /workspace/env.sh\nexport OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1\n" + shlex.join(args)
            result = subprocess.run(
                ["docker", "-H", "unix:///run/hp-pr-two-commits-20261008/docker.sock", "exec",
                 "jt-adamw-hp-20261013", "bash", "-c", shell], stdout=log, stderr=subprocess.STDOUT, timeout=600)
            if result.returncode not in ({0} if index == 0 else {0, 1}):
                raise RuntimeError("Analysis process failed: " + command[0])
    inputs = json.loads((ROOT / "evidence" / (name + "-inputs.json")).read_text())
    losses = json.loads((ROOT / "evidence" / (name + "-components/component-comparison.json")).read_text())
    if not inputs["complete"] or inputs["batches"] != 200 or not inputs["all_supplied_inputs_equal"]:
        raise ValueError("Input window not fully audited")
    return {"run": name, "control": control, "restore_and_source_identical": True,
            "all200_inputs_equal": True, "losses": losses["losses"], "passed": losses["passed"]}


def main() -> None:
    """Analyze each completed run once; distinguish launch failures from numerical failures."""
    output = ROOT / "evidence/jt-f637-adamw-two-full200-analysis-status.json"
    if output.exists():
        raise FileExistsError(output)
    pending, completed, failed = list(JOBS), [], []
    deadline = time.monotonic() + 3 * 3600
    while pending and time.monotonic() < deadline:
        for name in list(pending):
            exit_path = ROOT / "runs" / (name + ".exit")
            if not exit_path.exists():
                continue
            try:
                if exit_path.read_text().strip() != "0":
                    raise ValueError("Training did not exit successfully")
                completed.append(analyze(name))
            except Exception as error:  # Preserve per-run diagnostics without losing the other finite analyses.
                failed.append({"run": name, "error": repr(error)})
            pending.remove(name)
        output.write_text(json.dumps({"pid": os.getpid(), "updated_at": time.time(), "pending": pending,
                                     "completed": completed, "failed": failed}, indent=2))
        if pending:
            time.sleep(45)
    if pending or failed:
        raise RuntimeError("Incomplete analysis; inspect the explicit per-run status")


if __name__ == "__main__":
    main()
