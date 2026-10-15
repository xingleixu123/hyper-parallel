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
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

ROOT = Path("/home/xxl/jt-adamw-align-20261013/phase12")
CONTROLS = {
    'hp-jtf637-adamw-all-sharedupfp32400': 'hp-jtf637-adamw-all-sharedupfp32206',
    'hp-jtf637-nativeadamw-sharedupfp32400': 'hp-cann92-jtf637-routercontig-denseadj-castall',
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
    """Analyze only complete jobs; never install a background polling loop."""
    results = []
    for name in JOBS:
        exit_path = ROOT / 'runs' / (name + '.exit')
        if not exit_path.exists() or exit_path.read_text().strip() != '0':
            raise ValueError('Training not complete: ' + name)
        factor = json.loads((ROOT / 'runs' / name / 'SHARED_UP_FP32_FACTOR.json').read_text())
        if factor['teacher_values_used'] or factor['source_sha256'] != hashlib.sha256(
                (ROOT / 'tools-f637/hp_shared_up_fp32_factor.py').read_bytes()).hexdigest():
            raise ValueError('Independent factor source failed validation')
        actual_source = json.loads((ROOT / 'runs' / name / 'SOURCE_REVISION.json').read_text())
        if actual_source.get('commit') != '9213a97147020fb75e7a98a40f7e7b636f98d349':
            raise ValueError('Unexpected integrated framework revision')
        results.append(analyze(name))
    output = ROOT / 'evidence/jt-f637-sharedup-two-full200-analysis.json'
    output.write_text(json.dumps({'complete': True, 'results': results}, indent=2))
    print(json.dumps({'complete': True, 'results': results}))


if __name__ == '__main__':
    main()
