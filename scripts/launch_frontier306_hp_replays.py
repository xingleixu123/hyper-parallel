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

"""Replay current and candidate Q-up updates on two independent available devices."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path('/home/xxl/jt-adamw-align-20261013/phase12')
STATE = Path('/dev/shm/jt-adamw-align-20261013/state/jt-recipe-20261014')


def main() -> None:
    """Require faithful actual states before attributing intermediate differences."""
    audit = json.loads((ROOT / 'evidence/frontier306-state-comparison.json').read_text())
    if not audit['complete'] or not all(all(row.values()) for row in audit['fidelity'].values()):
        raise ValueError('Actual state observation fidelity failed')
    status = subprocess.run(['npu-smi', 'info'], check=True, capture_output=True, text=True).stdout
    if any('No running processes found in NPU ' + str(card) not in status for card in [0, 1]):
        raise ValueError('Replay device is busy')
    ports = subprocess.run(['ss', '-ltn'], check=True, capture_output=True, text=True).stdout
    if any(':' + str(port) + ' ' in ports for port in [31684, 31685]):
        raise ValueError('Rendezvous port is occupied')
    if min(shutil.disk_usage(ROOT).free, shutil.disk_usage(STATE).free) < 1_000_000_000:
        raise ValueError('Insufficient metadata headroom')
    for mode, card, port in [('control', 0, 31684), ('both', 1, 31685)]:
        run = 'frontier306-hp-' + mode + '-replay'
        launcher = ROOT / ('run-' + run + '.sh')
        work = STATE / ('work-' + run)
        if any(path.exists() for path in [launcher, work, ROOT / 'runs' / (run + '.log')]):
            raise ValueError('Existing replay destination')
        script = 'set -euo pipefail\nsource /workspace/env.sh\n'
        script += 'export ASCEND_RT_VISIBLE_DEVICES=' + str(card) + '\n'
        script += 'export PYTHONPATH=/alignment/phase12/tools-e213:/alignment/phase12/tools-f637:/workspace:' + '$' + '{PYTHONPATH:-}\n'
        script += 'export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CLOSE_MATMUL_K_SHIFT=0\n'
        script += 'export P12_FRONTIER306_REPLAY_SCOPE=' + mode + '\n'
        script += 'cd /state/jt-recipe-20261014/work-' + run + '\n'
        script += 'exec > /alignment/phase12/runs/' + run + '.log 2>&1\n'
        script += "trap 'code=$?; printf \"%s\\n\" \"$code\" > /alignment/phase12/runs/" + run + ".exit' EXIT\n"
        script += ('/state/jt-recipe-20261014/jt-e213-pr192-venv/bin/python -m torch.distributed.run '
                   '--nproc_per_node=1 --master_addr=127.0.0.1 --master_port=' + str(port) +
                   ' /alignment/phase12/tools-f637/replay_hp_frontier306_muon.py\n')
        launcher.write_text(script)
        work.mkdir()
        command = ['docker', '-H', 'unix:///run/hp-pr-two-commits-20261008/docker.sock', 'exec',
                   'jt-adamw-hp-20261013', 'bash', '/alignment/phase12/' + launcher.name]
        with (ROOT / 'runs' / (run + '.dispatch.log')).open('wb') as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, start_new_session=True)
        row = {'run': run, 'pid': process.pid, 'card': card, 'port': port, 'command': command,
               'started_at': datetime.now().astimezone().isoformat(), 'mode': mode,
               'launcher_sha256': hashlib.sha256(launcher.read_bytes()).hexdigest(),
               'source_sha256': hashlib.sha256((ROOT / 'tools-f637/replay_hp_frontier306_muon.py').read_bytes()).hexdigest()}
        (ROOT / 'evidence' / (run + '-launch.json')).write_text(json.dumps(row, indent=2))
        print(json.dumps(row))


if __name__ == '__main__':
    main()
