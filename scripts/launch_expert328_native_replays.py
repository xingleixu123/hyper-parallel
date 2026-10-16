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

"""Launch independent true-boundary and neighboring-input normalization replays."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path('/home/xxl/jt-adamw-align-20261013/phase12')
STATE = Path('/dev/shm/jt-adamw-align-20261013/state/jt-recipe-20261014')


def main() -> None:
    """Require two free cards and validated native input/sum comparisons before launching."""
    audit = json.loads((ROOT / 'evidence/frontier328-329-dvm-schedule-screen.json').read_text())
    if not audit['complete'] or not all(any(row['all_exact'] and row['join'] == 'cores5'
            and row['lane_order'] == 'adjacent' for row in case['results']) for case in audit['records']):
        raise ValueError('Native reduction schedule was not reproduced')
    status = subprocess.run(['npu-smi', 'info'], check=True, capture_output=True, text=True).stdout
    cards = [card for card in range(8) if 'No running processes found in NPU ' + str(card) in status]
    if len(cards) < 2:
        raise ValueError('Two independent idle devices required')
    ports = subprocess.run(['ss', '-ltn'], check=True, capture_output=True, text=True).stdout
    if any(':' + str(port) + ' ' in ports for port in [31697, 31698]):
        raise ValueError('Replay rendezvous port occupied')
    if min(shutil.disk_usage(ROOT).free, shutil.disk_usage(STATE).free) < 2_000_000_000:
        raise ValueError('Insufficient metadata headroom')
    for index, step in enumerate([328, 329]):
        base = 'frontier328-hp-control-replay'
        run = 'frontier' + str(step) + '-hp-expert-native-replay'
        launcher = ROOT / ('run-' + run + '.sh')
        work = STATE / ('work-' + run)
        if any(path.exists() for path in [launcher, work, ROOT / 'runs' / (run + '.log')]):
            raise ValueError('Replay destination already exists')
        script = (ROOT / ('run-' + base + '.sh')).read_text().replace(base, run)
        script = script.replace('replay_hp_frontier328_muon.py', 'replay_hp_expert_native_schedule.py')
        script = script.replace('source /workspace/env.sh\n', 'source /workspace/env.sh\n'
                                + 'export P12_EXPERT_REPLAY_STEP=' + str(step) + '\n')
        script = re.sub(r'ASCEND_RT_VISIBLE_DEVICES=\d+', 'ASCEND_RT_VISIBLE_DEVICES=' + str(cards[index]), script)
        script = re.sub(r'--master_port=\d+', '--master_port=' + str(31697 + index), script)
        launcher.write_text(script)
        work.mkdir()
        command = ['docker', '-H', 'unix:///run/hp-pr-two-commits-20261008/docker.sock', 'exec',
                   'jt-adamw-hp-20261013', 'bash', '/alignment/phase12/' + launcher.name]
        with (ROOT / 'runs' / (run + '.dispatch.log')).open('wb') as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, start_new_session=True)
        files = [launcher, ROOT / 'tools-f637/replay_hp_expert_native_schedule.py',
                 ROOT / 'tools-f637/hp_expert_norm328_factor.py']
        row = {'run': run, 'pid': process.pid, 'card': cards[index], 'command': command,
               'started_at': datetime.now().astimezone().isoformat(),
               'files': [{'name': str(path.relative_to(ROOT)),
                          'sha256': hashlib.sha256(path.read_bytes()).hexdigest()} for path in files],
               'scope': 'Actual native control plus live DVM reduction; epsilon separate; no teacher values.'}
        (ROOT / 'evidence' / (run + '-launch.json')).write_text(json.dumps(row, indent=2))
        print(json.dumps(row))


if __name__ == '__main__':
    main()
