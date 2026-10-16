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

"""Retry the HP state observer after installing the existing norm factor first."""
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
    """Require an idle device and new output paths, then launch one observer."""
    run = 'hp-jtf637-qup-state293-r3'
    base = 'hp-jtf637-qup-state293-r2'
    card, port = 4, 31672
    status = subprocess.run(['npu-smi', 'info'], check=True, capture_output=True, text=True).stdout
    if 'No running processes found in NPU ' + str(card) not in status:
        raise ValueError('Requested device is not idle')
    if shutil.disk_usage(STATE).free < 5_000_000_000:
        raise ValueError('Insufficient selected-state headroom')
    ports = subprocess.run(['ss', '-ltn'], check=True, capture_output=True, text=True).stdout
    if ':' + str(port) + ' ' in ports:
        raise ValueError('Rendezvous port is occupied')
    targets = [ROOT / 'runs' / run, ROOT / ('run-' + run + '.sh'),
               ROOT / 'configs' / (run + '.yaml'), STATE / ('work-' + run)]
    if any(path.exists() for path in targets):
        raise ValueError('Existing run destination')
    entry = 'capture_hp_frontier293_state_r2.py'
    config = (ROOT / 'configs' / (base + '.yaml')).read_text().replace(base, run)
    script = (ROOT / ('run-' + base + '.sh')).read_text().replace(base, run)
    script = script.replace('capture_hp_frontier293_state.py', entry)
    script = re.sub(r'ASCEND_RT_VISIBLE_DEVICES=\d+', 'ASCEND_RT_VISIBLE_DEVICES=' + str(card), script)
    script = re.sub(r'--master_port=\d+', '--master_port=' + str(port), script)
    targets[2].write_text(config)
    targets[1].write_text(script)
    targets[3].mkdir()
    command = ['docker', '-H', 'unix:///run/hp-pr-two-commits-20261008/docker.sock',
               'exec', 'jt-adamw-hp-20261013', 'bash', '/alignment/phase12/' + targets[1].name]
    with (ROOT / 'runs' / (run + '.dispatch.log')).open('wb') as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, start_new_session=True)
    record = {'run': run, 'pid': process.pid, 'card': card, 'port': port,
              'started_at': datetime.now().astimezone().isoformat(), 'command': command,
              'retry_of': base, 'reason': 'Previous observer failed with NPU OOM at202; unchanged arithmetic and observer on idle card4',
              'readonly_observation': True, 'state_steps': [292, 293], 'gradient_step': 293,
              'expected_retained_array_bytes': 2516582400,
              'config_sha256': hashlib.sha256(targets[2].read_bytes()).hexdigest(),
              'launcher_sha256': hashlib.sha256(targets[1].read_bytes()).hexdigest(),
              'entry_sha256': hashlib.sha256((ROOT / 'tools-f637' / entry).read_bytes()).hexdigest()}
    (ROOT / 'evidence' / (run + '-launch.json')).write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
