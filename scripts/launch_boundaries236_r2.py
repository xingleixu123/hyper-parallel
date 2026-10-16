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

"""Repeat targeted boundary statistics without redundant model hash observations."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path('/home/xxl/jt-adamw-align-20261013/phase12')
STATE = Path('/dev/shm/jt-adamw-align-20261013/state/jt-recipe-20261014')
BASE = 'hp-jtf637-global-sqrt-boundaries236'
RUN = BASE + '-r2'


def digest(path: Path) -> str:
    """Hash launch provenance."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Launch only on the released task card, preserving every training factor."""
    status = subprocess.run(['npu-smi', 'info'], check=True, capture_output=True, text=True).stdout
    if 'No running processes found in NPU 4' not in status:
        raise ValueError('Task card still occupied')
    ports = subprocess.run(['ss', '-ltn'], check=True, capture_output=True, text=True).stdout
    if ':31652 ' in ports:
        raise ValueError('Rendezvous port occupied')
    if shutil.disk_usage(STATE).free < 2_000_000_000 or shutil.disk_usage(ROOT).free < 5_000_000_000:
        raise ValueError('Insufficient headroom')
    paths = [ROOT / 'runs' / RUN, ROOT / ('run-' + RUN + '.sh'), STATE / ('work-' + RUN)]
    if any(path.exists() for path in paths):
        raise ValueError('Repeat run already exists')
    if not (ROOT / 'runs' / (BASE + '.exit')).exists():
        raise ValueError('Previous task run is not terminal')
    config = ROOT / 'configs' / (RUN + '.yaml')
    config.write_text((ROOT / 'configs' / (BASE + '.yaml')).read_text().replace(BASE, RUN))
    launcher = ROOT / ('run-' + RUN + '.sh')
    script = (ROOT / ('run-' + BASE + '.sh')).read_text().replace(BASE, RUN).replace('31650', '31652')
    if script.count('P12_NORM219_MODEL_HASH=1') != 1:
        raise ValueError('Unexpected model observation switch')
    launcher.write_text(script.replace('P12_NORM219_MODEL_HASH=1', 'P12_NORM219_MODEL_HASH=0'))
    (STATE / ('work-' + RUN)).mkdir()
    command = ['docker', '-H', 'unix:///run/hp-pr-two-commits-20261008/docker.sock', 'exec',
               'jt-adamw-hp-20261013', 'bash', '/alignment/phase12/' + launcher.name]
    with (ROOT / 'runs' / (RUN + '.dispatch.log')).open('wb') as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, start_new_session=True)
    record = {'run': RUN, 'pid': process.pid, 'card': 4, 'port': 31652,
              'started_at': datetime.now().astimezone().isoformat(), 'command': command,
              'config_sha256': digest(config), 'launcher_sha256': digest(launcher),
              'entry_sha256': digest(ROOT / 'tools-f637/capture_hp_boundaries236.py'),
              'raw_arrays_written': False, 'readonly_observation': True,
              'disabled_only': 'redundant full-model hash observer',
              'expected_training_steps': [236],
              'free_bytes': {'state': shutil.disk_usage(STATE).free, 'output': shutil.disk_usage(ROOT).free}}
    (ROOT / 'evidence' / (RUN + '-launch.json')).write_text(json.dumps(record, indent=2))
    print(json.dumps(record))


if __name__ == '__main__':
    main()
