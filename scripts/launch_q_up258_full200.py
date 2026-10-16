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

"""Test Q-up normalization in an independent 200-step continuation."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path('/home/xxl/jt-adamw-align-20261013/phase12')
STATE = Path('/dev/shm/jt-adamw-align-20261013/state/jt-recipe-20261014')
BASE = 'hp-jtf637-weighted-mtp400'
RUN = 'hp-jtf637-qup-nope400'
CARD = 6
PORT = 31663


def digest(path: Path) -> str:
    """Hash source and configuration for the immutable launch receipt."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Launch one isolated single-factor candidate after checking capacity and ownership."""
    status = subprocess.run(['npu-smi', 'info'], check=True, capture_output=True, text=True).stdout
    if 'No running processes found in NPU ' + str(CARD) not in status:
        raise ValueError('Requested card is occupied')
    ports = subprocess.run(['ss', '-ltn'], check=True, capture_output=True, text=True).stdout
    if ':' + str(PORT) + ' ' in ports:
        raise ValueError('Rendezvous port is occupied')
    free = {'state': shutil.disk_usage(STATE).free, 'output': shutil.disk_usage(ROOT).free}
    if free['state'] < 2_000_000_000 or free['output'] < 5_000_000_000:
        raise ValueError('Insufficient headroom for training logs and transient compilation')
    config = ROOT / 'configs' / (RUN + '.yaml')
    launcher = ROOT / ('run-' + RUN + '.sh')
    work = STATE / ('work-' + RUN)
    if any(path.exists() for path in [config, launcher, work, ROOT / 'runs' / RUN]):
        raise ValueError('Run destination exists')
    script = (ROOT / ('run-' + BASE + '.sh')).read_text()
    if script.count('P12_WEIGHTED_DVM_SCOPE=both') != 1 or 'P12_NORM219_MODEL_HASH=0' not in script:
        raise ValueError('Unexpected control scope or model observation policy')
    config.write_text((ROOT / 'configs' / (BASE + '.yaml')).read_text().replace(BASE, RUN))
    script = script.replace(BASE, RUN).replace('capture_hp_global_sqrt400.py', 'capture_hp_q_up258.py')
    script = script.replace('export ASCEND_RT_VISIBLE_DEVICES',
                            'export P12_QUP258_SCOPE=nope\nexport ASCEND_RT_VISIBLE_DEVICES')
    script = re.sub(r'ASCEND_RT_VISIBLE_DEVICES=\d+', 'ASCEND_RT_VISIBLE_DEVICES=' + str(CARD), script)
    script = re.sub(r'--master_port=\d+', '--master_port=' + str(PORT), script)
    launcher.write_text(script)
    work.mkdir()
    command = ['docker', '-H', 'unix:///run/hp-pr-two-commits-20261008/docker.sock', 'exec',
               'jt-adamw-hp-20261013', 'bash', '/alignment/phase12/' + launcher.name]
    with (ROOT / 'runs' / (RUN + '.dispatch.log')).open('wb') as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, start_new_session=True)
    record = {'run': RUN, 'control': BASE, 'pid': process.pid, 'card': CARD, 'port': PORT,
              'started_at': datetime.now().astimezone().isoformat(), 'command': command,
              'config_sha256': digest(config), 'launcher_sha256': digest(launcher),
              'entry_sha256': digest(ROOT / 'tools-f637/capture_hp_q_up258.py'),
              'factor_sha256': digest(ROOT / 'tools-f637/hp_q_up258_factor.py'),
              'candidate_change': 'Q-up non-RoPE live square-sum schedule; all other policies unchanged',
              'raw_arrays_written': False, 'expected_steps': 200, 'free_bytes': free,
              'source_sha256': digest(Path(__file__))}
    (ROOT / 'evidence' / (RUN + '-launch.json')).write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
