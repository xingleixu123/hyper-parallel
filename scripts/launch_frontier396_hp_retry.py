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

"""Retry only the failed HP observation with correct wrapper installation order."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path('/home/xxl/jt-adamw-align-20261013/phase12')
STATE = Path('/dev/shm/jt-adamw-align-20261013/state/jt-recipe-20261014')


def digest(path: Path) -> str:
    """Hash a bounded source or metadata file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Preserve the failed attempt and retry on one verified idle device."""
    prior = 'hp-jte213-expertnorm328-state396'
    run = prior + '-r2'
    if (ROOT / 'runs' / (prior + '.exit')).read_text().strip() != '1':
        raise ValueError('The original observation is not a confirmed failed job')
    if (ROOT / 'runs' / prior / 'metrics.jsonl').exists():
        raise ValueError('Unexpected numerical output from failed observer')
    status = subprocess.run(['npu-smi', 'info'], capture_output=True, text=True, check=True).stdout
    cards = [card for card in [4, 6, 7, 2, 3, 0, 1]
             if 'No running processes found in NPU ' + str(card) in status]
    if not cards:
        raise ValueError('No isolated idle device')
    ports = subprocess.run(['ss', '-ltn'], capture_output=True, text=True, check=True).stdout
    if ':31715 ' in ports:
        raise ValueError('Retry port is occupied')
    free = {'state': shutil.disk_usage(STATE).free, 'metadata': shutil.disk_usage(ROOT).free}
    if min(free.values()) < 2_000_000_000:
        raise ValueError('Insufficient observation headroom')
    config = ROOT / 'configs' / (run + '.yaml')
    launcher = ROOT / ('run-' + run + '.sh')
    work = STATE / ('work-' + run)
    if any(path.exists() for path in [config, launcher, work, ROOT / 'runs' / run, STATE / run]):
        raise ValueError('Retry destination already exists')
    config.write_text((ROOT / 'configs' / (prior + '.yaml')).read_text().replace(prior, run))
    source = (ROOT / ('run-' + prior + '.sh')).read_text().replace(prior, run)
    source = source.replace('capture_hp_frontier396_state.py', 'capture_hp_frontier396_state_r2.py')
    source = re.sub(r'ASCEND_RT_VISIBLE_DEVICES=\d+', 'ASCEND_RT_VISIBLE_DEVICES=' + str(cards[0]), source)
    source = re.sub(r'--master_port=\d+', '--master_port=31715', source)
    launcher.write_text(source)
    work.mkdir()
    command = ['docker', '-H', 'unix:///run/hp-pr-two-commits-20261008/docker.sock', 'exec',
               'jt-adamw-hp-20261013', 'bash', '/alignment/phase12/' + launcher.name]
    with (ROOT / 'runs' / (run + '.dispatch.log')).open('wb') as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, start_new_session=True)
    report = {'run': run, 'base': 'hp-jte213-expertnorm328400', 'prior_failed_attempt': prior,
              'pid': process.pid, 'card': cards[0], 'port': 31715,
              'started_at': datetime.now().astimezone().isoformat(), 'command': command,
              'expected_steps': 196, 'readonly_observation': True,
              'expected_state_steps': [395, 396], 'expected_gradient_step': 396,
              'teacher_values_used': False, 'free_bytes_at_launch': free,
              'config_sha256': digest(config), 'launcher_sha256': digest(launcher),
              'entry_sha256': digest(ROOT / 'tools-e213/capture_hp_frontier396_state_r2.py'),
              'launch_source_sha256': digest(Path(__file__)),
              'reason': 'Build original clipping factor before installing the read-only observer wrapper.'}
    (ROOT / 'evidence' / (run + '-launch.json')).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
