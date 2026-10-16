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

"""Run MTP routing hashes and an independent scope ablation on disjoint idle cards."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path('/home/xxl/jt-adamw-align-20261013/phase12')
STATE = Path('/dev/shm/jt-adamw-align-20261013/state/jt-recipe-20261014')
BASE = 'hp-jtf637-global-sqrt-preclip236'


def digest(path: Path) -> str:
    """Hash a source or launch configuration."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Keep training configurations comparable and isolate candidate scope from observation."""
    status = subprocess.run(['npu-smi', 'info'], check=True, capture_output=True, text=True).stdout
    if any('No running processes found in NPU ' + str(card) not in status for card in [4, 6]):
        raise ValueError('Requested card is occupied')
    ports = subprocess.run(['ss', '-ltn'], check=True, capture_output=True, text=True).stdout
    if any(':' + str(port) + ' ' in ports for port in [31653, 31654]):
        raise ValueError('Rendezvous port is occupied')
    if shutil.disk_usage(STATE).free < 2_000_000_000 or shutil.disk_usage(ROOT).free < 5_000_000_000:
        raise ValueError('Statistics need filesystem headroom')
    jobs = []
    for run, card, port, is_observation in [('hp-jtf637-global-sqrt-moe236', 4, 31653, True),
                                           ('hp-jtf637-weighted-mtp236', 6, 31654, False)]:
        if any(path.exists() for path in [ROOT / 'runs' / run, ROOT / ('run-' + run + '.sh'),
                                         STATE / ('work-' + run)]):
            raise ValueError('Run destination exists')
        config = ROOT / 'configs' / (run + '.yaml')
        config.write_text((ROOT / 'configs' / (BASE + '.yaml')).read_text().replace(BASE, run))
        script = (ROOT / ('run-' + BASE + '.sh')).read_text().replace(BASE, run)
        script = re.sub(r'ASCEND_RT_VISIBLE_DEVICES=\d+', 'ASCEND_RT_VISIBLE_DEVICES=' + str(card), script)
        script = re.sub(r'--master_port=\d+', '--master_port=' + str(port), script)
        entry = 'capture_hp_moe236.py' if is_observation else 'capture_hp_preclip236.py'
        if is_observation:
            script = script.replace('capture_hp_preclip236.py', entry)
            script = script.replace('P12_NORM219_MODEL_HASH=1', 'P12_NORM219_MODEL_HASH=0')
        else:
            if script.count('P12_WEIGHTED_DVM_SCOPE=main') != 1:
                raise ValueError('Unexpected existing weighted-gradient scope')
            script = script.replace('P12_WEIGHTED_DVM_SCOPE=main', 'P12_WEIGHTED_DVM_SCOPE=both')
        launcher = ROOT / ('run-' + run + '.sh')
        launcher.write_text(script)
        (STATE / ('work-' + run)).mkdir()
        command = ['docker', '-H', 'unix:///run/hp-pr-two-commits-20261008/docker.sock', 'exec',
                   'jt-adamw-hp-20261013', 'bash', '/alignment/phase12/' + launcher.name]
        with (ROOT / 'runs' / (run + '.dispatch.log')).open('wb') as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, start_new_session=True)
        record = {'run': run, 'pid': process.pid, 'card': card, 'port': port,
                  'started_at': datetime.now().astimezone().isoformat(), 'command': command,
                  'config_sha256': digest(config), 'launcher_sha256': digest(launcher),
                  'entry_sha256': digest(ROOT / 'tools-f637' / entry),
                  'readonly_observation': is_observation, 'raw_arrays_written': False,
                  'candidate_change': None if is_observation else 'weighted reduction scope main to both',
                  'expected_steps': 36, 'free_bytes': {'state': shutil.disk_usage(STATE).free,
                                                    'output': shutil.disk_usage(ROOT).free}}
        (ROOT / 'evidence' / (run + '-launch.json')).write_text(json.dumps(record, indent=2))
        jobs.append(record)
    print(json.dumps(jobs, indent=2))


if __name__ == '__main__':
    main()
