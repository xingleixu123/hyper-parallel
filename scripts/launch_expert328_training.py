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

"""Validate the single expert normalization factor in independent200 training and native330 observations."""
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
    """Hash a source or metadata file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Require the native optimizer boundary and idle single devices before dispatch."""
    audit = json.loads((ROOT / 'evidence/frontier328-native-cause-audit.json').read_text())
    if not audit['complete'] or not all(all(row['fidelity'].values()) for row in audit['records']):
        raise ValueError('Actual-update and holdout replay fidelity failed')
    status = subprocess.run(['npu-smi', 'info'], check=True, capture_output=True, text=True).stdout
    cards = [card for card in [4, 5, 6, 7, 2, 3, 0, 1]
             if 'No running processes found in NPU ' + str(card) in status][:2]
    if len(cards) != 2:
        raise ValueError('Two isolated idle devices are unavailable')
    ports = subprocess.run(['ss', '-ltn'], check=True, capture_output=True, text=True).stdout
    if any(':' + str(port) + ' ' in ports for port in [31699, 31700]):
        raise ValueError('Rendezvous port is occupied')
    free = {'state': shutil.disk_usage(STATE).free, 'metadata': shutil.disk_usage(ROOT).free}
    if free['state'] < 7_000_000_000 or free['metadata'] < 2_000_000_000:
        raise ValueError('Insufficient headroom for selected state and temporary outputs')
    memory = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    available = int(memory['MemAvailable'].split()[0]) * 1024
    if available < 80_000_000_000:
        raise ValueError('Insufficient host memory for paired jobs and model hashes')
    jobs = [
        ('hp', 'hp-jte213-fc1norm-qupboth400', 'hp-jte213-expertnorm328400',
         cards[0], 31699, 'capture_hp_fc1_norm293.py', 'capture_hp_expert_norm328.py'),
        ('hp', 'hp-jte213-fc1norm-qupboth-preclip330', 'hp-jte213-expertnorm328-preclip330',
         cards[1], 31700, 'capture_hp_fc1_preclip330.py', 'capture_hp_expert_norm328.py'),
    ]
    records = []
    for family, base, run, card, port, old_entry, entry in jobs:
        config_path = ROOT / 'configs' / (run + '.yaml')
        launcher = ROOT / ('run-' + run + '.sh')
        work = STATE / ('work-' + run)
        if any(path.exists() for path in [config_path, launcher, work, ROOT / 'runs' / run, STATE / run]):
            raise ValueError('Existing run destination: ' + run)
        config = (ROOT / 'configs' / (base + '.yaml')).read_text().replace(base, run)
        config_path.write_text(config)
        script = (ROOT / ('run-' + base + '.sh')).read_text().replace(base, run).replace(old_entry, entry)
        script = re.sub(r'ASCEND_RT_VISIBLE_DEVICES=\d+', 'ASCEND_RT_VISIBLE_DEVICES=' + str(card), script)
        script = re.sub(r'--master_port=\d+', '--master_port=' + str(port), script)
        mode = 'observe330' if run.endswith('preclip330') else 'train'
        script = script.replace('source /workspace/env.sh\n', 'source /workspace/env.sh\n'
                                + 'export P12_EXPERT328_ENTRY=' + mode + '\n')
        launcher.write_text(script)
        work.mkdir()
        socket = 'hp-pr-two-commits-20261008' if family == 'hp' else 'mf-image-verify-20261008'
        command = ['docker', '-H', 'unix:///run/' + socket + '/docker.sock', 'exec',
                   'jt-adamw-' + family + '-20261013', 'bash', '/alignment/phase12/' + launcher.name]
        with (ROOT / 'runs' / (run + '.dispatch.log')).open('wb') as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, start_new_session=True)
        folder = 'tools-e213' if family == 'hp' else 'tools'
        row = {'run': run, 'base': base, 'pid': process.pid, 'card': card, 'port': port,
               'started_at': datetime.now().astimezone().isoformat(), 'command': command,
               'readonly_observation': mode == 'observe330', 'expected_steps': 130 if mode == 'observe330' else 200,
               'only_added_factor': 'hp_expert_norm328_factor.py',
               'teacher_values_used': False, 'target': 'expert-down normalization shape(8,5120,1536)',
               'factor_sha256': digest(ROOT / 'tools-f637/hp_expert_norm328_factor.py'),
               'config_sha256': digest(config_path), 'launcher_sha256': digest(launcher),
               'entry_sha256': digest(ROOT / folder / entry),
               'host_available_bytes': available, 'free_bytes_at_launch': free}
        (ROOT / 'evidence' / (run + '-launch.json')).write_text(json.dumps(row, indent=2))
        records.append(row)
    print(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
