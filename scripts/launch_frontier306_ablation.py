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

"""Observe the306 optimizer boundary and independently test the remaining Q-up partition."""
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
    """Hash the executed source or configuration."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Use four disjoint single devices, preserving data and all prior factors."""
    audit = json.loads((ROOT / 'evidence/frontier307-native-statistics-comparison.json').read_text())
    if not audit['complete'] or not all(all(row.values()) for row in audit['fidelity'].values()):
        raise ValueError('Boundary fidelity is not established')
    first = audit['comparisons'][0]
    if first['step'] != 306 or first['gradient_exact_count'] != 46 or not first['global_norm_exact']:
        raise ValueError('Expected an optimizer-only boundary')
    status = subprocess.run(['npu-smi', 'info'], check=True, capture_output=True, text=True).stdout
    if any('No running processes found in NPU ' + str(card) not in status for card in [4, 5, 6, 7]):
        raise ValueError('Requested device is busy')
    ports = subprocess.run(['ss', '-ltn'], check=True, capture_output=True, text=True).stdout
    if any(':' + str(port) + ' ' in ports for port in range(31680, 31684)):
        raise ValueError('Rendezvous port is occupied')
    if shutil.disk_usage(STATE).free < 5_000_000_000 or shutil.disk_usage(ROOT).free < 3_000_000_000:
        raise ValueError('Insufficient headroom for selected state and metadata')
    memory = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    if int(memory['MemAvailable'].split()[0]) * 1024 < 100_000_000_000:
        raise ValueError('Insufficient host memory for four isolated single-card jobs')
    jobs = [
        ('hp', 'hp-jte213-fc1norm400', 'hp-jte213-fc1norm-qupboth400',
         7, 31680, 400, 'capture_hp_fc1_norm293.py', 'capture_hp_fc1_norm293.py', 'both'),
        ('hp', 'hp-jte213-fc1norm-preclip307', 'hp-jte213-fc1norm-qupboth-preclip307',
         6, 31681, 307, 'capture_hp_fc1_preclip307.py', 'capture_hp_fc1_preclip307.py', 'both'),
        ('hp', 'hp-jte213-fc1norm400', 'hp-jte213-fc1norm-state306',
         4, 31682, 306, 'capture_hp_fc1_norm293.py', 'capture_hp_frontier306_state.py', 'nope'),
        ('mf', 'mf-frontier258-state', 'mf-frontier306-state',
         5, 31683, 306, 'capture_mf_frontier258_state.py', 'capture_mf_frontier306_state.py', None),
    ]
    records = []
    for family, base, run, card, port, steps, old_entry, entry, scope in jobs:
        config_path = ROOT / 'configs' / (run + '.yaml')
        launcher = ROOT / ('run-' + run + '.sh')
        work = STATE / ('work-' + run)
        if any(path.exists() for path in [config_path, launcher, work, ROOT / 'runs' / run, STATE / run]):
            raise ValueError('Existing run destination: ' + run)
        config = (ROOT / 'configs' / (base + '.yaml')).read_text().replace(base, run)
        if family == 'hp':
            config = config.replace('  train_iters: 400', '  train_iters: ' + str(steps))
        else:
            config = config.replace('  stop_step: 58', '  stop_step: 106')
        config_path.write_text(config)
        script = (ROOT / ('run-' + base + '.sh')).read_text().replace(base, run).replace(old_entry, entry)
        script = re.sub(r'ASCEND_RT_VISIBLE_DEVICES=\d+', 'ASCEND_RT_VISIBLE_DEVICES=' + str(card), script)
        script = re.sub(r'--master_port=\d+', '--master_port=' + str(port), script)
        if scope == 'both':
            script = script.replace('P12_QUP258_SCOPE=nope', 'P12_QUP258_SCOPE=both')
        if family == 'hp' and steps == 306:
            script = script.replace('P12_NORM219_MODEL_HASH=0', 'P12_NORM219_MODEL_HASH=1')
        launcher.write_text(script)
        work.mkdir()
        socket = 'hp-pr-two-commits-20261008' if family == 'hp' else 'mf-image-verify-20261008'
        command = ['docker', '-H', 'unix:///run/' + socket + '/docker.sock', 'exec',
                   'jt-adamw-' + family + '-20261013', 'bash', '/alignment/phase12/' + launcher.name]
        with (ROOT / 'runs' / (run + '.dispatch.log')).open('wb') as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, start_new_session=True)
        folder = 'tools-e213' if family == 'hp' else 'tools'
        row = {'run': run, 'pid': process.pid, 'card': card, 'port': port,
               'started_at': datetime.now().astimezone().isoformat(), 'command': command,
               'base': base, 'train_iters': steps, 'q_up_scope': scope,
               'factor_sha256': digest(ROOT / 'tools-f637/hp_q_up258_factor.py'),
               'teacher_values_used': False, 'readonly_state_observation': steps == 306,
               'config_sha256': digest(config_path), 'launcher_sha256': digest(launcher),
               'entry_sha256': digest(ROOT / folder / entry),
               'host_available_bytes': int(memory['MemAvailable'].split()[0]) * 1024}
        (ROOT / 'evidence' / (run + '-launch.json')).write_text(json.dumps(row, indent=2))
        records.append(row)
    print(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
