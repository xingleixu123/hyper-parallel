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

"""Capture selected shared-expert state around the verified396 update boundary."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path('/home/xxl/jt-adamw-align-20261013/phase12')
STATE = Path('/dev/shm/jt-adamw-align-20261013/state/jt-recipe-20261014')
TARGET = 'model.layers.1.mlp.shared_experts.down_proj.weight'


def digest(path: Path) -> str:
    """Hash a bounded source or metadata file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Require unchanged trajectories, the exact boundary, and two isolated idle cards."""
    audit = json.loads((ROOT / 'evidence/frontier397-native-statistics-comparison.json').read_text())
    if not audit['complete'] or not all(all(row.values()) for row in audit['fidelity'].values()):
        raise ValueError('Read-only native statistics fidelity is not established')
    models = {row['step']: row['different'] for row in audit['models']}
    if any(models[395].values()) or any(names != [TARGET] for names in models[396].values()):
        raise ValueError('Unexpected395/396 full model boundary')
    gradient = next(row for row in audit['comparisons'] if row['step'] == 396)
    if not (gradient['gradient_exact_count'] == gradient['square_sum_exact_count'] == 46
            and gradient['global_norm_exact'] and gradient['total_square_sum_exact']):
        raise ValueError('Complete396 preclip equality is not established')
    status = subprocess.run(['npu-smi', 'info'], check=True, capture_output=True, text=True).stdout
    cards = [card for card in [4, 5, 6, 7, 2, 3, 0, 1]
             if 'No running processes found in NPU ' + str(card) in status][:2]
    if len(cards) != 2:
        raise ValueError('Two isolated idle devices are unavailable')
    ports = subprocess.run(['ss', '-ltn'], check=True, capture_output=True, text=True).stdout
    if any(':' + str(port) + ' ' in ports for port in [31713, 31714]):
        raise ValueError('Rendezvous port is occupied')
    free = {'state': shutil.disk_usage(STATE).free, 'metadata': shutil.disk_usage(ROOT).free}
    if free['state'] < 4_000_000_000 or free['metadata'] < 2_000_000_000:
        raise ValueError('Insufficient headroom for selected arrays and temporary outputs')
    memory = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    available = int(memory['MemAvailable'].split()[0]) * 1024
    if available < 80_000_000_000:
        raise ValueError('Insufficient host memory for paired jobs and full model hashes')
    jobs = [
        ('hp', 'hp-jte213-expertnorm328400', 'hp-jte213-expertnorm328-state396',
         cards[0], 31713, 'capture_hp_expert_norm328.py', 'capture_hp_frontier396_state.py'),
        ('mf', 'mf-frontier328-state', 'mf-frontier396-state',
         cards[1], 31714, 'capture_mf_frontier328_state.py', 'capture_mf_frontier396_state.py'),
    ]
    for _, _, run, _, _, _, _ in jobs:
        paths = [ROOT / 'configs' / (run + '.yaml'), ROOT / ('run-' + run + '.sh'),
                 STATE / ('work-' + run), ROOT / 'runs' / run, STATE / run]
        if any(path.exists() for path in paths):
            raise ValueError('Existing run destination: ' + run)
    records = []
    for family, base, run, card, port, old_entry, entry in jobs:
        config_path = ROOT / 'configs' / (run + '.yaml')
        launcher = ROOT / ('run-' + run + '.sh')
        work = STATE / ('work-' + run)
        config = (ROOT / 'configs' / (base + '.yaml')).read_text().replace(base, run)
        before, after = ('  train_iters: 400', '  train_iters: 396') if family == 'hp' else (
            '  stop_step: 128', '  stop_step: 196')
        if config.count(before) != 1:
            raise ValueError('Unexpected training window configuration')
        config_path.write_text(config.replace(before, after))
        script = (ROOT / ('run-' + base + '.sh')).read_text().replace(base, run).replace(old_entry, entry)
        script = re.sub(r'ASCEND_RT_VISIBLE_DEVICES=\d+', 'ASCEND_RT_VISIBLE_DEVICES=' + str(card), script)
        script = re.sub(r'--master_port=\d+', '--master_port=' + str(port), script)
        if family == 'hp':
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
        row = {'run': run, 'base': base, 'pid': process.pid, 'card': card, 'port': port,
               'started_at': datetime.now().astimezone().isoformat(), 'command': command,
               'readonly_observation': True, 'expected_state_steps': [395, 396],
               'expected_gradient_step': 396, 'expected_steps': 196,
               'teacher_values_used': False, 'target': TARGET,
               'estimated_retained_array_bytes': 692_060_160,
               'native_earlier_gradient_arrays': 'hashed and removed each completed step',
               'config_sha256': digest(config_path), 'launcher_sha256': digest(launcher),
               'entry_sha256': digest(ROOT / folder / entry),
               'host_available_bytes': available, 'free_bytes_at_launch': free}
        (ROOT / 'evidence' / (run + '-launch.json')).write_text(json.dumps(row, indent=2))
        records.append(row)
    print(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
