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

"""Replay one actual expert-down update in each native framework."""
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
    """Start only after both captured training trajectories pass fidelity checks."""
    audit = json.loads((ROOT / 'evidence/frontier328-state-comparison.json').read_text())
    if not audit['complete'] or not all(all(row.values()) for row in audit['fidelity'].values()):
        raise ValueError('Actual-state observation fidelity failed')
    status = subprocess.run(['npu-smi', 'info'], check=True, capture_output=True, text=True).stdout
    cards = [card for card in [0, 1, 2, 3, 6, 7]
             if 'No running processes found in NPU ' + str(card) in status]
    if not cards:
        raise ValueError('No isolated replay device is idle')
    ports = subprocess.run(['ss', '-ltn'], check=True, capture_output=True, text=True).stdout
    if ':31696 ' in ports:
        raise ValueError('Rendezvous port is occupied')
    if min(shutil.disk_usage(ROOT).free, shutil.disk_usage(STATE).free) < 2_000_000_000:
        raise ValueError('Insufficient metadata headroom')
    jobs = [('hp', cards[0], 'frontier328-hp-control-replay', 'frontier328-hp-normalization-screen',
             'replay_hp_frontier328_normalization_screen.py')]
    for family, card, base, run, source in jobs:
        launcher = ROOT / ('run-' + run + '.sh')
        work = STATE / ('work-' + run)
        if any(path.exists() for path in [launcher, work, ROOT / 'runs' / (run + '.log'), STATE / run]):
            raise ValueError('Existing replay destination')
        script = (ROOT / ('run-' + base + '.sh')).read_text().replace(base, run)
        script = script.replace('replay_hp_frontier328_muon.py', source)
        script = re.sub(r'ASCEND_RT_VISIBLE_DEVICES=\d+', 'ASCEND_RT_VISIBLE_DEVICES=' + str(card), script)
        script = re.sub(r'--master_port=\d+', '--master_port=31696', script)
        script = script.replace('export P12_FRONTIER306_REPLAY_SCOPE=control\n', '')
        config_path = ROOT / 'configs' / (run + '.json')
        if family == 'mf':
            config = json.loads((ROOT / 'configs' / (base + '.json')).read_text())
            config['common_dump_settings'].update(
                path='/state/jt-recipe-20261014/' + run, net_name='native_muon328',
                dump_mode=0, kernels=[], saved_data='statistic')
            config_path.write_text(json.dumps(config, indent=2))
        launcher.write_text(script)
        work.mkdir()
        socket = 'hp-pr-two-commits-20261008' if family == 'hp' else 'mf-image-verify-20261008'
        command = ['docker', '-H', 'unix:///run/' + socket + '/docker.sock', 'exec',
                   'jt-adamw-' + family + '-20261013', 'bash', '/alignment/phase12/' + launcher.name]
        with (ROOT / 'runs' / (run + '.dispatch.log')).open('wb') as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, start_new_session=True)
        folder = 'tools-f637' if family == 'hp' else 'tools'
        row = {'run': run, 'pid': process.pid, 'card': card, 'command': command,
               'started_at': datetime.now().astimezone().isoformat(),
               'launcher_sha256': digest(launcher), 'source_sha256': digest(ROOT / folder / source),
               'scope': 'Four live-value normalization choices; require original-control and all momentum fidelity; no teacher values.'}
        if family == 'mf':
            row['config_sha256'] = digest(config_path)
        (ROOT / 'evidence' / (run + '-launch.json')).write_text(json.dumps(row, indent=2))
        print(json.dumps(row))


if __name__ == '__main__':
    main()
