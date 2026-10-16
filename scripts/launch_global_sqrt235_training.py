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

"""Launch separate short and full independent global-sqrt compatibility experiments."""
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
    """Hash a small source file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Keep model restoration and all other numerical factors unchanged."""
    status = subprocess.run(['npu-smi', 'info'], check=True, capture_output=True, text=True).stdout
    if any('No running processes found in NPU ' + str(card) not in status for card in [4, 6]):
        raise ValueError('Requested device is no longer idle')
    if shutil.disk_usage(STATE).free < 2_000_000_000:
        raise ValueError('Independent observations need at least 2 GB headroom')
    ports = subprocess.run(['ss', '-ltn'], check=True, capture_output=True, text=True).stdout
    if any(':' + str(port) + ' ' in ports for port in [31644, 31645]):
        raise ValueError('Requested rendezvous port is occupied')
    jobs = []
    for end, card, port in [(235, 4, 31644), (400, 6, 31645)]:
        family = 'hp'
        base = 'hp-jtf637-epsilon235' if end == 235 else 'hp-jtf637-epsilon400'
        run = 'hp-jtf637-global-sqrt' + str(end)
        if any(path.exists() for path in [ROOT / 'runs' / run, ROOT / ('run-' + run + '.sh'),
                                          ROOT / 'runs' / (run + '.exit'), STATE / ('work-' + run)]):
            raise ValueError('Existing run destination')
        config = (ROOT / 'configs' / (base + '.yaml')).read_text().replace(base, run)
        config_path = ROOT / 'configs' / (run + '.yaml')
        config_path.write_text(config)
        script = (ROOT / ('run-' + base + '.sh')).read_text().replace(base, run)
        script = re.sub(r'ASCEND_RT_VISIBLE_DEVICES=\d+', 'ASCEND_RT_VISIBLE_DEVICES=' + str(card), script)
        script = re.sub(r'--master_port=\d+', '--master_port=' + str(port), script)
        old_entry = 'capture_hp_epsilon235.py' if end == 235 else 'capture_hp_epsilon229.py'
        entry = 'capture_hp_global_sqrt235.py' if end == 235 else 'capture_hp_global_sqrt400.py'
        script = script.replace(old_entry, entry)
        launcher = ROOT / ('run-' + run + '.sh')
        launcher.write_text(script)
        (STATE / ('work-' + run)).mkdir()
        socket = 'hp-pr-two-commits-20261008' if family == 'hp' else 'mf-image-verify-20261008'
        command = ['docker', '-H', 'unix:///run/' + socket + '/docker.sock', 'exec',
                   'jt-adamw-' + family + '-20261013', 'bash', '/alignment/phase12/' + launcher.name]
        with (ROOT / 'runs' / (run + '.dispatch.log')).open('wb') as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, start_new_session=True)
        record = {'run': run, 'pid': process.pid, 'card': card, 'port': port,
                  'started_at': datetime.now().astimezone().isoformat(), 'command': command,
                  'config_sha256': digest(config_path), 'launcher_sha256': digest(launcher),
                  'entry_sha256': digest(ROOT / ('tools-f637' if family == 'hp' else 'tools') / entry),
                  'single_factor': 'Global gradient norm sqrt on NPU instead of CPU',
                  'model_hash_steps': list(range(231, 236)) if end == 235 else [],
                  'raw_model_arrays_written': False, 'teacher_values_used': False}
        (ROOT / 'evidence' / (run + '-launch.json')).write_text(json.dumps(record, indent=2))
        jobs.append(record)
    print(json.dumps(jobs, indent=2))


if __name__ == '__main__':
    main()
