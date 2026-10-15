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

"""Launch paired read-only preclip hashes with bounded temporary storage."""
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
    """Preserve both native trajectories while observing only full hashes at225/226."""
    status = subprocess.run(['npu-smi', 'info'], check=True, capture_output=True, text=True).stdout
    if any('No running processes found in NPU ' + str(card) not in status for card in [4, 5]):
        raise ValueError('Requested device is no longer idle')
    if shutil.disk_usage(STATE).free < 25_000_000_000:
        raise ValueError('MF temporary full-gradient dumps need at least25GB headroom')
    jobs = []
    for family, card, port in [('hp', 4, 31615), ('mf', 5, 31616)]:
        base = 'hp-jtf637-qsqrt222-both226' if family == 'hp' else 'mf-preclip217'
        run = 'hp-jtf637-qsqrt222-preclip226' if family == 'hp' else 'mf-preclip226'
        if any(path.exists() for path in [ROOT / 'runs' / run, ROOT / ('run-' + run + '.sh'),
                                          ROOT / 'runs' / (run + '.exit'), STATE / ('work-' + run)]):
            raise ValueError('Existing run destination')
        config = (ROOT / 'configs' / (base + '.yaml')).read_text().replace(base, run)
        if family == 'hp':
            config = config.replace('train_iters: 226', 'train_iters: 226')
        else:
            config = config.replace('  stop_step: 17', '  stop_step: 26')
        config_path = ROOT / 'configs' / (run + '.yaml')
        config_path.write_text(config)
        script = (ROOT / ('run-' + base + '.sh')).read_text().replace(base, run)
        script = re.sub(r'ASCEND_RT_VISIBLE_DEVICES=\d+', 'ASCEND_RT_VISIBLE_DEVICES=' + str(card), script)
        script = re.sub(r'--master_port=\d+', '--master_port=' + str(port), script)
        old_entry = 'capture_hp_qsqrt222.py' if family == 'hp' else 'capture_mf_preclip217.py'
        entry = 'capture_' + family + '_preclip226.py'
        script = script.replace(old_entry, entry)
        if family == 'hp':
            script = script.replace('export P12_NORM219_MODEL_HASH=1', 'export P12_NORM219_MODEL_HASH=0')
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
                  'readonly_observation': True, 'raw_gradients_retained': 'MF226 only; HP hashes only'}
        (ROOT / 'evidence' / (run + '-launch.json')).write_text(json.dumps(record, indent=2))
        jobs.append(record)
    print(json.dumps(jobs, indent=2))


if __name__ == '__main__':
    main()
