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

"""Launch isolated independent packed semantics windows on verified idle devices."""
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
    """Hash a small reproducibility source."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Allocate disjoint cards and ports; never replace an existing experiment."""
    status = subprocess.run(['npu-smi', 'info'], check=True, capture_output=True, text=True).stdout
    if any('No running processes found in NPU ' + str(card) not in status for card in range(4)):
        raise ValueError('Requested device is no longer idle')
    if shutil.disk_usage(STATE).free < 12_000_000_000:
        raise ValueError('Insufficient state headroom')
    (ROOT / 'evidence/packed217-full-resource-check.txt').write_text(status)
    jobs = []
    cases = [('absolute-positions', 400), ('global-mtp', 400), ('both', 400), ('both', 226)]
    for card, (scope, end) in enumerate(cases):
        base = 'hp-jtf637-packed217-' + scope + '226'
        run = 'hp-jtf637-packed217-' + scope + (str(end) if end == 400 else '-hash226')
        if any(path.exists() for path in [ROOT / 'runs' / run, ROOT / ('run-' + run + '.sh'),
                                          ROOT / 'runs' / (run + '.exit'), STATE / ('work-' + run)]):
            raise ValueError('Experiment destination already exists: ' + run)
        config = (ROOT / 'configs' / (base + '.yaml')).read_text().replace(base, run)
        config = config.replace('train_iters: 226', 'train_iters: ' + str(end))
        config_path = ROOT / 'configs' / (run + '.yaml')
        config_path.write_text(config)
        script = (ROOT / ('run-' + base + '.sh')).read_text().replace(base, run)
        script = re.sub(r'ASCEND_RT_VISIBLE_DEVICES=\d+', 'ASCEND_RT_VISIBLE_DEVICES=' + str(card), script)
        script = re.sub(r'--master_port=\d+', '--master_port=' + str(31589 + card), script)
        entry = 'capture_hp_packed217.py'
        if end == 226:
            entry = 'capture_hp_packed217_hashes.py'
            script = script.replace('capture_hp_packed217.py', entry)
        launcher = ROOT / ('run-' + run + '.sh')
        launcher.write_text(script)
        (STATE / ('work-' + run)).mkdir()
        command = ['docker', '-H', 'unix:///run/hp-pr-two-commits-20261008/docker.sock',
                   'exec', 'jt-adamw-hp-20261013', 'bash', '/alignment/phase12/' + launcher.name]
        with (ROOT / 'runs' / (run + '.dispatch.log')).open('wb') as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, start_new_session=True)
        record = {'run': run, 'pid': process.pid, 'card': card, 'port': 31589 + card,
                  'started_at': datetime.now().astimezone().isoformat(), 'command': command,
                  'config_sha256': digest(config_path), 'launcher_sha256': digest(launcher),
                  'entry_sha256': digest(ROOT / 'tools-f637' / entry), 'scope': scope,
                  'teacher_values_used': False, 'archive_compatibility_semantics': True,
                  'readonly_hash_observer': end == 226}
        (ROOT / 'evidence' / (run + '-launch.json')).write_text(json.dumps(record, indent=2))
        jobs.append(record)
    print(json.dumps(jobs, indent=2))


if __name__ == '__main__':
    main()
