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

"""Run separate-card dense-fc1 holdouts with statistics-only native Dump."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path('/home/xxl/jt-adamw-align-20261013/phase12')
STATE = Path('/dev/shm/jt-adamw-align-20261013/state/jt-recipe-20261014')


def main() -> None:
    """Do not retain raw synthetic arrays on either framework."""
    status = subprocess.run(['npu-smi', 'info'], check=True, capture_output=True, text=True).stdout
    if any('No running processes found in NPU ' + str(card) not in status for card in [6, 5]):
        raise ValueError('Requested device is busy')
    if min(shutil.disk_usage(ROOT).free, shutil.disk_usage(STATE).free) < 1_000_000_000:
        raise ValueError('Insufficient metadata headroom')
    records = []
    for family, card in [('hp', 6), ('mf', 5)]:
        suffix = '-hash' if family == 'mf' else ''
        base = 'frontier258-' + family + '-norm-holdout' + suffix
        run = base.replace('258', '293')
        work = STATE / ('work-' + run)
        launcher = ROOT / ('run-' + run + '.sh')
        if work.exists() or launcher.exists():
            raise ValueError('Existing holdout destination')
        if family == 'mf':
            config = json.loads((ROOT / 'configs' / (base + '.json')).read_text())
            config['common_dump_settings']['path'] = '/state/jt-recipe-20261014/' + run
            config['common_dump_settings']['net_name'] = 'fc1_holdout'
            (ROOT / 'configs' / (run + '.json')).write_text(json.dumps(config, indent=2))
        script = (ROOT / ('run-' + base + '.sh')).read_text().replace('258', '293')
        script = re.sub(r'ASCEND_RT_VISIBLE_DEVICES=\d+', 'ASCEND_RT_VISIBLE_DEVICES=' + str(card), script)
        script = script.replace('jt-f637-pr192-venv', 'jt-e213-pr192-venv')
        launcher.write_text(script)
        work.mkdir()
        socket = 'hp-pr-two-commits-20261008' if family == 'hp' else 'mf-image-verify-20261008'
        command = ['docker', '-H', 'unix:///run/' + socket + '/docker.sock', 'exec',
                   'jt-adamw-' + family + '-20261013', 'bash', '/alignment/phase12/' + launcher.name]
        with (ROOT / 'runs' / (run + '.dispatch.log')).open('wb') as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, start_new_session=True)
        row = {'run': run, 'pid': process.pid, 'card': card,
               'started_at': datetime.now().astimezone().isoformat(), 'command': command,
               'launcher_sha256': hashlib.sha256(launcher.read_bytes()).hexdigest(),
               'raw_arrays_written': False, 'cases': 3, 'elements_per_case': 62914560}
        (ROOT / 'evidence' / (run + '-launch.json')).write_text(json.dumps(row, indent=2))
        records.append(row)
    print(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
