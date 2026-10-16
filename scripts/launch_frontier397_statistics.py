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

"""Observe396/397 using only native gradient statistics and full model hashes."""
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
    """Reuse unchanged archived MF and latest-JT candidate configurations."""
    status = subprocess.run(['npu-smi', 'info'], check=True, capture_output=True, text=True).stdout
    if any('No running processes found in NPU ' + str(card) not in status for card in [4, 5]):
        raise ValueError('Requested device is busy')
    ports = subprocess.run(['ss', '-ltn'], check=True, capture_output=True, text=True).stdout
    if any(':' + str(port) + ' ' in ports for port in [31711, 31712]):
        raise ValueError('Rendezvous port is occupied')
    if min(shutil.disk_usage(ROOT).free, shutil.disk_usage(STATE).free) < 2_000_000_000:
        raise ValueError('Insufficient metadata headroom')
    records = []
    for family, card, port in [('hp', 4, 31711), ('mf', 5, 31712)]:
        base = 'hp-jte213-expertnorm328-preclip330' if family == 'hp' else 'mf-native-stat330'
        run = 'hp-jte213-expertnorm328-preclip397' if family == 'hp' else 'mf-native-stat397'
        config_path = ROOT / 'configs' / (run + '.yaml')
        launcher = ROOT / ('run-' + run + '.sh')
        work = STATE / ('work-' + run)
        if any(path.exists() for path in [config_path, launcher, work, ROOT / 'runs' / run]):
            raise ValueError('Existing run destination')
        config = (ROOT / 'configs' / (base + '.yaml')).read_text().replace(base, run)
        config = config.replace('  train_iters: 330', '  train_iters: 397')
        config = config.replace('  stop_step: 130', '  stop_step: 197')
        config_path.write_text(config)
        script = (ROOT / ('run-' + base + '.sh')).read_text().replace(base, run)
        script = re.sub(r'ASCEND_RT_VISIBLE_DEVICES=\d+', 'ASCEND_RT_VISIBLE_DEVICES=' + str(card), script)
        script = re.sub(r'--master_port=\d+', '--master_port=' + str(port), script)
        old_entry = 'capture_hp_expert_norm328.py' if family == 'hp' else 'capture_mf_model_hash330.py'
        entry = 'capture_hp_expert_preclip397.py' if family == 'hp' else 'capture_mf_model_hash397.py'
        script = script.replace(old_entry, entry)
        script = script.replace('P12_EXPERT328_ENTRY=observe330', 'P12_EXPERT328_ENTRY=observe397')
        if family == 'mf':
            dump = json.loads((ROOT / 'configs' / (base + '.json')).read_text())
            dump['common_dump_settings'].update(path='/alignment/phase12/diagnostic-statistics/' + run,
                                                net_name='train397', iteration='195|196')
            (ROOT / 'configs' / (run + '.json')).write_text(json.dumps(dump, indent=2))
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
               'readonly_observation': True, 'expected_training_steps': [396, 397],
               'model_hash_steps': [350, 375, 390, 395, 396, 397],
               'raw_arrays_written': False, 'config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(),
               'launcher_sha256': hashlib.sha256(launcher.read_bytes()).hexdigest(),
               'entry_sha256': hashlib.sha256((ROOT / folder / entry).read_bytes()).hexdigest()}
        if family == 'mf':
            row['dump_config_sha256'] = hashlib.sha256((ROOT / 'configs' / (run + '.json')).read_bytes()).hexdigest()
        (ROOT / 'evidence' / (run + '-launch.json')).write_text(json.dumps(row, indent=2))
        records.append(row)
    print(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
