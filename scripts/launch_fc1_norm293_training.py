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

"""Launch independent full and observed short fc1-normalization ablations."""
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
    """Keep model, data, restored state and existing factors fixed."""
    audit = json.loads((ROOT / 'evidence/jt-e213-independent200-audit.json').read_text())
    if not audit['complete'] or not all(audit['fidelity'].values()):
        raise ValueError('Latest source regression not verified')
    replay = json.loads((ROOT / 'evidence/frontier293-hp-fc1-replay.json').read_text())
    if not all(row['weight_exact_mf_training'] and row['momentum_exact_training'] for row in replay['checks']):
        raise ValueError('Candidate failed complete target-state replay')
    status = subprocess.run(['npu-smi', 'info'], check=True, capture_output=True, text=True).stdout
    if any('No running processes found in NPU ' + str(card) not in status for card in [7, 4]):
        raise ValueError('Requested independent devices are busy')
    ports = subprocess.run(['ss', '-ltn'], check=True, capture_output=True, text=True).stdout
    if any(':' + str(port) + ' ' in ports for port in [31675, 31676]):
        raise ValueError('Rendezvous port is occupied')
    if min(shutil.disk_usage(ROOT).free, shutil.disk_usage(STATE).free) < 2_000_000_000:
        raise ValueError('Insufficient output headroom')
    base = 'hp-jte213-qup-nope400'
    records = []
    for run, card, port, steps, entry in [
            ('hp-jte213-fc1norm400', 7, 31675, 400, 'capture_hp_fc1_norm293.py'),
            ('hp-jte213-fc1norm-preclip294', 4, 31676, 294, 'capture_hp_fc1_preclip294.py')]:
        config_path = ROOT / 'configs' / (run + '.yaml')
        launcher = ROOT / ('run-' + run + '.sh')
        work = STATE / ('work-' + run)
        if any(path.exists() for path in [config_path, launcher, work, ROOT / 'runs' / run]):
            raise ValueError('Existing run destination')
        config = (ROOT / 'configs' / (base + '.yaml')).read_text().replace(base, run)
        config = config.replace('  train_iters: 400', '  train_iters: ' + str(steps))
        config_path.write_text(config)
        script = (ROOT / ('run-' + base + '.sh')).read_text().replace(base, run)
        script = re.sub(r'ASCEND_RT_VISIBLE_DEVICES=\d+', 'ASCEND_RT_VISIBLE_DEVICES=' + str(card), script)
        script = re.sub(r'--master_port=\d+', '--master_port=' + str(port), script)
        script = script.replace('/tools-e213/capture_hp_q_up258.py', '/tools-e213/' + entry)
        if steps == 294:
            script = script.replace('P12_NORM219_MODEL_HASH=0', 'P12_NORM219_MODEL_HASH=1')
        launcher.write_text(script)
        work.mkdir()
        command = ['docker', '-H', 'unix:///run/hp-pr-two-commits-20261008/docker.sock', 'exec',
                   'jt-adamw-hp-20261013', 'bash', '/alignment/phase12/' + launcher.name]
        with (ROOT / 'runs' / (run + '.dispatch.log')).open('wb') as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, start_new_session=True)
        row = {'run': run, 'pid': process.pid, 'card': card, 'port': port,
               'started_at': datetime.now().astimezone().isoformat(), 'command': command,
               'comparison_control': base, 'train_iters': steps, 'teacher_values_used': False,
               'factor': 'Independent 9776-element reduction only for dense fc1 shape12288x5120',
               'config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(),
               'launcher_sha256': hashlib.sha256(launcher.read_bytes()).hexdigest(),
               'entry_sha256': hashlib.sha256((ROOT / 'tools-e213' / entry).read_bytes()).hexdigest(),
               'factor_sha256': hashlib.sha256((ROOT / 'tools-f637/hp_fc1_norm293_factor.py').read_bytes()).hexdigest()}
        (ROOT / 'evidence' / (run + '-launch.json')).write_text(json.dumps(row, indent=2))
        records.append(row)
    print(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
