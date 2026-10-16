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

"""Validate one live reduction factor in independent200 and197-step observation runs."""
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
    """Hash a bounded experiment source or configuration."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Require actual-state causality before spending two independent single-card runs."""
    audit = json.loads((ROOT / 'evidence/frontier396-state-comparison.json').read_text())
    if not audit['complete'] or not all(all(row.values()) for row in audit['fidelity'].values()):
        raise ValueError('Actual-state observation fidelity is not established')
    if any(audit['states']['395'][key]['fp32_bitwise_different'] for key in ['weight', 'muon_m']):
        raise ValueError('Previous selected state is not identical')
    if audit['clipped_gradient']['fp32_bitwise_different'] or audit['states']['396']['muon_m']['fp32_bitwise_different']:
        raise ValueError('Gradient or updated momentum is an earlier differing boundary')
    control = json.loads((ROOT / 'evidence/frontier396-hp-muon-replay.json').read_text())
    if not control['complete'] or not all(row['weight_exact_training'] and row['momentum_exact_training']
                                         for row in control['checks']):
        raise ValueError('Actual HP update has not been reproduced')
    candidate = json.loads((ROOT / 'evidence/frontier396-hp-shared-down-replay.json').read_text())
    if not candidate['complete'] or not all(row['weight_exact_mf_training'] and row['momentum_exact_training']
                                           for row in candidate['checks']):
        raise ValueError('Candidate has not reproduced the complete MF update from actual HP inputs')
    if candidate['candidate_policy']['source_sha256'] != digest(ROOT / 'tools-f637/hp_shared_down396_factor.py'):
        raise ValueError('The validated candidate source changed')
    status = subprocess.run(['npu-smi', 'info'], check=True, capture_output=True, text=True).stdout
    cards = [card for card in [4, 5, 6, 7, 2, 3, 0, 1]
             if 'No running processes found in NPU ' + str(card) in status][:2]
    if len(cards) != 2:
        raise ValueError('Two isolated idle devices are unavailable')
    ports = subprocess.run(['ss', '-ltn'], check=True, capture_output=True, text=True).stdout
    if any(':' + str(port) + ' ' in ports for port in [31717, 31718]):
        raise ValueError('Rendezvous port is occupied')
    free = {'state': shutil.disk_usage(STATE).free, 'metadata': shutil.disk_usage(ROOT).free}
    if min(free.values()) < 3_000_000_000:
        raise ValueError('Insufficient metadata headroom')
    memory = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    available = int(memory['MemAvailable'].split()[0]) * 1024
    if available < 100_000_000_000:
        raise ValueError('Insufficient host memory for two independent jobs and model hashes')
    jobs = [
        ('hp-jte213-expertnorm328400', 'hp-jte213-shareddown396400', cards[0], 31717,
         'capture_hp_expert_norm328.py', 200),
        ('hp-jte213-expertnorm328-preclip397', 'hp-jte213-shareddown396-preclip397', cards[1], 31718,
         'capture_hp_expert_preclip397.py', 197),
    ]
    for _, run, _, _, _, _ in jobs:
        if any(path.exists() for path in [ROOT / 'configs' / (run + '.yaml'), ROOT / ('run-' + run + '.sh'),
                                         ROOT / 'runs' / run, STATE / ('work-' + run)]):
            raise ValueError('Existing validation destination: ' + run)
    reports = []
    for base, run, card, port, old_entry, count in jobs:
        config = ROOT / 'configs' / (run + '.yaml')
        launcher = ROOT / ('run-' + run + '.sh')
        config.write_text((ROOT / 'configs' / (base + '.yaml')).read_text().replace(base, run))
        script = (ROOT / ('run-' + base + '.sh')).read_text().replace(base, run)
        script = script.replace(old_entry, 'capture_hp_shared_down396.py')
        script = re.sub(r'ASCEND_RT_VISIBLE_DEVICES=\d+', 'ASCEND_RT_VISIBLE_DEVICES=' + str(card), script)
        script = re.sub(r'--master_port=\d+', '--master_port=' + str(port), script)
        launcher.write_text(script)
        (STATE / ('work-' + run)).mkdir()
        command = ['docker', '-H', 'unix:///run/hp-pr-two-commits-20261008/docker.sock', 'exec',
                   'jt-adamw-hp-20261013', 'bash', '/alignment/phase12/' + launcher.name]
        with (ROOT / 'runs' / (run + '.dispatch.log')).open('wb') as stream:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        report = {'run': run, 'base': base, 'card': card, 'port': port, 'pid': process.pid,
                  'expected_steps': count, 'started_at': datetime.now().astimezone().isoformat(),
                  'command': command, 'free_bytes_at_launch': free, 'host_available_bytes': available,
                  'config_sha256': digest(config), 'launcher_sha256': digest(launcher),
                  'entry_sha256': digest(ROOT / 'tools-e213/capture_hp_shared_down396.py'),
                  'factor_sha256': digest(ROOT / 'tools-f637/hp_shared_down396_factor.py'),
                  'teacher_values_used': False, 'raw_arrays_written': False,
                  'scope': 'Only live5120x3072 shared-expert normalization reduction; every step, both branches.'}
        (ROOT / 'evidence' / (run + '-launch.json')).write_text(json.dumps(report, indent=2))
        reports.append(report)
    print(json.dumps(reports, indent=2))


if __name__ == '__main__':
    main()
