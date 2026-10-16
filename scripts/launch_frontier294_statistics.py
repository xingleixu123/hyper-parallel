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

"""Observe two gradient boundaries with hashes and native statistics only."""
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
    """Hash a small configuration or source."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Launch independent HP and MF observations on disjoint idle devices."""
    status = subprocess.run(['npu-smi', 'info'], check=True, capture_output=True, text=True).stdout
    if any('No running processes found in NPU ' + str(card) not in status for card in [7, 5]):
        raise ValueError('Requested devices are no longer idle')
    ports = subprocess.run(['ss', '-ltn'], check=True, capture_output=True, text=True).stdout
    if any(':' + str(port) + ' ' in ports for port in [31666, 31667]):
        raise ValueError('Requested rendezvous port is occupied')
    if shutil.disk_usage(STATE).free < 2_000_000_000 or shutil.disk_usage(ROOT).free < 5_000_000_000:
        raise ValueError('Statistics observations need state and output filesystem headroom')
    probe = json.loads((ROOT / 'evidence/frontier235-stat-hash-capability.json').read_text())
    if not probe['complete'] or not probe['native_state_fidelity'] or probe['tensor_file_count'] != 0:
        raise ValueError('Native statistics/hash capability was not verified')
    graph = (ROOT / 'diagnostic-statistics/mf-native-stat235/rank_0/graphs/ms_output_trace_code_graph_1.ir').read_text()
    kernels = sorted(set(re.findall(r'Fullname with scope: \((Default/GraphKernel_Mul_Mul_ReduceSum_split-op\d+)\)', graph)))
    if len(kernels) != 46:
        raise ValueError('Expected all 46 verified gradient producers')
    kernels += ['Default/network-MFTrainOneStepCell/clip_grad_norm-ClipGradNorm/AddN-op0',
                'Default/GraphKernel_Sqrt_GreaterEqual_Select_Add_Div_Minimum_split-op0']
    jobs = []
    for family, card, port in [('hp', 7, 31666), ('mf', 5, 31667)]:
        base = 'hp-jtf637-global-sqrt235' if family == 'hp' else 'mf-modelhash235'
        run = 'hp-jtf637-qup-preclip294' if family == 'hp' else 'mf-native-stat294'
        destinations = [ROOT / 'runs' / run, ROOT / ('run-' + run + '.sh'),
                        ROOT / 'runs' / (run + '.exit'), STATE / ('work-' + run)]
        if any(path.exists() for path in destinations):
            raise ValueError('Existing run destination: ' + run)
        config = (ROOT / 'configs' / (base + '.yaml')).read_text().replace(base, run)
        config = config.replace('  train_iters: 235', '  train_iters: 294').replace('  stop_step: 35', '  stop_step: 94')
        config_path = ROOT / 'configs' / (run + '.yaml')
        config_path.write_text(config)
        script = (ROOT / ('run-' + base + '.sh')).read_text().replace(base, run)
        script = re.sub(r'ASCEND_RT_VISIBLE_DEVICES=\d+', 'ASCEND_RT_VISIBLE_DEVICES=' + str(card), script)
        script = re.sub(r'--master_port=\d+', '--master_port=' + str(port), script)
        entry = 'capture_hp_qup_preclip294.py' if family == 'hp' else 'capture_mf_model_hash294.py'
        if family == 'hp':
            script = script.replace('capture_hp_global_sqrt235.py', entry)
            script = script.replace('P12_WEIGHTED_DVM_SCOPE=main', 'P12_WEIGHTED_DVM_SCOPE=both')
            script = script.replace('export ASCEND_RT_VISIBLE_DEVICES',
                                    'export P12_QUP258_SCOPE=nope\nexport ASCEND_RT_VISIBLE_DEVICES')
        else:
            script = script.replace('capture_mf_model_hash235.py', entry)
            dump_path = ROOT / 'diagnostic-statistics' / run
            if dump_path.exists():
                raise ValueError('Existing Dump destination')
            dump_path.parent.mkdir(exist_ok=True)
            dump_config = {
                'common_dump_settings': {
                    'op_debug_mode': 0, 'dump_mode': 1,
                    'path': '/alignment/phase12/diagnostic-statistics/' + run,
                    'net_name': 'train294', 'iteration': '92|93', 'saved_data': 'statistic',
                    'input_output': 2, 'kernels': kernels, 'support_device': list(range(8)),
                    'statistic_category': ['hash', 'max', 'min', 'l2norm'],
                },
                'e2e_dump_settings': {'enable': True, 'trans_flag': True, 'stat_calc_mode': 'host'},
            }
            dump_config_path = ROOT / 'configs' / (run + '.json')
            dump_config_path.write_text(json.dumps(dump_config, indent=2))
            script = script.replace('msrun --worker_num',
                                    'export MINDSPORE_DUMP_CONFIG=/alignment/phase12/configs/'
                                    + dump_config_path.name + '\nmsrun --worker_num')
        launcher = ROOT / ('run-' + run + '.sh')
        launcher.write_text(script)
        (STATE / ('work-' + run)).mkdir()
        socket = 'hp-pr-two-commits-20261008' if family == 'hp' else 'mf-image-verify-20261008'
        command = ['docker', '-H', 'unix:///run/' + socket + '/docker.sock', 'exec',
                   'jt-adamw-' + family + '-20261013', 'bash', '/alignment/phase12/' + launcher.name]
        with (ROOT / 'runs' / (run + '.dispatch.log')).open('wb') as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, start_new_session=True)
        record = {
            'run': run, 'pid': process.pid, 'card': card, 'port': port,
            'started_at': datetime.now().astimezone().isoformat(), 'command': command,
            'config_sha256': digest(config_path), 'launcher_sha256': digest(launcher),
            'entry_sha256': digest(ROOT / ('tools-f637' if family == 'hp' else 'tools') / entry),
            'readonly_observation': True, 'expected_training_steps': [293, 294],
            'raw_arrays_written': False, 'free_bytes': {
                'state': shutil.disk_usage(STATE).free, 'statistics': shutil.disk_usage(ROOT).free,
            },
        }
        if family == 'mf':
            record['dump_config_sha256'] = digest(dump_config_path)
            record['iteration_mapping_requires_verification'] = True
        (ROOT / 'evidence' / (run + '-launch.json')).write_text(json.dumps(record, indent=2))
        jobs.append(record)
    print(json.dumps(jobs, indent=2))


if __name__ == '__main__':
    main()
