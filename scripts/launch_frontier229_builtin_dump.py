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

"""Launch a bounded, single-graph native MindSpore statistics Dump."""
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
    """Capture only one replay iteration and persist invocation provenance."""
    status = subprocess.run(['npu-smi', 'info'], capture_output=True, text=True, check=True).stdout
    if 'No running processes found in NPU 6' not in status:
        raise ValueError('Requested device is no longer idle')
    if shutil.disk_usage(STATE).free < 2_000_000_000:
        raise ValueError('Insufficient state filesystem headroom')
    base = 'frontier229-mf-muon-fp32norm-cast-replay'
    run = 'frontier229-mf-builtin-stat-dump'
    destinations = [ROOT / ('run-' + run + '.sh'), ROOT / 'runs' / (run + '.log'),
                    ROOT / 'evidence' / (run + '.json'), STATE / ('work-' + run)]
    if any(path.exists() for path in destinations):
        raise ValueError('Existing run destination')
    config = {'common_dump_settings': {
        'op_debug_mode': 0, 'dump_mode': 0,
        'path': '/state/jt-recipe-20261014/' + run,
        'net_name': 'native_muon229', 'iteration': '0', 'saved_data': 'statistic',
        'input_output': 0, 'kernels': [], 'support_device': list(range(8)),
        'statistic_category': ['max', 'min', 'l2norm']},
        'e2e_dump_settings': {'enable': True, 'trans_flag': True, 'stat_calc_mode': 'host'}}
    config_path = ROOT / 'configs' / (run + '.json')
    config_path.write_text(json.dumps(config, indent=2))
    script = (ROOT / ('run-' + base + '.sh')).read_text().replace(base, run)
    script = re.sub(r'ASCEND_RT_VISIBLE_DEVICES=\d+', 'ASCEND_RT_VISIBLE_DEVICES=6', script)
    script = script.replace('replay_mf_frontier229_fp32norm_cast.py', 'replay_mf_frontier229_builtin_dump.py')
    script = script.replace('python /alignment',
        'export MINDSPORE_DUMP_CONFIG=/alignment/phase12/configs/' + config_path.name + '\n'
        'export P12_DUMP_REPORT=/alignment/phase12/evidence/' + run + '.json\npython /alignment')
    launcher = ROOT / ('run-' + run + '.sh')
    launcher.write_text(script)
    (STATE / ('work-' + run)).mkdir()
    command = ['docker', '-H', 'unix:///run/mf-image-verify-20261008/docker.sock', 'exec',
               'jt-adamw-mf-20261013', 'bash', '/alignment/phase12/' + launcher.name]
    with (ROOT / 'runs' / (run + '.dispatch.log')).open('wb') as stream:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    report = {'run': run, 'pid': process.pid, 'card': 6,
              'started_at': datetime.now().astimezone().isoformat(), 'command': command,
              'source_sha256': hashlib.sha256((ROOT / 'tools/replay_mf_frontier229_builtin_dump.py').read_bytes()).hexdigest(),
              'launcher_sha256': hashlib.sha256(launcher.read_bytes()).hexdigest(),
              'config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(),
              'scope': 'One native single-parameter graph, iteration0; statistics and IR, no full tensors.'}
    (ROOT / 'evidence' / (run + '-launch.json')).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
