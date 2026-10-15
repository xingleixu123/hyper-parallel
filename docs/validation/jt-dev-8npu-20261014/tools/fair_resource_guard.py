# Copyright 2026 Huawei Technologies Co., Ltd
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

"""Single-job device guard: never preempt other users, never retain devices between jobs."""

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


ROOT = Path('/home/xxl/jt-dev-8npu-20261014')
OWNER = '01a13aa4-a553-7e22-a1c1-ebd16b083bb9'
ALLOWED_IMAGES = {
    'sha256:4d302cd5859ea8cee8af72130937a446e79e709ed93d787b79fedd20f3a1e3a1',
    'sha256:5f9e677f439d11ab4e7c888f54b142335a27eee7411cbd2ce894a4798a70146d',
}
GAP_SECONDS = 1200
IDENTITY_FILE_BYTES = 4096
IDENTITY_PID_LIMIT = 16
OWNERSHIP_EVENT_BYTES = 8 * 1024 * 1024
RUNNER_REPOSITORIES = {
    '/work/tools/run_fair_training_case.sh': 'repo',
    '/work/tools/run_fair_pr199_st.sh': 'fixes/pr199-f637-st',
}


def cooldown_remaining(state, wall_now, monotonic_now):
    """Use the coordinator's clock, conservatively handling clock changes/reboots."""
    if 'released_wall' not in state:
        return 0
    elapsed_wall = wall_now - state['released_wall']
    elapsed_monotonic = monotonic_now - state['released_monotonic']
    elapsed = max(0, min(elapsed_wall, elapsed_monotonic))
    return max(0, GAP_SECONDS - elapsed)


def parse_cards(output):
    """Require explicit healthy, idle card rows and explicit zero-process evidence."""
    cards = {}
    current = None
    for line in output.splitlines():
        header = re.match(r'^\|\s*(\d+)\s+910B\w*\s*\|\s*(\w+)', line)
        if header:
            current = int(header.group(1))
            cards[current] = {'healthy': header.group(2) == 'OK', 'pids': [], 'empty': False}
        usage = re.match(r'^\|\s*0\s*\|\s*\S+\s*\|\s*(\d+)\s+(\d+)\s*/\s*(\d+)\s+(\d+)\s*/\s*(\d+)', line)
        if usage and current is not None:
            cards[current].update(core=int(usage.group(1)), hbm_mb=int(usage.group(4)))
        empty = re.search(r'No running processes found in NPU\s+(\d+)', line)
        if empty and int(empty.group(1)) in cards:
            cards[int(empty.group(1))]['empty'] = True
        process = re.match(r'^\|\s*(\d+)\s+0\s*\|\s*(\d+)\s*\|', line)
        if process and int(process.group(1)) in cards:
            cards[int(process.group(1))]['pids'].append(int(process.group(2)))
    return cards


def card_observation():
    """Keep the raw device report alongside its parsed process identities."""
    output = subprocess.check_output(['npu-smi', 'info'], text=True, timeout=20)
    return {'raw': output, 'cards': parse_cards(output)}


def card_snapshot():
    return card_observation()['cards']


def process_identity(pid: int) -> dict:
    """Capture bounded, non-command-line ownership evidence while a process exists."""
    path = Path('/proc') / str(pid)
    result = {'pid': pid, 'observed_wall': time.time()}
    for name in ('status', 'stat', 'cgroup'):
        try:
            with (path / name).open('rb') as stream:
                data = stream.read(IDENTITY_FILE_BYTES + 1)
            result[name] = data[:IDENTITY_FILE_BYTES].decode('utf-8', errors='backslashreplace')
            result[name + '_truncated'] = len(data) > IDENTITY_FILE_BYTES
        except OSError as exc:
            result[name + '_error'] = str(exc)
    try:
        result['pid_namespace'] = os.readlink(path / 'ns/pid')
    except OSError as exc:
        result['pid_namespace_error'] = str(exc)
    return result


def free_cards(cards):
    return [index for index, row in cards.items() if row['healthy'] and row['empty'] and not row['pids']
            and row.get('core', -1) == 0 and row.get('hbm_mb', 999999) < 4096]


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def _save_ownership_event(path: Path, value: dict) -> None:
    """Keep diagnostic evidence below the collector's per-file budget."""
    payload = (json.dumps(value, indent=2) + '\n').encode('utf-8')
    if len(payload) > OWNERSHIP_EVENT_BYTES:
        value = {key: value[key] for key in (
            'observed_wall', 'container_id', 'devices', 'reported_unknown_pids')}
        value.update(evidence_truncated=True, original_serialized_bytes=len(payload))
    save(path, value)


def remote_job(request):
    case = request['case']
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{1,100}', case):
        raise ValueError('Invalid case name')
    if request['host'] not in ('125', '129') or request['cards'] not in (1, 2, 4, 8):
        raise ValueError('Invalid host/card request')
    image = request['image']
    if image not in ALLOWED_IMAGES:
        raise ValueError('Image must match a verified immutable image ID')
    if not 1 <= request['timeout_seconds'] <= 900:
        raise ValueError('Every experiment must have a hard limit of at most 900 seconds')
    if request['runner'] not in RUNNER_REPOSITORIES:
        raise ValueError('Runner is outside the authorized one-experiment entry point')
    if request['runner'] == '/work/tools/run_fair_pr199_st.sh' and request['cards'] != 4:
        raise ValueError('PR199 regression requires exactly four devices')
    if (ROOT / 'runs' / case).exists():
        raise ValueError('Refusing to replace an existing run')
    repo = (ROOT / RUNNER_REPOSITORIES[request['runner']]).resolve()
    if ROOT not in repo.parents:
        raise ValueError('Source checkout resolves outside the task directory')
    source = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    patch = subprocess.check_output(['git', '-C', str(repo), 'diff', '--binary', 'HEAD'])
    if source != request['source'] or hashlib.sha256(patch).hexdigest() != request['patch_sha256']:
        raise ValueError('Source or patch fingerprint changed')
    for relative, expected in request['files'].items():
        path = (ROOT / relative).resolve()
        if ROOT not in path.parents or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Pinned input changed: ' + relative)
    directory = ROOT / 'scheduling'
    directory.mkdir(exist_ok=True)
    save(directory / (case + '.request.json'), request)
    before = card_snapshot()
    devices = free_cards(before)[:request['cards']]
    receipt = {'case': case, 'host': request['host'], 'owner': OWNER, 'attempted': False,
               'released': True, 'devices': devices, 'source': source, 'image': image,
               'patch_sha256': request['patch_sha256'], 'before': before}
    if len(devices) != request['cards']:
        receipt['reason'] = 'insufficient-free-devices'
        return receipt
    time.sleep(5)
    if not set(devices).issubset(free_cards(card_snapshot())):
        receipt['reason'] = 'device-became-busy-before-launch'
        return receipt
    docker = ['docker'] if request['host'] == '129' else [
        'docker', '-H', 'unix:///run/hp-pr-two-commits-20261008/docker.sock']
    name = 'jt-fair-' + case
    command = docker + ['create', '--name', name, '--label', 'codex.owner=' + OWNER,
                        '--label', 'codex.case=' + case, '--network', 'host', '--shm-size', '32g',
                        '--memory', '320g', '--pids-limit', '4096', '--entrypoint', '/bin/bash',
                        '-w', '/work', '-v', str(ROOT) + ':/work',
                        '-v', str(repo) + ':/workspace/hyper-parallel']
    for path in ('/usr/local/Ascend/driver', '/usr/local/Ascend/firmware', '/usr/local/dcmi',
                 '/usr/local/bin/npu-smi', '/etc/ascend_install.info', '/etc/hccn.conf', '/etc/vnpu.cfg'):
        if Path(path).exists():
            command += ['-v', path + ':' + path + ':ro']
    for index in devices:
        command += ['--device', '/dev/davinci' + str(index)]
    for path in ('/dev/davinci_manager', '/dev/devmm_svm', '/dev/hisi_hdc'):
        command += ['--device', path]
    visible = ','.join(str(index) for index in devices)
    command += ['-e', 'ASCEND_RT_VISIBLE_DEVICES=' + visible, image,
                request['runner'], case, str(request['port']), str(request['cards'])]
    container = subprocess.check_output(command, text=True).strip()
    receipt['container_id'] = container
    receipt['container_name'] = name
    owned = set()

    def inspect():
        info = json.loads(subprocess.check_output(docker + ['inspect', container], text=True))[0]
        if info['Config']['Labels'].get('codex.owner') != OWNER or info['Id'] != container:
            raise RuntimeError('Container ownership mismatch; no stop action allowed')
        return info

    def stop_owned():
        if inspect()['State']['Running']:
            subprocess.run(docker + ['stop', '--time', '10', container], check=True, capture_output=True)

    started = time.monotonic()
    try:
        receipt.update(attempted=True, released=False, started_wall=time.time())
        save(directory / (case + '.state.json'), receipt)
        with (directory / (case + '.container.log')).open('w') as log:
            process = subprocess.Popen(docker + ['start', '-a', container], stdout=log, stderr=subprocess.STDOUT)
            while process.poll() is None:
                time.sleep(5)
                info = inspect()
                if not info['State']['Running']:
                    continue
                if time.monotonic() - started > request['timeout_seconds']:
                    receipt['reason'] = 'experiment-wall-time-limit'
                    stop_owned()
                    continue
                top = subprocess.run(docker + ['top', container, '-eo', 'pid'], capture_output=True, text=True)
                if top.returncode:
                    continue
                owned.update(int(value) for value in top.stdout.split() if value.isdigit())
                observation = card_observation()
                snapshot = observation['cards']
                foreign = {pid for index in devices for pid in snapshot.get(index, {}).get('pids', [])
                           if pid not in owned and Path('/proc/' + str(pid)).exists()}
                if foreign:
                    identity_candidates_before = len(foreign)
                    identities_before = {pid: process_identity(pid) for pid in sorted(foreign)[:IDENTITY_PID_LIMIT]}
                    # A worker may start between docker top and the NPU snapshot.
                    refreshed = subprocess.run(docker + ['top', container, '-eo', 'pid'],
                                               capture_output=True, text=True)
                    if refreshed.returncode:
                        continue
                    owned.update(int(value) for value in refreshed.stdout.split() if value.isdigit())
                    foreign.difference_update(owned)
                if foreign:
                    event_path = directory / (case + '.ownership-event.json')
                    _save_ownership_event(event_path, {
                        'observed_wall': time.time(), 'container_id': container,
                        'container_init': process_identity(info['State']['Pid']),
                        'devices': devices, 'npu_smi': observation['raw'][:65536],
                        'npu_smi_truncated': len(observation['raw']) > 65536,
                        'top_before': {'stdout': top.stdout[:65536], 'stderr': top.stderr[:65536],
                                       'truncated': len(top.stdout) > 65536 or len(top.stderr) > 65536,
                                       'returncode': top.returncode},
                        'top_after': {'stdout': refreshed.stdout[:65536], 'stderr': refreshed.stderr[:65536],
                                      'truncated': len(refreshed.stdout) > 65536 or len(refreshed.stderr) > 65536,
                                      'returncode': refreshed.returncode},
                        'owned_pids_seen': sorted(owned)[:4096], 'owned_pids_truncated': len(owned) > 4096,
                        'reported_unknown_pids': sorted(foreign)[:128],
                        'unknown_pids_truncated': len(foreign) > 128,
                        'identity_pid_limit': IDENTITY_PID_LIMIT,
                        'identity_candidates_before_refresh': identity_candidates_before,
                        'identities_before_truncated': identity_candidates_before > IDENTITY_PID_LIMIT,
                        'identities_after_truncated': len(foreign) > IDENTITY_PID_LIMIT,
                        'identities_before_refresh': identities_before,
                        'identities_after_refresh': {
                            pid: process_identity(pid) for pid in sorted(foreign)[:IDENTITY_PID_LIMIT]},
                    })
                    receipt.update(reason='foreign-job-appeared-on-selected-devices', foreign_pids=sorted(foreign))
                    receipt['ownership_event'] = str(event_path.relative_to(ROOT))
                    stop_owned()
            process.wait(timeout=30)
            receipt['exit_code'] = inspect()['State']['ExitCode']
    finally:
        stop_owned()
        info = inspect()
        if info['State']['Running']:
            raise RuntimeError('Owned container did not stop; block subsequent experiments')
        remaining = set()
        for _ in range(6):
            after = card_snapshot()
            remaining = {pid for row in after.values() for pid in row['pids']} & owned
            if not remaining:
                break
            time.sleep(5)
        receipt.update(released=not remaining, remaining_owned_pids=sorted(remaining), after=after,
                       finished_wall=time.time(), duration_seconds=time.monotonic() - started)
        save(directory / (case + '.receipt.json'), receipt)
        if not remaining:
            subprocess.run(docker + ['rm', container], check=True, capture_output=True)
    return receipt


def local_job(path):
    import msvcrt

    root = Path(__file__).resolve().parents[1]
    directory = root / 'scheduling'
    directory.mkdir(exist_ok=True)
    state_path = directory / 'global-resource-state.json'
    with (directory / 'global-resource.lock').open('a+b') as lock:
        if lock.tell() == 0:
            lock.write(b'0')
            lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        if state.get('phase') in ('running', 'uncertain'):
            raise RuntimeError('Previous job must be explicitly inspected before another launch')
        remaining = cooldown_remaining(state, time.time(), time.monotonic())
        if remaining > 0:
            print(json.dumps({'phase': 'cooldown', 'remaining_seconds': remaining}), flush=True)
            return 75
        request = json.loads(Path(path).read_text())
        state.update(phase='running', request=request, requested_wall=time.time())
        save(state_path, state)
        ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', request['host'] + '-root',
               'python3', (ROOT / 'tools/fair_resource_guard.py').as_posix(), '--remote']
        print(json.dumps({'phase': 'checking-and-launching', 'host': request['host'], 'case': request['case']}), flush=True)
        try:
            result = subprocess.run(ssh, input=json.dumps(request), capture_output=True, text=True,
                                    timeout=request['timeout_seconds'] + 180)
            (directory / (request['case'] + '.launcher.log')).write_text(result.stdout + result.stderr)
            receipts = [json.loads(line)['final_receipt'] for line in result.stdout.splitlines()
                        if line.startswith('{"final_receipt":')]
            if len(receipts) != 1 or not receipts[0]['released']:
                raise RuntimeError('Missing verified device-release receipt; do not launch another experiment')
            receipt = receipts[0]
            save(directory / (request['case'] + '.receipt.json'), receipt)
            if receipt['attempted']:
                state.update(phase='cooldown', released_wall=time.time(), released_monotonic=time.monotonic(),
                             minimum_gap_seconds=GAP_SECONDS, last_receipt=receipt)
            else:
                state.update(phase='waiting-for-free-cards', last_receipt=receipt)
            save(state_path, state)
            print(json.dumps({'phase': state['phase'], 'receipt': receipt}), flush=True)
            return receipt.get('exit_code', 75)
        except BaseException:
            state['phase'] = 'uncertain'
            save(state_path, state)
            raise


if __name__ == '__main__':
    if sys.argv[1] == '--remote':
        outcome = remote_job(json.load(sys.stdin))
        print(json.dumps({'final_receipt': outcome}), flush=True)
    else:
        raise SystemExit(local_job(sys.argv[1]))
