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

"""Remove completed328/329 diagnostic arrays after preserving verified identities and metadata."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path('/home/xxl/jt-adamw-align-20261013/phase12')
STATE = Path('/dev/shm/jt-adamw-align-20261013/state/jt-recipe-20261014')
TARGETS = {'hp-jte213-fc1norm-qupboth-state328': 6, 'mf-frontier328-state': 5,
           'hp-jte213-fc1norm-qupboth-state329': 6, 'mf-frontier329-state': 5}
EVIDENCE = ROOT / 'evidence/frontier328-completed-boundary-state-cleanup'


def main() -> None:
    """Keep the MF200 acceptance checkpoint and all historical logs and manifests."""
    if EVIDENCE.exists():
        raise ValueError('Cleanup receipt already exists')
    for name in ['frontier328-expert-independent200-audit.json',
                 'frontier328-expert-native-statistics-comparison.json',
                 'frontier328-native-cause-audit.json', 'frontier328-normalization-precision.json']:
        evidence = json.loads((ROOT / 'evidence' / name).read_text())
        if not evidence.get('complete'):
            raise ValueError('Completed stage evidence is missing: ' + name)
    acceptance = json.loads((ROOT / 'evidence/frontier328-expert-independent200-audit.json').read_text())
    if not acceptance['total_loss_passed']:
        raise ValueError('Current total-loss phase has not passed')
    paths = []
    for run, count in TARGETS.items():
        target = STATE / run
        if target.is_symlink() or target.resolve() != STATE.resolve() / run:
            raise ValueError('Unexpected cleanup root')
        if (ROOT / 'runs' / (run + '.exit')).read_text().strip() != '0':
            raise ValueError('Diagnostic state capture is incomplete')
        selected = sorted(target.rglob('*.npy'))
        if len(selected) != count:
            raise ValueError('Unexpected completed state inventory')
        paths.extend(selected)
    for path in paths:
        if path.is_symlink() or not path.is_file() or STATE.resolve() not in path.resolve().parents or path.resolve().relative_to(STATE).parts[0] not in TARGETS:
            raise ValueError('Unsafe cleanup target')
        if path.stat().st_nlink != 1:
            raise ValueError('Shared inode requires separate accounting')
    active = subprocess.run(['fuser', *map(str, paths)], capture_output=True, text=True)
    if active.returncode != 1 or active.stdout.strip():
        raise ValueError('A diagnostic tensor is in use or fuser failed')
    EVIDENCE.mkdir()
    for run in TARGETS:
        for path in (STATE / run).rglob('*.json'):
            destination = EVIDENCE / 'metadata' / path.relative_to(STATE)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
    rows = []
    for path in paths:
        hasher = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(16 * 1024 * 1024), b''):
                hasher.update(block)
        stat = path.stat()
        rows.append({'path': str(path), 'bytes': stat.st_size, 'allocated_bytes': stat.st_blocks * 512,
                     'inode': stat.st_ino, 'sha256': hasher.hexdigest(), 'deleted': False})
    report = {'complete': False, 'started_at': datetime.now().astimezone().isoformat(),
              'reason': '328 actual-state localization, native replay, precision study,329 holdout and independent200 acceptance are complete. '
                        'Total-loss phase has passed; no MTP/Aux diagnosis has been started. These selected boundary arrays are no longer needed. '
                        'MF200, scripts, all logs, hashes and metadata remain available to regenerate selected states.',
              'original_metadata_preserved': True, 'active_users': [], 'records': rows}
    receipt = ROOT / 'evidence/frontier328-completed-boundary-state-cleanup.json'
    receipt.write_text(json.dumps(report, indent=2))
    for path, row in zip(paths, rows):
        if path.stat().st_ino != row['inode'] or path.stat().st_size != row['bytes']:
            raise ValueError('File changed after hashing')
        path.unlink()
        row['deleted'] = not path.exists()
    report.update(complete=all(row['deleted'] for row in rows),
                  deleted_bytes=sum(row['bytes'] for row in rows),
                  deleted_allocated_bytes=sum(row['allocated_bytes'] for row in rows),
                  completed_at=datetime.now().astimezone().isoformat())
    receipt.write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key != 'records'}))


if __name__ == '__main__':
    main()
