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

"""Remove obsolete diagnostic teacher checkpoint arrays after preserving their identities."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path('/home/xxl/jt-adamw-align-20261013/phase12')
STATE = Path('/dev/shm/jt-adamw-align-20261013/state/jt-recipe-20261014')
TARGET = STATE / 'mf-anchor357'
EVIDENCE = ROOT / 'evidence/frontier306-obsolete-anchor357-cleanup'


def main() -> None:
    """Keep the MF200 acceptance checkpoint and all historical logs and manifests."""
    if TARGET.is_symlink() or TARGET.resolve() != STATE.resolve() / 'mf-anchor357':
        raise ValueError('Unexpected cleanup root')
    if (ROOT / 'runs/mf-anchor357.exit').read_text().strip() != '0':
        raise ValueError('Diagnostic checkpoint generation is not complete')
    if EVIDENCE.exists():
        raise ValueError('Cleanup receipt already exists')
    paths = sorted(TARGET.rglob('*.npy'))
    if len(paths) != 129:
        raise ValueError('Unexpected diagnostic checkpoint inventory')
    for path in paths:
        if path.is_symlink() or not path.is_file() or TARGET.resolve() not in path.resolve().parents:
            raise ValueError('Unsafe cleanup target')
        if path.stat().st_nlink != 1:
            raise ValueError('Shared inode requires separate accounting')
    active = subprocess.run(['fuser', *map(str, paths)], capture_output=True, text=True)
    if active.returncode != 1 or active.stdout.strip():
        raise ValueError('A diagnostic tensor is in use or fuser failed')
    EVIDENCE.mkdir()
    for path in TARGET.rglob('*.json'):
        destination = EVIDENCE / 'metadata' / path.relative_to(TARGET)
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
              'reason': 'Historical teacher357 diagnostic is excluded from independent acceptance. '
                        'All continuation experiments restore MF200, which is retained. '
                        'Teacher357 values can be regenerated using the retained original run configuration.',
              'original_metadata_preserved': True, 'active_users': [], 'records': rows}
    receipt = ROOT / 'evidence/frontier306-obsolete-anchor357-cleanup.json'
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
