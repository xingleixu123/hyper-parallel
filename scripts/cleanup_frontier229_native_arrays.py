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

"""Remove four redundant native229 arrays after publication and hash verification."""
from datetime import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path('/home/xxl/jt-adamw-align-20261013/phase12')
STATE = Path('/dev/shm/jt-adamw-align-20261013/state/jt-recipe-20261014')
RUN = 'frontier229-mf-builtin-tensor-dump'


def file_sha(path: Path) -> str:
    """Hash file bytes without allocating another tensor-sized buffer."""
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8_388_608), b''):
            result.update(block)
    return result.hexdigest()


def main() -> None:
    """Keep scalar arrays, manifests, source state and all causal evidence."""
    report_path = ROOT / 'evidence/frontier229-native-array-cleanup.json'
    if report_path.exists():
        raise ValueError('A cleanup receipt already exists')
    root = STATE / RUN
    if root.is_symlink() or root.resolve().parent != STATE.resolve():
        raise ValueError('Unexpected task-owned Dump root')
    audit = json.loads((ROOT / 'evidence/frontier229-builtin-boundaries.json').read_text())
    if not audit['complete'] or not all(audit['fidelity'].values()):
        raise ValueError('Native boundary audit is incomplete')
    if (ROOT / 'runs' / (RUN + '.exit')).read_text().strip() != '0':
        raise ValueError('Native Dump process is not terminal and successful')
    publication = json.loads((ROOT / 'evidence/frontier229-parent-publication.json').read_text())
    if not publication['readback_exact']:
        raise ValueError('The evidence has not been published and read back')
    candidates = [row for row in audit['arrays'] if row['boundary'] in ['pre_norm', 'normalized']]
    if len(candidates) != 4:
        raise ValueError('Expected precisely four redundant arrays')
    targets, rows, identities = [], [], set()
    for candidate in candidates:
        path = root / candidate['path']
        if path.is_symlink() or root.resolve() not in path.resolve().parents or path.suffix != '.npy':
            raise ValueError('Unsafe candidate path')
        stat = path.stat()
        if stat.st_nlink != 1 or stat.st_size != candidate['bytes'] or file_sha(path) != candidate['file_sha256']:
            raise ValueError('Candidate bytes or ownership links changed')
        targets.append(path)
        identities.add((stat.st_dev, stat.st_ino))
        rows.append({**candidate, 'uid': stat.st_uid, 'allocated_bytes': stat.st_blocks * 512,
                     'device': stat.st_dev, 'inode': stat.st_ino})
    for path in Path('/proc').glob('[0-9]*/fd/*'):
        try:
            stat = path.stat()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if (stat.st_dev, stat.st_ino) in identities:
            raise ValueError('A candidate is still open: ' + str(path))
    receipt = {
        'status': 'verified_before_removal', 'checked_at': datetime.now().astimezone().isoformat(),
        'root': str(root), 'files': rows, 'publication_commit': publication['public_commit'],
        'allocated_bytes_released': sum(row['allocated_bytes'] for row in rows),
        'logical_bytes_released': sum(row['bytes'] for row in rows),
        'retained': 'Two native square-sum scalars, IR, execution order, source/config/logs, '
                    'all actual229 source states and original faithful replays.',
        'reproduction': 'Run the saved native Dump entry with the saved two-kernel configuration '
                        'under a fresh output/run name before rerunning the array boundary auditor.',
    }
    report_path.write_text(json.dumps(receipt, indent=2))
    for path, row in zip(targets, rows):
        stat = path.stat()
        if (stat.st_dev, stat.st_ino) != (row['device'], row['inode']):
            raise ValueError('Candidate identity changed before removal')
        path.unlink()
    receipt['status'] = 'complete'
    report_path.write_text(json.dumps(receipt, indent=2))
    print(json.dumps({'status': receipt['status'], 'files': len(rows),
                      'allocated_bytes_released': receipt['allocated_bytes_released']}))


if __name__ == '__main__':
    main()
