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

"""Remove verified duplicate task-owned kernel-probe arrays after native fidelity passed."""
import hashlib
import json
from pathlib import Path

root = Path('/home/xxl/jt-adamw-align-20261013/phase12')
state = Path('/dev/shm/jt-adamw-align-20261013/state/jt-recipe-20261014')
receipt = root / 'evidence/ce226-duplicate-array-cleanup.json'
if receipt.exists() or not json.loads((root / 'evidence/ce226-kernel-fidelity.json').read_text())['complete']:
    raise ValueError('Cleanup already accounted or fidelity not complete')
for run in ['mf-ce226', 'mf-ce226-kernel']:
    if (root / 'runs' / (run + '.exit')).read_text().strip() != '0':
        raise ValueError('Run is not complete')
def records(run):
    last = json.loads((root / 'runs' / run / 'CE226_OPERANDS.jsonl').read_text().splitlines()[-1])
    return {(r['name'], r['kind']): r for r in last['records']}
def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()
base, duplicate = records('mf-ce226'), records('mf-ce226-kernel')
plan = []
for key, row in duplicate.items():
    source = state / 'mf-ce226-kernel/deep' / row['file']
    canonical = state / 'mf-ce226/deep' / base[key]['file']
    if (source.is_symlink() or canonical.is_symlink() or source.resolve().parent != state / 'mf-ce226-kernel/deep'
            or canonical.resolve().parent != state / 'mf-ce226/deep'):
        raise ValueError('Unexpected array path')
    stat = source.stat()
    if stat.st_uid != 0 or stat.st_nlink != 1 or row['sha256'] != base[key]['sha256']:
        raise ValueError('Ownership or observed checksum mismatch')
    checksum = digest(source)
    if checksum != digest(canonical):
        raise ValueError('Complete file bytes differ')
    plan.append({'source': str(source), 'canonical': str(canonical), 'file_sha256': checksum,
                 'logical_bytes': stat.st_size, 'physical_bytes': stat.st_blocks * 512})
paths = {Path(row['source']) for row in plan}
for process in Path('/proc').glob('[0-9]*'):
    try:
        for fd in (process / 'fd').iterdir():
            try:
                if fd.resolve() in paths:
                    raise RuntimeError('A process still has a duplicate array open')
            except (OSError, PermissionError):
                pass
    except (OSError, PermissionError):
        pass
report = {'complete': False, 'files': [], 'net_physical_freed_bytes': 0,
          'restore': 'Copy canonical back to source and verify file_sha256; the original CE arrays remain.'}
for row in plan:
    Path(row['source']).unlink()
    report['files'].append(row)
    report['net_physical_freed_bytes'] += row['physical_bytes']
    receipt.write_text(json.dumps(report, indent=2))
report['complete'] = True
receipt.write_text(json.dumps(report, indent=2))
print(json.dumps({'complete': True, 'files': len(plan), 'net_physical_freed_bytes': report['net_physical_freed_bytes']}))
