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

"""Deduplicate completed owned MoE/shared observations using verified full-file hashes."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path('/home/xxl/jt-adamw-align-20261013/phase12')
STATE = Path('/dev/shm/jt-adamw-align-20261013/state/jt-recipe-20261014')
RUNS = ['mf-moe205-boundaries206', 'hp-jtf637-adamw-all-moe205206', 'mf-moe205-branches206-r2', 'hp-jtf637-adamw-all-moebranches206', 'mf-sharedfc1-205206', 'hp-jtf637-adamw-all-sharedfc1206', 'mf-sharedactrow205206', 'hp-jtf637-adamw-all-sharedactrow206']


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for piece in iter(lambda: stream.read(8*1024*1024), b''):
            value.update(piece)
    return value.hexdigest()


def main():
    groups = {}
    records = []
    for run in RUNS:
        if (ROOT/'runs'/(run+'.exit')).read_text().strip() != '0':
            raise ValueError('Only completed observations may be deduplicated')
        folder = STATE/run
        if folder.is_symlink() or not folder.resolve().is_relative_to(STATE.resolve()):
            raise ValueError('Unsafe state directory')
        for path in sorted(folder.rglob('*.npy')):
            if path.is_symlink() or not path.resolve().is_relative_to(folder.resolve()):
                raise ValueError('Unsafe observation')
            size = path.stat().st_size
            key = (size, digest(path))
            if key not in groups:
                groups[key] = path
                continue
            source = groups[key]
            old = path.stat()
            if old.st_ino == source.stat().st_ino:
                continue
            temporary = path.with_name(path.name+'.dedup-link')
            if temporary.exists():
                raise ValueError('Temporary dedup path already exists')
            os.link(source, temporary)
            if digest(temporary) != key[1] or digest(path) != key[1]:
                raise ValueError('Observation changed during dedup')
            os.replace(temporary, path)
            saved = old.st_blocks*512 if old.st_nlink == 1 else 0
            records.append({'path': str(path), 'source': str(source), 'sha256': key[1],
                            'bytes': size, 'physical_bytes_saved': saved})
    report = {'complete': True, 'files': records,
              'physical_bytes_saved': sum(row['physical_bytes_saved'] for row in records),
              'scope': 'All observation paths preserved; only byte-identical completed owned arrays hardlinked.'}
    target = ROOT/'evidence/shared205-array-dedup.json'
    if target.exists():
        raise ValueError('Cleanup receipt already exists; do not overwrite accounting')
    target.write_text(json.dumps(report, indent=2))
    print(json.dumps({'files': len(records), 'physical_bytes_saved': report['physical_bytes_saved']}))


if __name__ == '__main__':
    main()
