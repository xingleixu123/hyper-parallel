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

"""Losslessly archive completed HP217 preclip arrays after full numerical audit."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil

ROOT = Path('/home/xxl/jt-adamw-align-20261013/phase12')
SOURCE = Path('/dev/shm/jt-adamw-align-20261013/state/jt-recipe-20261014/hp-jtf637-weighted-dvm-preclip217')
DEST = ROOT / 'private-archives/hp-jtf637-weighted-dvm-preclip217'
RECEIPT = ROOT / 'evidence/preclip217-hp-lossless-archive.json'


def digest(stream) -> str:
    """Hash exact original NPY bytes in bounded memory."""
    value = hashlib.sha256()
    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
        value.update(chunk)
    return value.hexdigest()


def main() -> None:
    """Compress, read back, verify unchanged sources, and remove only proven duplicates."""
    audit = json.loads((ROOT / 'evidence/preclip217-comparison.json').read_text())
    if not audit['usable_as_native_training'] or not all(all(c.values()) for c in audit['fidelity'].values()):
        raise ValueError('Completed native audit required')
    for name in ['hp-jtf637-weighted-dvm-preclip217', 'mf-preclip217']:
        if (ROOT / 'runs' / (name + '.exit')).read_text().strip() != '0':
            raise ValueError('Training still active')
    if SOURCE.is_symlink() or SOURCE.resolve() != SOURCE:
        raise ValueError('Unexpected source root')
    rows = json.loads((ROOT / 'runs/hp-jtf637-weighted-dvm-preclip217/PRECLIP-217.json').read_text())['records']
    paths = [SOURCE / row['file'] for row in rows]
    if len(paths) != 46 or any(p.is_symlink() or p.resolve().parent != SOURCE or not p.is_file() for p in paths):
        raise ValueError('Unexpected task-owned gradient files')
    targets = {str(p) for p in paths}
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            descriptors = list((proc / 'fd').iterdir())
        except (PermissionError, FileNotFoundError):
            continue
        for descriptor in descriptors:
            try:
                if os.readlink(descriptor) in targets:
                    raise RuntimeError('A gradient is still open: ' + str(descriptor))
            except FileNotFoundError:
                pass
    if shutil.disk_usage(ROOT).free < sum(p.stat().st_size for p in paths):
        raise ValueError('Insufficient archive staging space')
    DEST.mkdir(parents=True, exist_ok=False)
    report = {'complete': False, 'lossless': True, 'files': [], 'net_physical_freed_bytes': 0,
              'scope': 'Only completed HP217 gradient copies; MF217 references and all forward arrays retained.',
              'restore': 'For each record: gzip.open(archive, rb) -> source, then verify original_file_sha256.'}
    for source in paths:
        before = source.stat()
        if before.st_nlink != 1:
            raise ValueError('Unexpected shared inode')
        with source.open('rb') as stream:
            original_hash = digest(stream)
        archive = DEST / (source.name + '.gz')
        with source.open('rb') as inp, archive.open('xb') as out:
            with gzip.GzipFile(filename='', mode='wb', fileobj=out, compresslevel=1, mtime=0) as compressed:
                shutil.copyfileobj(inp, compressed, length=8 * 1024 * 1024)
        with gzip.open(archive, 'rb') as decoded:
            if digest(decoded) != original_hash:
                raise ValueError('Lossless archive verification failed')
        current = source.stat()
        if (current.st_ino, current.st_size, current.st_mtime_ns) != (before.st_ino, before.st_size, before.st_mtime_ns):
            raise ValueError('Source changed during archival')
        savings = before.st_blocks * 512 - archive.stat().st_blocks * 512
        if savings <= 0:
            raise ValueError('Archive would not save physical storage')
        row = {'source': str(source), 'archive': str(archive), 'original_file_sha256': original_hash,
               'original_size': before.st_size, 'original_physical_bytes': before.st_blocks * 512,
               'archive_size': archive.stat().st_size, 'archive_physical_bytes': archive.stat().st_blocks * 512,
               'restoration_exact': True, 'source_removed': False}
        report['files'].append(row)
        RECEIPT.write_text(json.dumps(report, indent=2))
        source.unlink()
        row['source_removed'] = True
        report['net_physical_freed_bytes'] += savings
        RECEIPT.write_text(json.dumps(report, indent=2))
        print(json.dumps({'files': len(report['files']), 'net_physical_freed_bytes': report['net_physical_freed_bytes']}), flush=True)
    report['complete'] = True
    RECEIPT.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
