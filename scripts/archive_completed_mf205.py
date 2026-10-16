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

"""Move completed diagnostic arrays to a verified archive on another filesystem."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile

ROOT = Path('/home/xxl/jt-adamw-align-20261013/phase12')
STATE = Path('/dev/shm/jt-adamw-align-20261013/state/jt-recipe-20261014').resolve()
SOURCE = STATE / 'mf-grad205-state'
COLD = Path('/var/tmp/xxl-jt-alignment-20261013')
ARCHIVE = COLD / 'mf-grad205-state.tar'
RECEIPT = ROOT / 'evidence/frontier229-mf205-cold-archive.json'


def sha(stream) -> str:
    """Hash a stream without allocating a whole tensor."""
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
        digest.update(block)
    return digest.hexdigest()


def require_idle_source() -> None:
    """Reject any process currently holding an original source file open."""
    for process in Path('/proc').iterdir():
        if not process.name.isdigit():
            continue
        try:
            descriptors = list((process / 'fd').iterdir())
        except (FileNotFoundError, PermissionError):
            continue
        for descriptor in descriptors:
            try:
                link = os.readlink(descriptor)
            except OSError:
                continue
            if link.startswith(str(SOURCE) + '/'):
                raise ValueError('Original state is in use by process ' + process.name)


def main() -> None:
    """Verify every archived byte before deleting only the original NPY payloads."""
    partial = ARCHIVE.with_suffix('.tar.partial')
    if SOURCE.is_symlink() or SOURCE.resolve().parent != STATE or COLD.is_symlink():
        raise ValueError('Unexpected archive source or destination')
    if any(path.exists() for path in [ARCHIVE, partial, RECEIPT]):
        raise ValueError('Archive destination already exists')
    if (ROOT / 'runs/mf-grad205-state.exit').read_text().strip() != '0':
        raise ValueError('Source experiment is not complete')
    files = sorted(path for path in SOURCE.rglob('*') if path.is_file())
    if len(files) != 177 or any(path.is_symlink() or path.suffix not in ['.npy', '.json'] for path in files):
        raise ValueError('Unexpected source inventory')
    logical = sum(path.stat().st_size for path in files)
    if shutil.disk_usage(COLD.parent).free < logical + 50_000_000_000:
        raise ValueError('Destination would leave insufficient free space')
    require_idle_source()
    COLD.mkdir(mode=0o700, exist_ok=True)
    if ARCHIVE.resolve().parent != COLD.resolve() or COLD.stat().st_dev == STATE.stat().st_dev:
        raise ValueError('Archive must be on the verified separate filesystem')
    report = {'status': 'copying', 'source': str(SOURCE), 'archive': str(ARCHIVE), 'files': [],
              'started_at': datetime.now().astimezone().isoformat(),
              'purpose': 'Retain full historical205 evidence on root filesystem; release nearly full tmpfs.',
              'current_checkpoint_untouched': 'MF200 and converted HP200 remain in place.',
              'restore': 'Verify archive entries against this manifest, then extract mf-grad205-state under STATE.'}
    RECEIPT.write_text(json.dumps(report, indent=2))
    with tarfile.open(partial, 'w') as archive:
        for path in files:
            stat = path.stat()
            if path.resolve().parent != SOURCE and SOURCE not in path.resolve().parents:
                raise ValueError('Source path escaped the task directory')
            if stat.st_nlink != 1:
                raise ValueError('Do not retire shared hardlinks')
            with path.open('rb') as stream:
                digest = sha(stream)
                stream.seek(0)
                archive.addfile(archive.gettarinfo(str(path), SOURCE.name + '/' + str(path.relative_to(SOURCE))), stream)
            if path.stat().st_mtime_ns != stat.st_mtime_ns or path.stat().st_size != stat.st_size:
                raise ValueError('Source changed while archiving')
            report['files'].append({'path': str(path.relative_to(SOURCE)), 'bytes': stat.st_size,
                                    'allocated_bytes': stat.st_blocks * 512, 'sha256': digest,
                                    'mtime_ns': stat.st_mtime_ns})
    with partial.open('rb') as stream:
        os.fsync(stream.fileno())
    expected = {SOURCE.name + '/' + row['path']: row for row in report['files']}
    seen = set()
    with tarfile.open(partial, 'r') as archive:
        for member in archive:
            if not member.isfile() or member.name not in expected or member.name in seen:
                raise ValueError('Unexpected archive member')
            with archive.extractfile(member) as stream:
                if sha(stream) != expected[member.name]['sha256']:
                    raise ValueError('Archive payload verification failed')
            seen.add(member.name)
    if seen != set(expected):
        raise ValueError('Incomplete archive')
    partial.rename(ARCHIVE)
    report.update(status='archive_verified', archive_bytes=ARCHIVE.stat().st_size, verified_files=len(seen))
    RECEIPT.write_text(json.dumps(report, indent=2))
    require_idle_source()
    deleted = []
    for row in report['files']:
        path = SOURCE / row['path']
        if path.suffix != '.npy':
            continue
        with path.open('rb') as stream:
            if sha(stream) != row['sha256']:
                raise ValueError('Source changed after archive verification')
        path.unlink()
        deleted.append(row['path'])
        report['retired_arrays'] = list(deleted)
        RECEIPT.write_text(json.dumps(report, indent=2))
    report.update(status='complete', finished_at=datetime.now().astimezone().isoformat(),
                  tmpfs_freed_allocated_bytes=sum(row['allocated_bytes'] for row in report['files']
                                                  if row['path'] in deleted),
                  root_free_bytes=shutil.disk_usage(COLD).free, tmpfs_free_bytes=shutil.disk_usage(STATE).free)
    RECEIPT.write_text(json.dumps(report, indent=2))
    (SOURCE / 'ARCHIVED.json').write_text(json.dumps({'archive': str(ARCHIVE), 'receipt': str(RECEIPT)}, indent=2))
    print(json.dumps({key: report[key] for key in ['status', 'verified_files', 'archive_bytes',
                                                  'tmpfs_freed_allocated_bytes', 'root_free_bytes', 'tmpfs_free_bytes']}))


if __name__ == '__main__':
    main()
