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

"""Losslessly archive completed219 native state arrays after fidelity and causal audits."""
import datetime
import gzip
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path('/home/xxl/jt-adamw-align-20261013/phase12')
STATE = Path('/dev/shm/jt-adamw-align-20261013/state/jt-recipe-20261014')
DEST = ROOT / 'private-archives/frontier219-native-state'
RECEIPT = ROOT / 'evidence/frontier219-state-lossless-archive.json'
NAMES = ['hp-jtf637-packed217-state219', 'mf-frontier219-state']


def digest_stream(stream):
    value = hashlib.sha256()
    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
        value.update(chunk)
    return value.hexdigest()


def main():
    assert not RECEIPT.exists() and not DEST.exists()
    assert all((ROOT / 'runs' / (run + '.exit')).read_text().strip() == '0' for run in NAMES)
    assert json.loads((ROOT / 'evidence/frontier219-state-comparison.json').read_text())['complete']
    assert json.loads((ROOT / 'evidence/norm219-independent200-audit.json').read_text())['complete']
    files = []
    for name in NAMES:
        directory = STATE / name
        assert directory.resolve() == directory and directory.is_dir()
        files.extend(sorted(directory.rglob('*.npy')))
    assert files
    paths = {path.resolve() for path in files}
    for process in Path('/proc').glob('[0-9]*'):
        try:
            for fd in (process / 'fd').iterdir():
                try:
                    assert fd.resolve() not in paths, 'Active process is reading a replay array'
                except (FileNotFoundError, PermissionError, OSError):
                    pass
        except (FileNotFoundError, PermissionError, OSError):
            pass
    assert shutil.disk_usage(ROOT).free > sum(path.stat().st_size for path in files) + 5_000_000_000
    DEST.mkdir(parents=True)
    report = {'complete': False, 'lossless': True, 'files': [], 'retained_uncompressed': [],
              'net_physical_freed_bytes': 0, 'scope': NAMES,
              'restore': 'gzip.open(archive, rb) -> source; verify original_file_sha256 before reuse.',
              'checked_at': datetime.datetime.now().astimezone().isoformat()}
    for path in files:
        stat = path.stat()
        assert not path.is_symlink() and path.resolve().is_relative_to(STATE)
        assert stat.st_uid == 0 and stat.st_nlink == 1
        if stat.st_size < 1_000_000:
            report['retained_uncompressed'].append({'source': str(path), 'bytes': stat.st_size})
            continue
        output = DEST / path.relative_to(STATE).with_suffix('.npy.gz')
        output.parent.mkdir(parents=True, exist_ok=True)
        assert not output.exists()
        with path.open('rb') as stream:
            original_hash = digest_stream(stream)
        with path.open('rb') as source, output.open('xb') as raw:
            with gzip.GzipFile(fileobj=raw, mode='wb', compresslevel=1, mtime=0) as target:
                shutil.copyfileobj(source, target, 8 * 1024 * 1024)
        with gzip.open(output, 'rb') as stream:
            assert digest_stream(stream) == original_hash
        assert path.stat().st_mtime_ns == stat.st_mtime_ns and path.stat().st_size == stat.st_size
        saving = stat.st_blocks * 512 - output.stat().st_blocks * 512
        assert saving > 0, 'Retain original if compression is not beneficial'
        row = {'source': str(path), 'archive': str(output), 'original_bytes': stat.st_size,
               'original_file_sha256': original_hash, 'archive_bytes': output.stat().st_size,
               'net_physical_freed_bytes': saving, 'decompressed_full_sha_exact': True}
        path.unlink()
        report['files'].append(row)
        report['net_physical_freed_bytes'] += saving
        RECEIPT.write_text(json.dumps(report, indent=2))
        print(path.name, saving, flush=True)
    report['complete'] = True
    report['total_original_files_accounted'] = len(files)
    assert len(report['files']) + len(report['retained_uncompressed']) == len(files)
    RECEIPT.write_text(json.dumps(report, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ['files', 'retained_uncompressed']}), flush=True)


if __name__ == '__main__':
    main()
