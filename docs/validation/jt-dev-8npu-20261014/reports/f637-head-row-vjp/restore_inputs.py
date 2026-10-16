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

"""将已验证的真实输出头局部反向诊断原始证据恢复到全新目录，不覆盖已有文件。"""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile


def digest(path: Path) -> str:
    """流式计算文件 SHA256，避免保留完整归档。"""
    result = hashlib.sha256()
    with path.open('rb') as handle:
        while data := handle.read(1024 * 1024):
            result.update(data)
    return result.hexdigest()


def main() -> None:
    """先核验包和路径，再逐文件核验原字节；目标必须尚不存在。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path, help='尚不存在的恢复目录')
    args = parser.parse_args()
    bundle = Path(__file__).resolve().parent
    manifest = json.loads((bundle / 'head-row-inputs-manifest.json').read_bytes())
    archive = bundle / 'head-row-inputs.tar.gz'
    if archive.stat().st_size != manifest['archive_bytes'] or digest(archive) != manifest['archive_sha256']:
        raise ValueError('归档大小或 SHA 不一致')
    entries = manifest['files']
    expected = {row['path']: row for row in entries}
    if len(expected) != len(entries) or sum(row['bytes'] for row in entries) != manifest['raw_bytes']:
        raise ValueError('清单重复或总大小不一致')
    destination = args.destination.resolve()
    if destination.exists() or args.destination.is_symlink():
        raise FileExistsError('拒绝覆盖已有目标目录')
    for name in expected:
        relative = PurePosixPath(name)
        if relative.is_absolute() or any(part in ('', '.', '..') or ':' in part or '\\' in part
                                         for part in relative.parts):
            raise ValueError('清单包含不安全路径')
        if destination not in (destination / name).resolve().parents:
            raise ValueError('清单路径越界')
    destination.mkdir(parents=False)
    seen = set()
    with tarfile.open(archive, 'r:gz') as source:
        for member in source:
            if not member.isfile() or member.name not in expected or member.name in seen:
                raise ValueError('归档存在重复、额外或非普通文件')
            row = expected[member.name]
            if member.size != row['bytes']:
                raise ValueError('归档成员大小不一致')
            target = destination / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            checksum = hashlib.sha256()
            with source.extractfile(member) as incoming, target.open('xb') as outgoing:
                while data := incoming.read(1024 * 1024):
                    checksum.update(data)
                    outgoing.write(data)
            if target.stat().st_size != row['bytes'] or checksum.hexdigest() != row['sha256']:
                raise ValueError('恢复文件与原始 SHA 不一致')
            seen.add(member.name)
    if seen != expected.keys():
        raise ValueError('归档缺少文件')
    print(json.dumps({'status': 'PASS', 'files': len(seen), 'bytes': manifest['raw_bytes'],
                      'destination': str(destination)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
