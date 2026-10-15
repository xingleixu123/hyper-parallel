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

"""Read fixed uncompressed checkpoint arrays with audited SHA256 and no extraction copies."""
import hashlib
import json
import struct
from pathlib import Path
import zipfile
import numpy as np

ORIGINAL_LOAD = np.load
ROOT = Path('/state/jt-recipe-20261014/hp-offline-step200')
REFERENCE = Path('/alignment/phase12/runs/hp-preclip-vocab201-r2/RESTORE.json')
VALIDATED = {}


class MappedArchive:
    """Support the read-only NPZ subset used by model loading and restore auditing."""

    def __init__(self, path):
        self.path = path
        self.arrays = {}
        report = json.loads(REFERENCE.read_text())
        expected = {r['name']: r for r in report['model' if path.name == 'model.npz' else 'optimizer']}
        if path.name == 'optimizer.npz':
            for row in report['unused_converted_placeholders']:
                expected[row['name']] = dict(row, dtype='float32', sha256=hashlib.sha256(
                    np.zeros(row['shape'], dtype=np.float32).tobytes()).hexdigest())
        with zipfile.ZipFile(path) as archive, path.open('rb') as stream:
            for info in archive.infolist():
                if info.compress_type != zipfile.ZIP_STORED or info.flag_bits & 1 or not info.filename.endswith('.npy'):
                    raise ValueError('Only unencrypted stored NPY entries are supported')
                name = info.filename[:-4]
                stream.seek(info.header_offset)
                header = stream.read(30)
                if header[:4] != b'PK\x03\x04':
                    raise ValueError('Invalid local ZIP header')
                filename_size, extra_size = struct.unpack_from('<HH', header, 26)
                start = info.header_offset + 30 + filename_size + extra_size
                stream.seek(start)
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version == (2, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise ValueError('Unsupported NPY header version')
                offset = stream.tell()
                if dtype.hasobject or fortran or offset + int(np.prod(shape))*dtype.itemsize != start+info.file_size:
                    raise ValueError('Unexpected array layout/bounds')
                value = np.memmap(path, mode='r', offset=offset, dtype=dtype, shape=shape)
                row = expected.get(name)
                if row is None or list(shape) != row['shape'] or str(dtype) != row['dtype']:
                    raise ValueError('Array metadata differs from audited state: '+name)
                digest = hashlib.sha256(memoryview(value).cast('B')).hexdigest()
                if digest != row['sha256']:
                    raise ValueError('Array contents differ from audited state: '+name)
                self.arrays[name] = value
        if set(self.arrays) != set(expected):
            raise ValueError('Checkpoint archive coverage differs from audited state')
        self.files = list(self.arrays)
        print('[P12-MAPPED-REFERENCE]',path.name,len(self.files),'arrays SHA256 verified',flush=True)

    def __getitem__(self, key):
        return self.arrays[key]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def mapped_load(file, *args, **kwargs):
    path = Path(file) if isinstance(file, (str, Path)) else None
    if path in (ROOT/'model.npz', ROOT/'optimizer.npz'):
        if args or kwargs not in ({}, {'allow_pickle': False}):
            raise ValueError('Unexpected checkpoint load options')
        if path not in VALIDATED:
            VALIDATED[path] = MappedArchive(path)
        return VALIDATED[path]
    return ORIGINAL_LOAD(file, *args, **kwargs)


def install():
    np.load = mapped_load


if __name__ == '__main__':
    for filename in ['model.npz','optimizer.npz']:
        MappedArchive(ROOT/filename)
    print('All uncompressed checkpoint arrays matched original restored-state SHA256',flush=True)
