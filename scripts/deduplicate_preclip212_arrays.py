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

"""Deduplicate equal FP32 payloads and retain original NPY headers for lossless reconstruction."""
import ast
import struct
import datetime
import hashlib
import json
import os
from pathlib import Path

ROOT = Path('/home/xxl/jt-adamw-align-20261013/phase12')
STATE = Path('/dev/shm/jt-adamw-align-20261013/state/jt-recipe-20261014')
HP = 'hp-jtf637-norm208-preclip212'
MF = 'mf-preclip212'


def file_sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()



def array_info(path, expected):
    with path.open('rb') as stream:
        magic = stream.read(8)
        if magic[:6] != b'\x93NUMPY' or magic[6] not in (1,2):
            raise ValueError('Unexpected NPY header format')
        size_bytes = stream.read(2 if magic[6] == 1 else 4)
        size = struct.unpack('<H' if magic[6] == 1 else '<I', size_bytes)[0]
        header = stream.read(size)
        fields = ast.literal_eval(header.decode('latin1').strip())
        if fields['descr'] != '<f4' or fields['fortran_order'] or list(fields['shape']) != expected['shape']:
            raise ValueError('Unexpected array representation')
        digest = hashlib.sha256()
        payload_size = 0
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            payload_size += len(chunk)
            digest.update(chunk)
    if payload_size != expected['array_bytes'] or digest.hexdigest() != expected['sha256']:
        raise ValueError('Complete gradient payload changed')
    return {'original_header_hex': (magic+size_bytes+header).hex(), 'payload_sha256':digest.hexdigest()}


def main():
    report = json.loads((ROOT / 'evidence/preclip212-comparison.json').read_text())
    if not report['complete'] or not report['usable_as_native_training']:
        raise ValueError('Array comparison did not pass fidelity')
    destination = ROOT / 'evidence/preclip212-array-deduplication.json'
    if destination.exists():
        raise ValueError('Deduplication receipt already exists')
    for run in [HP, MF]:
        if (ROOT / 'runs' / (run + '.exit')).read_text().strip() != '0':
            raise ValueError('Source observer is still active or failed')
    records = {run: {row['name']: row for row in json.loads(
        (ROOT / 'runs' / run / 'PRECLIP-212.json').read_text())['records']} for run in [HP, MF]}
    candidates, skipped = [], []
    for name, row in records[HP].items():
        reference = records[MF][name]
        if row['sha256'] != reference['sha256']:
            continue
        hp = STATE / HP / row['file']
        mf = STATE / MF / 'gradients' / reference['file']
        for path, expected in [(hp, STATE / HP), (mf, STATE / MF / 'gradients')]:
            if path.is_symlink() or path.resolve().parent != expected.resolve():
                raise ValueError('Path escaped its verified task-owned gradient folder')
        hp_stat, mf_stat = hp.stat(), mf.stat()
        if hp_stat.st_dev == mf_stat.st_dev and hp_stat.st_ino == mf_stat.st_ino:
            continue
        if hp_stat.st_dev != mf_stat.st_dev or hp_stat.st_nlink != 1:
            raise ValueError('Unexpected filesystem or already-shared destination inode')
        hp_info, mf_info = array_info(hp, row), array_info(mf, reference)
        if hp_info['payload_sha256'] != mf_info['payload_sha256']:
            raise ValueError('Payloads are not identical')
        candidates.append({'name': name, 'path': str(hp), 'canonical': str(mf), 'original_header_hex': hp_info['original_header_hex'], 'canonical_header_hex': mf_info['original_header_hex'],
                           'header_changed': hp_info['original_header_hex'] != mf_info['original_header_hex'],
                           'array_sha256': row['sha256'], 'physical_bytes': hp_stat.st_blocks * 512,
                           'old_inode': [hp_stat.st_dev, hp_stat.st_ino],
                           'canonical_inode': [mf_stat.st_dev, mf_stat.st_ino]})
    protected = {tuple(row[key]) for row in candidates for key in ['old_inode', 'canonical_inode']}
    open_files = []
    for process in Path('/proc').iterdir():
        if not process.name.isdigit():
            continue
        try:
            for descriptor in (process / 'fd').iterdir():
                try:
                    stat = descriptor.stat()
                    if (stat.st_dev, stat.st_ino) in protected:
                        open_files.append(str(descriptor))
                except FileNotFoundError:
                    pass
        except (FileNotFoundError, ProcessLookupError):
            pass
    if open_files:
        raise ValueError('Another process still holds candidate arrays: ' + str(open_files))
    for row in candidates:
        target, canonical = Path(row['path']), Path(row['canonical'])
        if [target.stat().st_dev, target.stat().st_ino] != row['old_inode']:
            raise ValueError('Destination changed after validation')
        temporary = target.with_name(target.name + '.verified-link')
        if temporary.exists():
            raise ValueError('Unexpected temporary link')
        os.link(canonical, temporary)
        os.replace(temporary, target)
        if target.stat().st_ino != canonical.stat().st_ino:
            raise ValueError('Hardlink replacement was not completed')
    receipt = {'completed_at': datetime.datetime.now().astimezone().isoformat(), 'complete': True,
               'files': candidates, 'skipped': skipped,
               'physical_bytes_freed': sum(row['physical_bytes'] for row in candidates),
               'scope': 'Complete FP32 payloads preserved; paths remain readable. NPY header may change; reconstruct original header bytes from original_header_hex plus canonical payload when exact original file bytes are needed. Completed observers and host open-FD scan required.'}
    destination.write_text(json.dumps(receipt, indent=2))
    print(json.dumps({'linked_files': len(candidates), 'skipped': len(skipped),
                      'physical_bytes_freed': receipt['physical_bytes_freed']}))


if __name__ == '__main__':
    main()
