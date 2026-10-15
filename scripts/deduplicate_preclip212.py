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

"""Deduplicate only byte-identical private preclip212 arrays after completed audits."""
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


def main():
    report = json.loads((ROOT / 'evidence/preclip212-comparison.json').read_text())
    if not report['complete'] or not report['usable_as_native_training']:
        raise ValueError('Array comparison did not pass fidelity')
    destination = ROOT / 'evidence/preclip212-deduplication.json'
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
        if hp_stat.st_dev != mf_stat.st_dev or hp_stat.st_nlink != 1:
            raise ValueError('Unexpected filesystem or already-shared destination inode')
        hp_sha, mf_sha = file_sha(hp), file_sha(mf)
        if hp_sha != mf_sha:
            skipped.append({'name': name, 'reason': 'NPY bytes differ; arrays were equal; originals retained'})
            continue
        candidates.append({'name': name, 'path': str(hp), 'canonical': str(mf), 'file_sha256': hp_sha,
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
               'scope': 'Byte-identical files only, completed observers, open-FD scan; paths and bytes preserved.'}
    destination.write_text(json.dumps(receipt, indent=2))
    print(json.dumps({'linked_files': len(candidates), 'skipped': len(skipped),
                      'physical_bytes_freed': receipt['physical_bytes_freed']}))


if __name__ == '__main__':
    main()
