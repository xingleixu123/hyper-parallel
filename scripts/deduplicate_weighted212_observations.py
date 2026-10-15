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

"""Hardlink byte-identical completed212 observations, preserving every path and byte."""
import datetime
import hashlib
import json
import os
from pathlib import Path

from deduplicate_preclip212 import file_sha

ROOT = Path('/home/xxl/jt-adamw-align-20261013/phase12')
STATE = Path('/dev/shm/jt-adamw-align-20261013/state/jt-recipe-20261014')
RUNS = ['hp-jtf637-norm208-moebranches212', 'mf-moebranches212',
        'hp-jtf637-norm208-routerscores212', 'mf-routerscores212',
        'hp-jtf637-norm208-weighted212', 'mf-weighted212']


def main() -> None:
    destination = ROOT / 'evidence/weighted212-observation-deduplication.json'
    if destination.exists():
        raise ValueError('Cleanup receipt already exists')
    for report_name in ['moe212-branch-comparison.json', 'router212-scores-comparison.json',
                        'weighted212-operands-comparison.json']:
        report = json.loads((ROOT / 'evidence' / report_name).read_text())
        if not report['complete'] or not all(report['fidelity'].values()):
            raise ValueError('Required completed fidelity audit failed')
    paths = []
    for run in RUNS:
        if (ROOT / 'runs' / (run + '.exit')).read_text().strip() != '0':
            raise ValueError('Observation still active')
        directory = ROOT / 'runs' / run
        hp = run.startswith('hp-')
        if hp:
            branches = json.loads((directory / 'MOE_BRANCHES.json').read_text())['records']
        else:
            branches = next(json.loads(line)['records'] for line in (
                directory / 'MOE_BRANCHES.jsonl').read_text().splitlines() if json.loads(line)['step'] == 212)
        records = [(row, 'branches' if hp else 'deep') for row in branches]
        for report_name, folder in [('ROUTER_SCORES-212.json', 'scores' if hp else 'deep'),
                                    ('WEIGHTED_PRODUCT-212.json', 'weighted' if hp else 'deep')]:
            if (directory / report_name).exists():
                records.extend((row, folder) for row in json.loads((directory / report_name).read_text())['records'])
        for row, folder in records:
            path = STATE / run / folder / row['file']
            if path.is_symlink() or path.resolve().parent != (STATE / run / folder).resolve():
                raise ValueError('Unexpected observation path')
            if path not in paths:
                paths.append(path)
    canonical, candidates = {}, []
    for path in paths:
        info = path.stat()
        digest = file_sha(path)
        key = (info.st_size, digest)
        if key not in canonical:
            canonical[key] = path
            continue
        source = canonical[key]
        original = source.stat()
        if info.st_ino == original.st_ino and info.st_dev == original.st_dev:
            continue
        if info.st_nlink != 1 or info.st_dev != original.st_dev:
            raise ValueError('Unexpected shared destination or filesystem')
        candidates.append({'path': str(path), 'canonical': str(source), 'file_sha256': digest,
                           'physical_bytes': info.st_blocks * 512,
                           'old_inode': [info.st_dev, info.st_ino],
                           'canonical_inode': [original.st_dev, original.st_ino]})
    protected = {tuple(row[key]) for row in candidates for key in ['old_inode', 'canonical_inode']}
    for process in Path('/proc').iterdir():
        if not process.name.isdigit():
            continue
        try:
            for descriptor in (process / 'fd').iterdir():
                try:
                    info = descriptor.stat()
                    if (info.st_dev, info.st_ino) in protected:
                        raise ValueError('An active process holds candidate data: ' + str(descriptor))
                except FileNotFoundError:
                    pass
        except (FileNotFoundError, ProcessLookupError):
            pass
    receipt = {'complete': False, 'files': candidates, 'scanned_files': len(paths),
               'planned_at': datetime.datetime.now().astimezone().isoformat()}
    destination.write_text(json.dumps(receipt, indent=2))
    for row in candidates:
        target, source = Path(row['path']), Path(row['canonical'])
        if [target.stat().st_dev, target.stat().st_ino] != row['old_inode']:
            raise ValueError('Destination changed after verification')
        temporary = target.with_name(target.name + '.verified-link')
        if temporary.exists():
            raise ValueError('Unexpected temporary link')
        os.link(source, temporary)
        os.replace(temporary, target)
        if target.stat().st_ino != source.stat().st_ino:
            raise ValueError('Hardlink did not complete')
    receipt.update(complete=True, completed_at=datetime.datetime.now().astimezone().isoformat(),
                   physical_bytes_freed=sum(row['physical_bytes'] for row in candidates),
                   scope='Completed task-owned arrays only; full file SHA equality, open-FD scan; bytes and paths unchanged.')
    destination.write_text(json.dumps(receipt, indent=2))
    print(json.dumps({'complete': True, 'linked_files': len(candidates),
                      'physical_bytes_freed': receipt['physical_bytes_freed']}))


if __name__ == '__main__':
    main()
