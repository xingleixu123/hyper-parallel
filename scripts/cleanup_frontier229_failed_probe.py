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

"""Retire only invalid diagnostic arrays after recording reproducible failure evidence."""
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path('/alignment/phase12')
STATE = Path('/state/jt-recipe-20261014').resolve()
TARGET = STATE / 'mf-frontier229-muon-fp32norm-replay'


def file_hash(path: Path) -> str:
    """Hash retained or retired evidence with bounded memory."""
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    """Check ownership, terminal failure and exact target paths before unlinking arrays."""
    run = 'frontier229-mf-muon-fp32norm-replay'
    receipt = ROOT / 'evidence/frontier229-invalid-probe-cleanup.json'
    if receipt.exists() or TARGET.is_symlink() or TARGET.resolve().parent != STATE:
        raise ValueError('Unexpected cleanup target or existing receipt')
    if (ROOT / 'runs' / (run + '.exit')).read_text().strip() != '1':
        raise ValueError('The diagnostic has not terminated with its recorded failure')
    failure_path = ROOT / 'evidence' / (run + '.json')
    failure = json.loads(failure_path.read_text())
    if any(row['observed_weight_exact_training'] for row in failure['records']):
        raise ValueError('Keep native-fidelity evidence')
    paths = sorted(path for path in TARGET.glob('*.npy') if path.stat().st_size > 1_000_000)
    if len(paths) != 7:
        raise ValueError('Unexpected invalid array inventory')
    rows = []
    for path in paths:
        info = path.stat()
        if path.is_symlink() or path.resolve().parent != TARGET or info.st_nlink != 1:
            raise ValueError('Cleanup cannot follow links or shared files')
        values = np.load(path, mmap_mode='r')
        row = {'name': path.name, 'bytes': info.st_size, 'allocated_bytes': info.st_blocks * 512,
               'sha256': file_hash(path), 'shape': list(values.shape), 'dtype': str(values.dtype),
               'strides': list(values.strides), 'uid': info.st_uid}
        del values
        rows.append(row)
    report = {'status': 'prepared', 'target': str(TARGET), 'files': rows,
              'time': datetime.now().astimezone().isoformat(), 'failure_report_sha256': file_hash(failure_path),
              'reason': 'Explicit FP32 normalized values changed the NS arithmetic and failed native state fidelity.',
              'retained': ['source', 'launch receipt', 'failure report', 'terminal log',
                           'generated observed module', 'four scalar norm/denominator arrays',
                           'report-path recovery receipt', 'valid native replay and actual training states']}
    receipt.write_text(json.dumps(report, indent=2))
    deleted = []
    for path, row in zip(paths, rows):
        if path.stat().st_size != row['bytes'] or file_hash(path) != row['sha256']:
            raise ValueError('Array changed after cleanup audit')
        path.unlink()
        deleted.append(row['name'])
        report['deleted'] = list(deleted)
        receipt.write_text(json.dumps(report, indent=2))
    report.update(status='complete', deleted_logical_bytes=sum(row['bytes'] for row in rows),
                  freed_allocated_bytes=sum(row['allocated_bytes'] for row in rows))
    receipt.write_text(json.dumps(report, indent=2))
    print(json.dumps({key: report[key] for key in ['status', 'deleted_logical_bytes', 'freed_allocated_bytes']}))


if __name__ == '__main__':
    main()
