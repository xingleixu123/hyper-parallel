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

"""Fetch only bounded scalar/hash evidence; never transfer model, gradient or dataset values."""
import io
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parent
RUNS = ['hp-jtf637-weighted-dvm-preclip217', 'mf-preclip217',
        'hp-jtf637-weighted-dvm-forward217', 'mf-forward217',
        'hp-jtf637-weighted-dvm-attention217', 'mf-attention217',
        'hp-jtf637-packed217-absolute-positions226', 'hp-jtf637-packed217-global-mtp226',
        'hp-jtf637-packed217-both226', 'hp-jtf637-packed217-absolute-positions400',
        'hp-jtf637-packed217-global-mtp400', 'hp-jtf637-packed217-both400',
        'hp-jtf637-packed217-both-hash226', 'hp-jtf637-packed217-preclip219', 'mf-preclip219']


def main():
    remote = r"""from pathlib import Path
import sys, tarfile
root = Path('/home/xxl/jt-adamw-align-20261013/phase12')
runs = __RUNS__
paths = []
for run in runs:
    paths += [root / 'configs' / (run + '.yaml'), root / 'configs' / (run + '.json'), root / ('run-' + run + '.sh')]
    paths += [root / 'runs' / (run + suffix) for suffix in ['.exit', '.log', '.dispatch.log']]
    paths += [p for p in (root / 'runs' / run).glob('*') if p.suffix in ('.json', '.jsonl')]
    paths += [root / 'evidence' / (run + '-launch.json')]
for prefix in ['preclip217-', 'forward217-', 'attention217-', 'packed217-', 'preclip219-']:
    paths += list((root / 'evidence').glob(prefix + '*.json'))
    paths += list((root / 'evidence').glob(prefix + '*.log'))
with tarfile.open(fileobj=sys.stdout.buffer, mode='w|gz') as archive:
    for path in sorted(set(paths)):
        if path.is_file() and not path.is_symlink() and path.stat().st_size < 20_000_000:
            archive.add(path, arcname=str(path.relative_to(root)), recursive=False)
""".replace('__RUNS__', repr(RUNS))
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '125-root', 'python3', '-'],
                            input=remote.encode(), capture_output=True, timeout=45, check=True)
    count = 0
    with tarfile.open(fileobj=io.BytesIO(result.stdout), mode='r:gz') as archive:
        for member in archive:
            path = (ROOT / member.name).resolve()
            if not member.isfile() or not path.is_relative_to(ROOT):
                raise ValueError('Evidence archive escaped task root')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(archive.extractfile(member).read())
            count += 1
    print('Fetched', count, 'small evidence files;', len(result.stdout), 'compressed bytes')


if __name__ == '__main__':
    main()
