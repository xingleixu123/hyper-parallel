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

"""Stage bounded CE-tail evidence, excluding live logs and private tensor values."""
import hashlib
import json
from pathlib import Path
from stage_frontier226_public import copy, sanitize

root = Path(__file__).resolve().parent
public = root / '公开证据'
runs = ['hp-jtf637-qsqrt222-ce226', 'mf-ce226', 'mf-ce226-kernel', 'ce226-reduction-replay',
        'ce226-reduction-replay-r2', 'ce226-reduction-replay-r3', 'ce226-row-positions',
        'ce226-tail-holdout', 'hp-jtf637-ce-tail226', 'hp-jtf637-ce-tail400']
manifest = json.loads((root / 'evidence/ce226-source-manifest.json').read_text())
for row in manifest['files']:
    source = root / row['name']
    if hashlib.sha256(source.read_bytes()).hexdigest() != row['sha256']:
        raise ValueError('Source changed: ' + row['name'])
    folder = 'scripts/latest-jt-f637' if row['folder'] == 'tools-f637' else 'scripts'
    copy(source, public / folder / row['name'])
for run in runs:
    copy(root / 'configs' / (run + '.yaml'), public / 'configs' / (run + '.yaml'))
    copy(root / ('run-' + run + '.sh'), public / 'launchers' / ('run-' + run + '.sh'))
    copy(root / 'evidence' / (run + '-launch.json'), public / 'evidence' / (run + '-launch.json'))
    if not (root / 'runs' / (run + '.exit')).exists():
        continue
    for suffix in ['.exit', '.log', '.dispatch.log']:
        copy(root / 'runs' / (run + suffix), public / 'runs' / (run + suffix))
    for source in (root / 'runs' / run).glob('*'):
        if source.suffix in ['.json', '.jsonl']:
            copy(source, public / 'runs' / run / source.name)
for source in (root / 'evidence').glob('ce226-*'):
    if source.suffix in ['.json', '.log'] and 'parent-before' not in source.name:
        copy(source, public / 'evidence' / source.name)
for name in ['ce226-training-dvm-ce-excerpts.txt', 'ce226-replay-dvm-kernel.txt']:
    copy(root / 'evidence' / name, public / 'evidence' / name)
for name in ['stage_ce226_public.py', 'sync_frontier217_evidence.py']:
    copy(root / name, public / 'scripts' / name)
copy(root / '阶段记录-226交叉熵求和定位.md', public / '阶段记录-226交叉熵求和定位.md')
print('CE observations, independent holdout and source evidence staged; no tensor values.')
