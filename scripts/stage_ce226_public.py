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
from stage_frontier226_public import copy

root = Path(__file__).resolve().parent
public = root / '公开证据'
runs = ['hp-jtf637-qsqrt222-ce226', 'mf-ce226', 'mf-ce226-kernel', 'ce226-reduction-replay', 'ce226-reduction-replay-r2', 'ce226-reduction-replay-r3', 'ce226-row-positions', 'ce226-tail-holdout', 'hp-jtf637-ce-tail226', 'hp-jtf637-ce-tail400', 'hp-jtf637-ce-tail230', 'mf-modelhash230', 'hp-jtf637-ce-tail-state229', 'mf-frontier229-state', 'frontier229-hp-muon-replay', 'frontier229-mf-muon-replay', 'frontier229-mf-muon-fp32norm-replay', 'frontier229-mf-muon-fp32norm-cast-replay', 'frontier229-norm-expression-screen', 'frontier229-norm-epsilon-screen', 'frontier229-hp-epsilon-replay', 'hp-jtf637-epsilon230', 'hp-jtf637-epsilon400', 'frontier229-mf-builtin-stat-dump', 'frontier229-mf-builtin-tensor-dump', 'hp-jtf637-epsilon235', 'mf-modelhash235', 'frontier235-mf-stat-hash-probe', 'hp-jtf637-epsilon-preclip235', 'mf-native-stat235', 'frontier235-global-sqrt-replay', 'hp-jtf637-global-sqrt235', 'hp-jtf637-global-sqrt400', 'hp-jtf637-epsilon-preclip235-zero', 'hp-jtf637-epsilon-preclip235-nativehash', 'hp-jtf637-global-sqrt-preclip236', 'mf-native-stat236', 'hp-jtf637-global-sqrt-boundaries236', 'mf-native-boundaries236', 'hp-jtf637-global-sqrt-boundaries236-r2', 'hp-jtf637-global-sqrt-moe236', 'hp-jtf637-weighted-mtp236', 'hp-jtf637-weighted-mtp400', 'hp-jtf637-weighted-precision236', 'hp-jtf637-weighted-preclip259', 'mf-native-stat259']
manifest = json.loads((root / 'evidence/ce226-source-manifest.json').read_text())
manifest['files'].extend(json.loads((root / 'evidence/frontier229-source-manifest.json').read_text())['files'])
manifest['files'].extend(json.loads((root / 'evidence/frontier235-source-manifest.json').read_text())['files'])
manifest['files'].extend(json.loads((root / 'evidence/frontier236-source-manifest.json').read_text())['files'])
manifest['files'].extend(json.loads((root / 'evidence/frontier259-source-manifest.json').read_text())['files'])
for row in manifest['files']:
    source = root / row['name']
    if hashlib.sha256(source.read_bytes()).hexdigest() != row['sha256']:
        raise ValueError('Source changed: ' + row['name'])
    folder = 'scripts/latest-jt-f637' if row['folder'] == 'tools-f637' else 'scripts'
    copy(source, public / folder / row['name'])
for run in runs:
    for suffix in ['.yaml', '.json']:
        copy(root / 'configs' / (run + suffix), public / 'configs' / (run + suffix))
    copy(root / ('run-' + run + '.sh'), public / 'launchers' / ('run-' + run + '.sh'))
    copy(root / 'evidence' / (run + '-launch.json'), public / 'evidence' / (run + '-launch.json'))
    if not (root / 'runs' / (run + '.exit')).exists():
        continue
    for suffix in ['.exit', '.log', '.dispatch.log']:
        copy(root / 'runs' / (run + suffix), public / 'runs' / (run + suffix))
    for source in (root / 'runs' / run).glob('*'):
        if source.suffix in ['.json', '.jsonl']:
            copy(source, public / 'runs' / run / source.name)
for source in [*(root / 'evidence').glob('ce226-*'), *(root / 'evidence').glob('frontier229-*'), *(root / 'evidence').glob('frontier235-*'), *(root / 'evidence').glob('frontier236-*'), *(root / 'evidence').glob('frontier259-*')]:
    if source.suffix in ['.json', '.log', '.txt', '.ir', '.csv', '.png', '.svg'] and 'parent-before' not in source.name:
        copy(source, public / 'evidence' / source.name)
for name in ['ce226-training-dvm-ce-excerpts.txt', 'ce226-replay-dvm-kernel.txt']:
    copy(root / 'evidence' / name, public / 'evidence' / name)
for name in ['stage_ce226_public.py', 'sync_frontier217_evidence.py']:
    copy(root / name, public / 'scripts' / name)
for suffix in ['json', 'csv']:
    copy(root / ('evidence/completed-cann92-inventory.' + suffix),
         public / ('evidence/completed-cann92-inventory.' + suffix))
copy(root / '阶段记录-226交叉熵求和定位.md', public / '阶段记录-226交叉熵求和定位.md')
print('CE observations, independent holdout and source evidence staged; no tensor values.')
copy(root / '阶段记录-229Muon归一化与原生Dump.md', public / '阶段记录-229Muon归一化与原生Dump.md')

copy(root / '阶段记录-235全局范数与统计哈希.md', public / '阶段记录-235全局范数与统计哈希.md')

copy(root / '阶段记录-236MTP加权归约与分块消融.md', public / '阶段记录-236MTP加权归约与分块消融.md')

copy(root / 'plot_weighted_mtp236_full200.py', public / 'scripts/plot_weighted_mtp236_full200.py')

copy(root / '总问题-235阶段原文归档.md', public / '总问题-235阶段原文归档.md')
