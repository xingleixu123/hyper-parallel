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

"""Stage completed293/306 evidence and prepared330 sources without live training logs."""
import hashlib
import json
from pathlib import Path

from stage_frontier258_public import copy

ROOT = Path(__file__).resolve().parent
PUBLIC = ROOT / '公开证据'


def main() -> None:
    """Require completed runs and unchanged source bytes."""
    for name in ['frontier293-306-source-manifest.json', 'frontier330-source-manifest.json']:
        manifest = json.loads((ROOT / 'evidence' / name).read_text())
        if not manifest['complete'] or not manifest['local_remote_exact'] or not manifest['ast_valid']:
            raise ValueError('Source manifest incomplete')
        for row in manifest['files']:
            source = ROOT / row['local_path']
            if hashlib.sha256(source.read_bytes()).hexdigest() != row['sha256']:
                raise ValueError('Source changed: ' + row['local_path'])
            copy(source, PUBLIC / row['public_path'])
        copy(ROOT / 'evidence' / name, PUBLIC / 'evidence' / name)
    runs = ['hp-jte213-fc1norm400', 'hp-jte213-fc1norm-preclip294',
            'hp-jte213-fc1norm-preclip307', 'mf-native-stat307',
            'frontier293-normalization-precision', 'hp-jte213-fc1norm-qupboth400',
            'hp-jte213-fc1norm-qupboth-preclip307', 'hp-jte213-fc1norm-state306',
            'mf-frontier306-state', 'frontier306-hp-control-replay',
            'frontier306-hp-both-replay', 'frontier306-mf-builtin-stat']
    for run in runs:
        if (ROOT / 'runs' / (run + '.exit')).read_text().strip() != '0':
            raise ValueError('Run is not complete: ' + run)
        for suffix in ['.yaml', '.json']:
            copy(ROOT / 'configs' / (run + suffix), PUBLIC / 'configs' / (run + suffix))
        copy(ROOT / ('run-' + run + '.sh'), PUBLIC / 'launchers' / ('run-' + run + '.sh'))
        copy(ROOT / 'evidence' / (run + '-launch.json'), PUBLIC / 'evidence' / (run + '-launch.json'))
        for suffix in ['.exit', '.log', '.dispatch.log']:
            copy(ROOT / 'runs' / (run + suffix), PUBLIC / 'runs' / (run + suffix))
        for source in (ROOT / 'runs' / run).glob('*'):
            if source.suffix in ['.json', '.jsonl']:
                copy(source, PUBLIC / 'runs' / run / source.name)
    for prefix in ['frontier293-', 'frontier306-', 'frontier307-', 'jt-e213-source-archive-cleanup']:
        for source in (ROOT / 'evidence').glob(prefix + '*'):
            if source.suffix in ['.json', '.log', '.txt', '.ir', '.csv', '.png', '.svg']:
                copy(source, PUBLIC / 'evidence' / source.name)
    for source in (ROOT / 'evidence/frontier306-obsolete-anchor357-cleanup').rglob('*.json'):
        copy(source, PUBLIC / source.relative_to(ROOT))
    for suffix in ['json', 'csv']:
        name = 'completed-cann92-inventory.' + suffix
        copy(ROOT / 'evidence' / name, PUBLIC / 'evidence' / name)
    for name in ['阶段记录-293DenseFC1归一化与306边界.md', '阶段记录-306QupRoPE条件消融与330边界.md',
                 '总问题-258阶段原文归档.md']:
        copy(ROOT / name, PUBLIC / name)
    for name in ['stage_frontier293_306_public.py', 'sync_frontier217_evidence.py',
                 'plot_fc1_norm293_full200.py', 'plot_frontier306_full200.py']:
        copy(ROOT / name, PUBLIC / 'scripts' / name)
    print('Completed97/98 and native306/307 evidence staged;330 training logs excluded')


if __name__ == '__main__':
    main()
