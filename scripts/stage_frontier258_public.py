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

"""Stage verified Q-up evidence without live logs or full tensor values."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
PUBLIC = ROOT / '公开证据'


def copy(source: Path, target: Path) -> None:
    """Copy only bounded ordinary artifacts inside the evidence checkout."""
    if not source.exists():
        return
    if not source.is_file() or source.is_symlink() or source.stat().st_size >= 20_000_000:
        raise ValueError('Unexpected artifact size or type: ' + str(source))
    if not target.resolve().is_relative_to(PUBLIC.resolve()):
        raise ValueError('Publication path escaped the evidence checkout')
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def main() -> None:
    """Require source digests and finished-run boundaries before staging."""
    for filename in ['frontier258-source-manifest.json', 'frontier294-source-manifest.json', 'frontier293-source-manifest.json']:
        for row in json.loads((ROOT / 'evidence' / filename).read_text())['files']:
            source = ROOT / row.get('local_path', row.get('name'))
            if hashlib.sha256(source.read_bytes()).hexdigest() != row['sha256']:
                raise ValueError('Source changed: ' + str(source))
            if 'public_path' in row:
                target = PUBLIC / row['public_path']
            else:
                folder = 'scripts/latest-jt-f637' if row['folder'] == 'tools-f637' else 'scripts'
                target = PUBLIC / folder / row['name']
            copy(source, target)
    runs = ['hp-jtf637-weighted-state258', 'mf-frontier258-state',
            'frontier258-hp-muon-replay', 'frontier258-mf-muon-replay',
            'frontier258-mf-builtin-tensor-dump', 'frontier258-norm-expression-screen',
            'frontier258-square-sum-screen', 'frontier258-mf-builtin-dvm-stat',
            'frontier258-mf-builtin-dvm-stat-retry1', 'frontier258-hp-q-up-replay',
            'frontier258-hp-norm-holdout', 'frontier258-mf-norm-holdout',
            'frontier258-mf-norm-holdout-hash', 'hp-jtf637-qup-nope400',
            'hp-jtf637-qup-both400', 'hp-jtf637-qup-preclip259',
            'hp-jtf637-qup-preclip294', 'mf-native-stat294',
            'hp-jtf637-qup-state293', 'hp-jtf637-qup-state293-r2', 'hp-jtf637-qup-state293-r3',
            'mf-frontier293-state', 'frontier293-mf-builtin-stat', 'frontier293-hp-muon-replay',
            'frontier293-hp-fc1-replay', 'frontier293-hp-norm-holdout',
            'frontier293-mf-norm-holdout-hash', 'hp-jte213-qup-nope400', 'jt-e213-pr192-setup']
    for run in runs:
        if not (ROOT / 'runs' / (run + '.exit')).exists():
            failure = ROOT / 'evidence/frontier258-dvm-launch-escape-error.json'
            if run != 'frontier258-mf-builtin-dvm-stat' or not failure.exists():
                raise ValueError('Run is not finished: ' + run)
            if json.loads(failure.read_text())['run'] != run:
                raise ValueError('Failed launch record mismatch')
        for suffix in ['.yaml', '.json']:
            copy(ROOT / 'configs' / (run + suffix), PUBLIC / 'configs' / (run + suffix))
        copy(ROOT / ('run-' + run + '.sh'), PUBLIC / 'launchers' / ('run-' + run + '.sh'))
        copy(ROOT / 'evidence' / (run + '-launch.json'), PUBLIC / 'evidence' / (run + '-launch.json'))
        for suffix in ['.exit', '.log', '.dispatch.log']:
            copy(ROOT / 'runs' / (run + suffix), PUBLIC / 'runs' / (run + suffix))
        for source in (ROOT / 'runs' / run).glob('*'):
            if source.suffix in ['.json', '.jsonl']:
                copy(source, PUBLIC / 'runs' / run / source.name)
    for prefix in ['frontier258-', 'frontier294-', 'frontier293-', 'jt-e213-']:
        for source in (ROOT / 'evidence').glob(prefix + '*'):
            if source.suffix in ['.json', '.log', '.txt', '.ir', '.csv', '.png', '.svg']:
                if 'parent-before' not in source.name and 'frozen-runtime-guard' not in source.name:
                    copy(source, PUBLIC / 'evidence' / source.name)
    for name in ['stage_frontier258_public.py', 'sync_frontier217_evidence.py',
                 'plot_q_up258_full200.py']:
        copy(ROOT / name, PUBLIC / 'scripts' / name)
    for suffix in ['json', 'csv']:
        name = 'completed-cann92-inventory.' + suffix
        copy(ROOT / 'evidence' / name, PUBLIC / 'evidence' / name)
    for name in ['阶段记录-258Qup平方和归约与独立消融.md', '总问题-236阶段原文归档.md']:
        copy(ROOT / name, PUBLIC / name)
    print('Verified bounded Q-up and frontier294 evidence staged')


if __name__ == '__main__':
    main()
