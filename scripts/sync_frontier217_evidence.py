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
import sys
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parent
RUNS = ['hp-jtf637-weighted-dvm-preclip217', 'mf-preclip217', 'hp-jtf637-weighted-dvm-forward217', 'mf-forward217', 'hp-jtf637-weighted-dvm-attention217', 'mf-attention217', 'hp-jtf637-packed217-absolute-positions226', 'hp-jtf637-packed217-global-mtp226', 'hp-jtf637-packed217-both226', 'hp-jtf637-packed217-absolute-positions400', 'hp-jtf637-packed217-global-mtp400', 'hp-jtf637-packed217-both400', 'hp-jtf637-packed217-both-hash226', 'hp-jtf637-packed217-preclip219', 'mf-preclip219', 'hp-jtf637-packed217-state219', 'mf-frontier219-state', 'frontier219-hp-muon-replay', 'frontier219-mf-muon-replay', 'frontier219-expert-denominator-replay', 'frontier219-dense-npusqrt-replay', 'mf-frontier219-ns-train', 'hp-jtf637-norm219-dense226', 'hp-jtf637-norm219-expert226', 'hp-jtf637-norm219-both226', 'hp-jtf637-norm219-dense400', 'hp-jtf637-norm219-expert400', 'hp-jtf637-norm219-both400', 'hp-jtf637-norm219-state222', 'mf-frontier222-state', 'frontier222-q-control-replay', 'frontier222-q-npu-replay', 'hp-jtf637-qsqrt222-both226', 'hp-jtf637-qsqrt222-expert226', 'hp-jtf637-qsqrt222-both400', 'hp-jtf637-qsqrt222-expert400', 'hp-jtf637-qsqrt222-preclip226', 'mf-preclip226', 'hp-jtf637-qsqrt222-forward226', 'mf-forward226', 'hp-jtf637-qsqrt222-backward226', 'mf-backward226', 'hp-jtf637-qsqrt222-head226', 'mf-head226', 'hp-jtf637-qsqrt222-head226-r2', 'hp-jtf637-qsqrt222-head226-r3', 'head226-gradient-cause-replay', 'hp-jtf637-qsqrt222-ce226', 'mf-ce226', 'mf-ce226-kernel', 'ce226-reduction-replay', 'ce226-reduction-replay-r2', 'ce226-reduction-replay-r3', 'ce226-row-positions', 'ce226-tail-holdout', 'hp-jtf637-ce-tail226', 'hp-jtf637-ce-tail400', 'hp-jtf637-ce-tail230', 'mf-modelhash230', 'hp-jtf637-ce-tail-state229', 'mf-frontier229-state', 'frontier229-hp-muon-replay', 'frontier229-mf-muon-replay', 'frontier229-mf-muon-fp32norm-replay', 'frontier229-mf-muon-fp32norm-cast-replay', 'frontier229-norm-expression-screen', 'frontier229-norm-epsilon-screen', 'frontier229-hp-epsilon-replay', 'hp-jtf637-epsilon230', 'hp-jtf637-epsilon400', 'frontier229-mf-builtin-stat-dump', 'frontier229-mf-builtin-tensor-dump', 'hp-jtf637-epsilon235', 'mf-modelhash235', 'frontier235-mf-stat-hash-probe', 'hp-jtf637-epsilon-preclip235', 'mf-native-stat235', 'frontier235-global-sqrt-replay', 'hp-jtf637-global-sqrt235', 'hp-jtf637-global-sqrt400', 'hp-jtf637-epsilon-preclip235-zero', 'hp-jtf637-epsilon-preclip235-nativehash', 'hp-jtf637-global-sqrt-preclip236', 'mf-native-stat236', 'hp-jtf637-global-sqrt-boundaries236', 'mf-native-boundaries236', 'hp-jtf637-global-sqrt-boundaries236-r2', 'hp-jtf637-global-sqrt-moe236', 'hp-jtf637-weighted-mtp236', 'hp-jtf637-weighted-mtp400', 'hp-jtf637-weighted-precision236', 'hp-jtf637-weighted-preclip259', 'mf-native-stat259', 'hp-jtf637-weighted-state258', 'mf-frontier258-state', 'frontier258-hp-muon-replay', 'frontier258-mf-muon-replay', 'frontier258-mf-builtin-tensor-dump', 'frontier258-norm-expression-screen', 'frontier258-square-sum-screen', 'frontier258-mf-builtin-dvm-stat', 'frontier258-mf-builtin-dvm-stat-retry1', 'frontier258-hp-q-up-replay', 'frontier258-hp-norm-holdout', 'frontier258-mf-norm-holdout', 'frontier258-mf-norm-holdout-hash', 'hp-jtf637-qup-nope400', 'hp-jtf637-qup-both400', 'hp-jtf637-qup-preclip259', 'hp-jtf637-qup-preclip294', 'mf-native-stat294', 'hp-jtf637-qup-state293', 'mf-frontier293-state', 'hp-jtf637-qup-state293-r2', 'hp-jte213-qup-nope400', 'jt-e213-pr192-setup', 'hp-jtf637-qup-state293-r3', 'frontier293-mf-builtin-stat', 'frontier293-hp-muon-replay', 'frontier293-hp-fc1-replay', 'frontier293-hp-norm-holdout', 'frontier293-mf-norm-holdout-hash', 'hp-jte213-fc1norm400', 'hp-jte213-fc1norm-preclip294']


def main():
    recent = sys.argv[1:] == ['--recent']
    runs = RUNS[RUNS.index('hp-jtf637-ce-tail226'):] if recent else RUNS
    prefixes = ['ce226-', 'frontier229-', 'frontier235-', 'frontier236-', 'frontier259-', 'frontier258-', 'frontier294-', 'frontier293-', 'jt-e213-'] if recent else ['preclip217-', 'forward217-', 'attention217-', 'packed217-', 'preclip219-', 'frontier219-', 'norm219-', 'frontier222-', 'qsqrt222-', 'preclip226-', 'frontier226-', 'forward226-', 'backward226-', 'head226-', 'ce226-', 'frontier229-', 'frontier235-', 'frontier236-', 'frontier259-', 'frontier258-', 'frontier294-', 'frontier293-', 'jt-e213-']
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
for prefix in __PREFIXES__:
    paths += list((root / 'evidence').glob(prefix + '*.json'))
    paths += list((root / 'evidence').glob(prefix + '*.log'))
    for suffix in ['.txt', '.ir', '.csv']:
        paths += list((root / 'evidence').glob(prefix + '*' + suffix))
with tarfile.open(fileobj=sys.stdout.buffer, mode='w|gz') as archive:
    for path in sorted(set(paths)):
        if path.is_file() and not path.is_symlink() and path.stat().st_size < 20_000_000:
            archive.add(path, arcname=str(path.relative_to(root)), recursive=False)
""".replace('__RUNS__', repr(runs)).replace('__PREFIXES__', repr(prefixes))
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
