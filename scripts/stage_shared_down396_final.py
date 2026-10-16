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

"""Publish verified three-loss acceptance, preserving failed audits and archival source bytes."""
import ast
import hashlib
import json
from pathlib import Path

from stage_frontier258_public import copy
from sync_frontier217_evidence import RUNS

ROOT = Path(__file__).resolve().parent
PUBLIC = ROOT / '公开证据'


def main() -> None:
    """Require complete independent evidence and copy only bounded task-owned artifacts."""
    full = json.loads((ROOT / 'evidence/frontier396-shared-independent200-audit.json').read_text())
    observed = json.loads((ROOT / 'evidence/frontier396-shared-native-statistics-comparison.json').read_text())
    if not full['acceptance_complete'] or not all(full['fidelity'].values()):
        raise ValueError('Full200 acceptance incomplete')
    if not observed['complete'] or not all(all(v.values()) for v in observed['fidelity'].values()):
        raise ValueError('Independent observation incomplete')
    manifest_path = ROOT / 'evidence/frontier396-final-source-manifest.json'
    manifest = json.loads(manifest_path.read_text())
    for row in manifest['files']:
        source = ROOT / row['name']
        if hashlib.sha256(source.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('Source changed: ' + str(source))
        ast.parse(source.read_text())
        remote = row['remote'] or source.name
        if remote.startswith('tools-f637/'):
            target = 'scripts/latest-jt-f637/' + source.name
        elif remote.startswith('tools-e213/'):
            target = 'scripts/tools-e213/' + source.name
        elif remote.startswith('tools/shared-down-pr/'):
            target = 'scripts/shared-down-pr/' + source.name
        else:
            target = 'scripts/' + source.name
        row['public_path'] = target
        copy(source, PUBLIC / target)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    runs = RUNS[RUNS.index('hp-jte213-expertnorm328-state396'):]
    for run in runs:
        expected = '1' if run == 'hp-jte213-expertnorm328-state396' else '0'
        if (ROOT / 'runs' / (run + '.exit')).read_text().strip() != expected:
            raise ValueError('Unexpected terminal run status: ' + run)
        for suffix in ['.yaml', '.json']:
            copy(ROOT / 'configs' / (run + suffix), PUBLIC / 'configs' / (run + suffix))
        copy(ROOT / ('run-' + run + '.sh'), PUBLIC / 'launchers' / ('run-' + run + '.sh'))
        copy(ROOT / 'evidence' / (run + '-launch.json'), PUBLIC / 'evidence' / (run + '-launch.json'))
        for suffix in ['.exit', '.log', '.dispatch.log']:
            copy(ROOT / 'runs' / (run + suffix), PUBLIC / 'runs' / (run + suffix))
        for source in (ROOT / 'runs' / run).glob('*'):
            if source.suffix in ['.json', '.jsonl']:
                copy(source, PUBLIC / 'runs' / run / source.name)
    for source in (ROOT / 'evidence').glob('frontier396-*'):
        if source.is_file() and source.suffix in ['.json', '.log', '.ir', '.txt', '.csv', '.png', '.svg']:
            copy(source, PUBLIC / 'evidence' / source.name)
    for source in (ROOT / 'industry').glob('frontier396-current-*.py'):
        copy(source, PUBLIC / 'industry' / source.name)
    for name in ['completed-cann92-inventory.csv', 'completed-cann92-inventory.json']:
        copy(ROOT / 'evidence' / name, PUBLIC / 'evidence' / name)
    for name in ['阶段记录-MTPAux397边界定位.md', '阶段记录-396归一化正确性与行业对照.md', '总问题-三项loss达标.md']:
        copy(ROOT / name, PUBLIC / name)
    copy(ROOT / 'pr-shared-down-normalization/examples/torch/alignment/README.md',
         PUBLIC / 'scripts/shared-down-pr/README.md')
    copy(Path(__file__), PUBLIC / 'scripts' / Path(__file__).name)
    (PUBLIC / 'README.md').write_text('''# JT 单卡200→400：三项loss同时达标

第100组独立完整200步通过：总loss首步≤0.0005，原始MTP/Aux首步各≤0.001；三项每一步≤0.005。
最大误差分别为1.90735e-6、3.17891e-6、1.29733e-6，全部200步梯度范数精确一致。

- [最终完整报告、优先级、配置、正确性及清理](总问题-三项loss达标.md)
- [完整200步审计](evidence/frontier396-shared-independent200-audit.json)
- [独立原生观测保真与396/397边界](evidence/frontier396-shared-native-statistics-comparison.json)
- [单因素完整因果链](evidence/frontier396-causal-chain.json)
- [最新MF master PyNative、Megatron与FP64正确性对照](阶段记录-396归一化正确性与行业对照.md)
- [100组实验清单](evidence/completed-cann92-inventory.csv)
- [29份本阶段源码清单与执行范围](evidence/frontier396-final-source-manifest.json)
- [独立诊断程序](scripts/shared-down-pr/README.md)
- [历史总loss阶段](总问题-总loss达标.md)和[此前README](README-历史总loss阶段.md)

只包含可编辑报告、源码、配置、原生IR/统计、逐步指标和完整hash，不含用户数据或完整模型/梯度数组。
新归约因素用于复现指定旧MF图，不代表HP普通归约是数学错误；产品默认保持独立判断。
目标数组已清理，重新运行单步replay需要先按采集脚本再生，不能用哈希代替数组。
''', encoding='utf-8')
    print(json.dumps({'completed_runs': runs, 'sources': len(manifest['files']), 'accepted': True}))


if __name__ == '__main__':
    main()
