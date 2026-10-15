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

"""Locate the first changed selected head operand on both real six-step trajectories."""
import json
from pathlib import Path
import numpy as np
from compare_head_selected_operands import stats
from compare_vocab_optimizer_trace import fidelity, read_rows
ROOT=Path('/alignment/phase12')
MF='mf-head-selected-operands206'
HP='hp-jtf637-adamw-all-headcontig206-r4'
STATE=Path('/state/jt-recipe-20261014')/MF/'deep'


def main():
    checks={label+'_'+key:value for label,run,base in [('mf',MF,'mf-resume'),('hp',HP,'hp-jtf637-adamw-all-archive206')]
            for key,value in fidelity(run,base,6).items()}
    if not all(checks.values()):raise ValueError(checks)
    rows=read_rows(ROOT/'runs'/MF/'HEAD_OPERANDS.jsonl')
    results=[]
    for row in rows:
        step=row['step']
        mf={(r['name'],r['kind']):np.load(STATE/r['file']) for r in row['records']}
        for branch,call in [('model.mtp.output_layer',(step-201)*2+1),('model.output_layer',(step-201)*2+2)]:
            with np.load(ROOT/'runs'/HP/f'head-operands-call{call}.npz') as hp:
                results.append({'step':step,'branch':branch,'x':stats(hp['x'],mf[branch,'x']),
                                'dy':stats(hp['dy'],mf[branch,'dy_matmul'])})
    report={'complete':True,'checks':checks,'records':results,'scope':'selected496hidden and539vocab columns, all4096tokens'}
    (ROOT/'evidence/head-selected-operands-six-step-comparison.json').write_text(json.dumps(report,indent=2))
    print(json.dumps([{'step':r['step'],'branch':r['branch'],'x_different':r['x']['different'],
                      'dy_different':r['dy']['different'],'dy_rel_l2':r['dy']['rel_l2']} for r in results]))


if __name__=='__main__':main()
