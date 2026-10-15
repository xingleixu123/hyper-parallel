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

"""Compare actual205 main MoE boundaries after independent six-step fidelity checks."""
import hashlib
import json
from pathlib import Path

import numpy as np
from compare_vocab_optimizer_trace import fidelity, read_rows
from compare_head_selected_operands import stats
from compare_head_weight_boundaries import rounded_bf16

ROOT = Path('/alignment/phase12')
STATE = Path('/state/jt-recipe-20261014')
MF = 'mf-moe205-boundaries206'
HP = 'hp-jtf637-adamw-all-moe205206'


def main():
    checks = {}
    for label, run, baseline in [('mf',MF,'mf-resume'),('hp',HP,'hp-jtf637-adamw-all-archive206')]:
        checks[label+'_exit_zero'] = (ROOT/'runs'/(run+'.exit')).read_text().strip()=='0'
        if not checks[label+'_exit_zero']:
            raise ValueError('Incomplete probe: '+run)
        checks.update({label+'_'+k:v for k,v in fidelity(run,baseline,6).items()})
    checks['hp_selected_optimizer_trace_exact'] = (ROOT/'runs'/HP/'VOCAB_OPTIMIZER_TRACE.jsonl').read_bytes()==(
        ROOT/'runs/hp-jtf637-adamw-all-archive206/VOCAB_OPTIMIZER_TRACE.jsonl').read_bytes()
    mm=json.loads((STATE/MF/'deep/manifest.json').read_text())
    hm=json.loads((ROOT/'runs'/HP/'MOE_OBSERVER.json').read_text())
    checks['mf_source_exact']=mm['source_sha256']==hashlib.sha256((ROOT/'tools/mf_moe205_boundaries_tap.py').read_bytes()).hexdigest()
    checks['hp_source_exact']=hm['source_sha256']==hashlib.sha256((ROOT/'tools-f637/hp_moe205_boundaries_factor.py').read_bytes()).hexdigest()
    if not all(checks.values()):
        raise ValueError(checks)
    mr=next(r for r in read_rows(ROOT/'runs'/MF/'MOE_BOUNDARIES.jsonl') if r['step']==205)['records']
    hr=json.loads((ROOT/'runs'/HP/'MOE_BOUNDARIES.json').read_text())['records']
    records=[]
    for kind in ['x','y','dy','dx']:
        m=next(r for r in mr if r['kind']==kind)
        h=next(r for r in hr if r['kind']==kind)
        a=np.load(STATE/MF/'deep'/m['file'],mmap_mode='r')
        b=np.load(STATE/HP/'moe'/h['file'],mmap_mode='r')
        for v,r in [(a,m),(b,h)]:
            if hashlib.sha256(v.tobytes()).hexdigest()!=r['sha256']:
                raise ValueError('Boundary array changed')
        a=a.reshape(-1,5120);b=b.reshape(-1,5120)
        record={'kind':kind,'hp_actual_dtype':h['dtype'],'comparison':stats(b,a)}
        if kind in ('dx','y'):
            record['after_bf16_cast']=stats(rounded_bf16(b),a)
        records.append(record)
    report={'complete':True,'fidelity':checks,'step':205,'records':records,
            'scope':'Entire main layer1 MoE x/y/dy/dx,4096x5120 each; original native computations, no teacher injection.'}
    (ROOT/'evidence/moe205-full-boundary-comparison.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report))


if __name__=='__main__':
    main()
