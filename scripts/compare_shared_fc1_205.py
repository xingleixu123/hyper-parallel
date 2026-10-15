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

"""Validate shared-FC1 observations and compare same-input dinput accuracy exactly."""
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from compare_vocab_optimizer_trace import fidelity,read_rows
from compare_head_selected_operands import stats

ROOT=Path('/alignment/phase12')
STATE=Path('/state/jt-recipe-20261014')
MF='mf-sharedfc1-205206'
HP='hp-jtf637-adamw-all-sharedfc1206'


def exact_dot(left,right):
    products=left.astype(np.float64)*right.astype(np.float64)
    ratios=[float(v).as_integer_ratio() for v in products if v!=0]
    shift=max((den.bit_length()-1 for _,den in ratios),default=0)
    total=sum(num<<(shift-(den.bit_length()-1)) for num,den in ratios)
    return Fraction(total,1<<shift),math.fsum(products)


def main():
    checks={}
    for label,run,baseline in [('mf',MF,'mf-resume'),('hp',HP,'hp-jtf637-adamw-all-archive206')]:
        checks[label+'_exit_zero']=(ROOT/'runs'/(run+'.exit')).read_text().strip()=='0'
        if not checks[label+'_exit_zero']:raise ValueError('Incomplete probe')
        checks.update({label+'_'+k:v for k,v in fidelity(run,baseline,6).items()})
    checks['hp_optimizer_trace_exact']=(ROOT/'runs'/HP/'VOCAB_OPTIMIZER_TRACE.jsonl').read_bytes()==(
        ROOT/'runs/hp-jtf637-adamw-all-archive206/VOCAB_OPTIMIZER_TRACE.jsonl').read_bytes()
    mm=json.loads((STATE/MF/'deep/manifest.json').read_text())
    hm=json.loads((ROOT/'runs'/HP/'SHARED_FC1_OBSERVER.json').read_text())
    checks['mf_source_exact']=mm['source_sha256']==hashlib.sha256((ROOT/'tools/mf_shared_fc1_205_tap.py').read_bytes()).hexdigest()
    checks['hp_source_exact']=hm['source_sha256']==hashlib.sha256((ROOT/'tools-f637/hp_shared_fc1_205_factor.py').read_bytes()).hexdigest()
    if not all(checks.values()):raise ValueError(checks)
    mr=next(r for r in read_rows(ROOT/'runs'/MF/'SHARED_FC1_BOUNDARIES.jsonl') if r['step']==205)['records']
    hr=json.loads((ROOT/'runs'/HP/'SHARED_FC1_BOUNDARIES.json').read_text())['records']
    arrays={}
    for label,rows,folder in [('mf',mr,STATE/MF/'deep'),('hp',hr,STATE/HP/'shared_fc1')]:
        for row in rows:
            a=np.load(folder/row['file'],mmap_mode='r')
            if hashlib.sha256(a.tobytes()).hexdigest()!=row['sha256']:raise ValueError('Observed data changed')
            arrays[label,row['kind']]=a
    if not mm['transpose_b']:raise ValueError('Unexpected native weight orientation')
    width=arrays['hp','weight'].shape[0]
    comparisons={}
    for kind in ['x','weight','y','dy','dx']:
        last=width if kind in ['y','dy'] else 5120
        comparisons[kind]=stats(arrays['hp',kind].reshape(-1,last),arrays['mf',kind].reshape(-1,last))
    accuracy=None
    if comparisons['weight']['exact'] and comparisons['dy']['exact']:
        dy=arrays['hp','dy'].reshape(-1,width);weight=arrays['hp','weight']
        a=arrays['hp','dx'].reshape(-1,5120);b=arrays['mf','dx'].reshape(-1,5120)
        coordinates=np.argwhere(a!=b)
        if len(coordinates)>10000:raise ValueError('Unexpected large mismatch; inspect before exact replay')
        for v in [dy,weight]:
            if np.any(v.view(np.uint32)&np.uint32(0xffff)):raise ValueError('Exact BF16 product assumption failed')
        counts={'hp_closer':0,'mf_closer':0,'tie':0,'fp64_fsum_exact_rounded':0}
        examples=[]
        for row,column in coordinates:
            ideal,fp64=exact_dot(dy[row],weight[:,column])
            ae=abs(Fraction.from_float(float(a[row,column]))-ideal)
            be=abs(Fraction.from_float(float(b[row,column]))-ideal)
            winner='hp_closer' if ae<be else 'mf_closer' if be<ae else 'tie'
            counts[winner]+=1;counts['fp64_fsum_exact_rounded']+=int(float(ideal)==fp64)
            if len(examples)<12:examples.append({'row':int(row),'column':int(column),'hp':float(a[row,column]),
                'mf':float(b[row,column]),'exact_fraction':str(ideal),'winner':winner})
        accuracy={'differing_coordinates':len(coordinates),'counts':counts,'examples':examples,
            'scope':'Exact rational sum of BF16 dy*weight at every differing dinput coordinate; only local GEMM correctness.'}
    report={'complete':True,'fidelity':checks,'comparisons':comparisons,'accuracy':accuracy,
            'scope':'Full main shared-expert FC1 x/weight/y/dy/dx, same independent all-AdamW trajectory.'}
    (ROOT/'evidence/shared-fc1-205-full-comparison.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='accuracy'}))
    print(json.dumps({'accuracy':{k:v for k,v in accuracy.items() if k!='examples'}}) if accuracy else 'Upstream operands differ; no accuracy comparison')


if __name__=='__main__':main()
