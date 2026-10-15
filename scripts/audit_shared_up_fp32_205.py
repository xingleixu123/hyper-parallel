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

"""Audit the independent FP32 shared-up factor against its unchanged control."""
import hashlib
import json
from pathlib import Path
from compare_vocab_optimizer_trace import fidelity, read_rows
from compare_loss_components import read_metrics

ROOT=Path('/alignment/phase12')
RUN='hp-jtf637-adamw-all-sharedupfp32206'
CONTROL='hp-jtf637-adamw-all-archive206'


def main():
    folder=ROOT/'runs'/RUN
    checks={'exit_zero':(ROOT/'runs'/(RUN+'.exit')).read_text().strip()=='0'}
    if not checks['exit_zero']:
        raise ValueError('Candidate did not finish')
    observed=read_rows(folder/'metrics.jsonl')
    control=read_rows(ROOT/'runs'/CONTROL/'metrics.jsonl')
    checks['complete_six_steps']=[r['step'] for r in observed]==list(range(201,207))
    checks['pre205_metrics_exact']=observed[:4]==control[:4]
    f=fidelity(RUN,CONTROL,6)
    checks.update({k:v for k,v in f.items() if k!='all_metrics_grad_norm_exact'})
    checks['source_revision_exact']=json.loads((folder/'SOURCE_REVISION.json').read_text())==json.loads((ROOT/'runs'/CONTROL/'SOURCE_REVISION.json').read_text())
    a=[r for r in read_rows(folder/'VOCAB_OPTIMIZER_TRACE.jsonl') if r['step']<=204]
    b=[r for r in read_rows(ROOT/'runs'/CONTROL/'VOCAB_OPTIMIZER_TRACE.jsonl') if r['step']<=204]
    checks['pre205_optimizer_trace_exact']=a==b
    factor=json.loads((folder/'SHARED_UP_FP32_FACTOR.json').read_text())
    checks['no_teacher_values']=factor['teacher_values_used'] is False
    checks['factor_source_exact']=factor['source_sha256']==hashlib.sha256((ROOT/'tools-f637/hp_shared_up_fp32_factor.py').read_bytes()).hexdigest()
    if not all(checks.values()):
        raise ValueError(checks)
    gradient=json.loads((ROOT/'evidence'/(RUN+'-grad205.json')).read_text())
    mf=read_metrics(ROOT/'runs/mf-resume/metrics.jsonl','mf',1e-4,.3)
    hp=read_metrics(folder/'metrics.jsonl','hp',1e-4,.3)
    base=read_metrics(ROOT/'runs'/CONTROL/'metrics.jsonl','hp',1e-4,.3)
    report={'complete':True,'diagnostic_only':False,'independent_arithmetic':True,'full200_acceptance':False,'checks':checks,
        'gradient_count':len(gradient['records']),
        'replay_exact':sum(r['preclip_replay_mf_coefficient']['different']==0 for r in gradient['records']),
        'actual_postclip_exact':sum(r['actual_postclip']['different']==0 for r in gradient['records']),
        'remaining_gradient_replay':[{'name':r['name'],**{k:v for k,v in r['preclip_replay_mf_coefficient'].items() if k not in ('examples','actual_unscaled_sha256')}} for r in gradient['records'] if r['preclip_replay_mf_coefficient']['different']],
        'loss_errors':{str(step):{'control':{k:base[step][k]-mf[step][k] for k in mf[step]},
                                'candidate':{k:hp[step][k]-mf[step][k] for k in mf[step]}} for step in (205,206)},
        'norm205':{'hp':gradient['hp_norm'],'mf':gradient['mf_norm']}}
    (ROOT/'evidence/shared-up-fp32-205-independent-audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report))


if __name__=='__main__':
    main()
