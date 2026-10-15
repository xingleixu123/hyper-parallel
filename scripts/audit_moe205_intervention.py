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

"""Audit the49-point diagnostic intervention without counting it as independent acceptance."""
import hashlib
import json
from pathlib import Path
from compare_vocab_optimizer_trace import fidelity, read_rows
from compare_loss_components import read_metrics

ROOT=Path('/alignment/phase12')
RUN='hp-jtf637-adamw-all-moe49diag206'
CONTROL='hp-jtf637-adamw-all-archive206'


def main():
    folder=ROOT/'runs'/RUN
    checks={'exit_zero':(ROOT/'runs'/(RUN+'.exit')).read_text().strip()=='0'}
    if not checks['exit_zero']:
        raise ValueError('Diagnostic did not finish')
    observed=read_rows(folder/'metrics.jsonl')
    control=read_rows(ROOT/'runs'/CONTROL/'metrics.jsonl')
    checks['complete_six_steps']=[r['step'] for r in observed]==list(range(201,207))
    checks['pre_intervention_metrics_exact']=observed[:4]==control[:4]
    f=fidelity(RUN,CONTROL,6)
    checks.update({k:v for k,v in f.items() if k!='all_metrics_grad_norm_exact'})
    checks['source_revision_exact']=json.loads((folder/'SOURCE_REVISION.json').read_text())==json.loads((ROOT/'runs'/CONTROL/'SOURCE_REVISION.json').read_text())
    a=[r for r in read_rows(folder/'VOCAB_OPTIMIZER_TRACE.jsonl') if r['step']<=204]
    b=[r for r in read_rows(ROOT/'runs'/CONTROL/'VOCAB_OPTIMIZER_TRACE.jsonl') if r['step']<=204]
    checks['pre_intervention_optimizer_trace_exact']=a==b
    marker=json.loads((folder/'DIAGNOSTIC_ONLY.json').read_text())
    change=json.loads((folder/'MOE205_INTERVENTION.json').read_text())
    checks['exact49_intervention_verified']=marker['diagnostic_only'] and not marker['eligible_for_acceptance'] and change['changed_elements']==49 and change['all_other_elements_unchanged'] and change['output_sha256']==change['native_mf_sha256']
    checks['intervention_source_exact']=change['source_sha256']==hashlib.sha256((ROOT/'tools-f637/hp_moe205_intervention.py').read_bytes()).hexdigest()
    if not all(checks.values()):
        raise ValueError(checks)
    gradient=json.loads((ROOT/'evidence'/(RUN+'-grad205.json')).read_text())
    mf=read_metrics(ROOT/'runs/mf-resume/metrics.jsonl','mf',1e-4,.3)
    hp=read_metrics(folder/'metrics.jsonl','hp',1e-4,.3)
    base=read_metrics(ROOT/'runs'/CONTROL/'metrics.jsonl','hp',1e-4,.3)
    report={'complete':True,'diagnostic_only':True,'eligible_for_acceptance':False,'checks':checks,
        'gradient_count':len(gradient['records']),
        'replay_exact':sum(r['preclip_replay_mf_coefficient']['different']==0 for r in gradient['records']),
        'actual_postclip_exact':sum(r['actual_postclip']['different']==0 for r in gradient['records']),
        'remaining_gradient_replay':[{'name':r['name'],**{k:v for k,v in r['preclip_replay_mf_coefficient'].items() if k not in ('examples','actual_unscaled_sha256')}} for r in gradient['records'] if r['preclip_replay_mf_coefficient']['different']],
        'loss_errors':{str(step):{'control':{k:base[step][k]-mf[step][k] for k in mf[step]},
                                'intervention':{k:hp[step][k]-mf[step][k] for k in mf[step]}} for step in (205,206)},
        'norm205':{'hp':gradient['hp_norm'],'mf':gradient['mf_norm']}}
    (ROOT/'evidence/moe205-49point-causal-audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report))


if __name__=='__main__':
    main()
