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

"""Compare native MoE212 branches, actual joins and same-input router precision."""
import hashlib
import json
from pathlib import Path

import numpy as np
from compare_vocab_optimizer_trace import fidelity, read_rows
from compare_head_selected_operands import stats
from compare_head_weight_boundaries import rounded_bf16

ROOT=Path('/alignment/phase12')
STATE=Path('/state/jt-recipe-20261014')
MF='mf-moebranches212'
HP='hp-jtf637-norm208-moebranches212'


def main():
    checks={}
    for label,run,baseline in [('mf',MF,'mf-resume'),('hp',HP,'hp-jtf637-norm208-q-router-independent226')]:
        checks[label+'_exit_zero']=(ROOT/'runs'/(run+'.exit')).read_text().strip()=='0'
        if not checks[label+'_exit_zero']:raise ValueError('Incomplete probe: '+run)
        checks.update({label+'_'+k:v for k,v in fidelity(run,baseline,12).items()})
    checks['hp_optimizer_trace_exact'] = read_rows(ROOT/'runs'/HP/'VOCAB_OPTIMIZER_TRACE.jsonl') == read_rows(
        ROOT/'runs/hp-jtf637-norm208-q-router-independent226/VOCAB_OPTIMIZER_TRACE.jsonl')[:24]
    for side,run,control in [('hp',HP,'hp-jtf637-norm208-preclip212'),('mf',MF,'mf-preclip212')]:
        for step in [211,212]:
            actual = json.loads((ROOT/'runs'/run/('PRECLIP-'+str(step)+'.json')).read_text())
            expected = json.loads((ROOT/'runs'/control/('PRECLIP-'+str(step)+'.json')).read_text())
            keep = lambda row: {k:row[k] for k in ['name','shape','dtype','sha256','square_sum_fp64','array_bytes']}
            checks[side+'_46_preclip_exact_'+str(step)] = len(actual['records']) == 46 and [keep(x) for x in actual['records']] == [keep(x) for x in expected['records']]
            source = 'hp_preclip212_hashes.py' if side == 'hp' else 'mf_preclip212_hashes.py'
            folder = 'tools-f637' if side == 'hp' else 'tools'
            checks[side+'_preclip_source_'+str(step)] = actual['source_sha256'] == hashlib.sha256((ROOT/folder/source).read_bytes()).hexdigest()
    mm=json.loads((STATE/MF/'deep/manifest.json').read_text())
    hm=json.loads((ROOT/'runs'/HP/'MOE_BRANCH_OBSERVER.json').read_text())
    checks['mf_source_exact']=mm['source_sha256']==hashlib.sha256((ROOT/'tools/mf_moe212_branches_tap.py').read_bytes()).hexdigest()
    checks['hp_source_exact']=hm['source_sha256']==hashlib.sha256((ROOT/'tools-f637/hp_moe212_branches_factor.py').read_bytes()).hexdigest()
    if not all(checks.values()):raise ValueError(checks)
    mr=next(r for r in read_rows(ROOT/'runs'/MF/'MOE_BRANCHES.jsonl') if r['step']==212)['records']
    hr=json.loads((ROOT/'runs'/HP/'MOE_BRANCHES.json').read_text())['records']
    mmap={(r['name'],r['kind']):r for r in mr};hmap={(r['name'],r['kind']):r for r in hr}
    def load(side,name,kind,width=5120):
        record=(mmap if side=='mf' else hmap)[name,kind]
        folder=STATE/MF/'deep' if side=='mf' else STATE/HP/'branches'
        a=np.load(folder/record['file'],mmap_mode='r')
        if hashlib.sha256(a.tobytes()).hexdigest()!=record['sha256']:raise ValueError('Dump changed')
        return a.reshape(-1,width)
    prefix='model.decoder.layers.1.mlp'
    m={kind:load('mf',prefix,kind) for kind in ['x','y','dy','dx']}
    h={kind:load('hp','main',kind) for kind in ['x','y','dy','dx']}
    me=load('mf',prefix+'.experts','dx');ms=load('mf',prefix+'.shared_experts.linear_fc1','dx')
    mx=load('mf',prefix+'.router','dx');mw=load('mf',prefix+'.router','weight')
    my=load('mf',prefix+'.router','dy',mw.shape[0])
    he=load('hp','pair','routed_dx')[hm['padding']:]
    hs=load('hp','shared','dx');hx=load('hp','router','dx');hw=load('hp','router','weight')
    hy=load('hp','router','dy',hw.shape[0]);hj=load('hp','pair','joined_dx')[hm['padding']:]
    hp_pair_router=load('hp','pair','router_dx')[hm['padding']:]
    comparisons={kind:stats(h[kind],m[kind]) for kind in ['x','y','dy','dx']}
    comparisons['y_hp_rounded_bf16_vs_mf'] = stats(rounded_bf16(h['y']), m['y'])
    comparisons.update(router_weight=stats(hw,mw),router_dy=stats(hy,my),router_dx=stats(hx,mx),
                       routed_dx=stats(he,me),shared_dx=stats(hs,ms),hp_router_to_pair=stats(hx,hp_pair_router))
    joins={'hp_actual_pair':stats(rounded_bf16(he+hx),hj),
           'readonly_hp_with_mf_router':stats(rounded_bf16(rounded_bf16(he+mx)+hs),m['dx']),
           'hp_actual_combined':stats(rounded_bf16(hj+hs),h['dx']),
           'readonly_hp_with_mf_shared':stats(rounded_bf16(hj+ms),m['dx']),
           'readonly_mf_with_hp_shared':stats(rounded_bf16(rounded_bf16(me+mx)+hs),h['dx']),
           'mf_round_expert_router_then_shared':stats(rounded_bf16(rounded_bf16(me+mx)+ms),m['dx']),
           'mf_round_router_shared_then_expert':stats(rounded_bf16(rounded_bf16(mx+ms)+me),m['dx']),
           'mf_fp32_expert_router_then_shared':stats(rounded_bf16((me+mx)+ms),m['dx']),
           'mf_fp32_expert_shared_then_router':stats(rounded_bf16((me+ms)+mx),m['dx'])}
    report={'complete':True,'fidelity':checks,'padding':hm['padding'],'comparisons':comparisons,'joins':joins,
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'forward_scope':'Raw HP MoE output is FP32 branch addition; BF16-rounded HP output equals MF. Raw y differences compare different precision boundaries and are not a new forward defect.',
            'scope':'Actual complete212 branch gradients, 12-step metrics/input/restore and46 preclip hashes at211/212 verified; no intervention or reference injection.'}
    (ROOT/'evidence/moe212-branch-comparison.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report))


if __name__=='__main__':main()
