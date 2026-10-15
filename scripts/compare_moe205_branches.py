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

"""Compare native MoE205 branches, actual joins and same-input router precision."""
import hashlib
import json
from pathlib import Path

import numpy as np
from compare_vocab_optimizer_trace import fidelity, read_rows
from compare_head_selected_operands import stats
from compare_head_weight_boundaries import rounded_bf16

ROOT=Path('/alignment/phase12')
STATE=Path('/state/jt-recipe-20261014')
MF='mf-moe205-branches206-r2'
HP='hp-jtf637-adamw-all-moebranches206'


def main():
    checks={}
    for label,run,baseline in [('mf',MF,'mf-resume'),('hp',HP,'hp-jtf637-adamw-all-archive206')]:
        checks[label+'_exit_zero']=(ROOT/'runs'/(run+'.exit')).read_text().strip()=='0'
        if not checks[label+'_exit_zero']:raise ValueError('Incomplete probe: '+run)
        checks.update({label+'_'+k:v for k,v in fidelity(run,baseline,6).items()})
    checks['hp_optimizer_trace_exact']=(ROOT/'runs'/HP/'VOCAB_OPTIMIZER_TRACE.jsonl').read_bytes()==(
        ROOT/'runs/hp-jtf637-adamw-all-archive206/VOCAB_OPTIMIZER_TRACE.jsonl').read_bytes()
    mm=json.loads((STATE/MF/'deep/manifest.json').read_text())
    hm=json.loads((ROOT/'runs'/HP/'MOE_BRANCH_OBSERVER.json').read_text())
    checks['mf_source_exact']=mm['source_sha256']==hashlib.sha256((ROOT/'tools/mf_moe205_branches_tap_r2.py').read_bytes()).hexdigest()
    checks['hp_source_exact']=hm['source_sha256']==hashlib.sha256((ROOT/'tools-f637/hp_moe205_branches_factor.py').read_bytes()).hexdigest()
    if not all(checks.values()):raise ValueError(checks)
    mr=next(r for r in read_rows(ROOT/'runs'/MF/'MOE_BRANCHES.jsonl') if r['step']==205)['records']
    hr=json.loads((ROOT/'runs'/HP/'MOE_BRANCHES.json').read_text())['records']
    mmap={(r['name'],r['kind']):r for r in mr};hmap={(r['name'],r['kind']):r for r in hr}
    def load(side,name,kind,width=5120):
        record=(mmap if side=='mf' else hmap)[name,kind]
        folder=STATE/MF/'deep' if side=='mf' else STATE/HP/'branches'
        a=np.load(folder/record['file'],mmap_mode='r')
        if hashlib.sha256(a.tobytes()).hexdigest()!=record['sha256']:raise ValueError('Dump changed')
        return a.reshape(-1,width)
    prefix='model.decoder.layers.1.mlp'
    m={kind:load('mf',prefix,kind) for kind in ['x','dy','dx']}
    h={kind:load('hp','main',kind) for kind in ['x','dy','dx']}
    me=load('mf',prefix+'.experts','dx');ms=load('mf',prefix+'.shared_experts.linear_fc1','dx')
    mx=load('mf',prefix+'.router','dx');mw=load('mf',prefix+'.router','weight')
    my=load('mf',prefix+'.router','dy',mw.shape[0])
    he=load('hp','pair','routed_dx')[hm['padding']:]
    hs=load('hp','shared','dx');hx=load('hp','router','dx');hw=load('hp','router','weight')
    hy=load('hp','router','dy',hw.shape[0]);hj=load('hp','pair','joined_dx')[hm['padding']:]
    hp_pair_router=load('hp','pair','router_dx')[hm['padding']:]
    comparisons={kind:stats(h[kind],m[kind]) for kind in ['x','dy','dx']}
    comparisons.update(router_weight=stats(hw,mw),router_dy=stats(hy,my),router_dx=stats(hx,mx),
                       routed_dx=stats(he,me),shared_dx=stats(hs,ms),hp_router_to_pair=stats(hx,hp_pair_router))
    joins={'hp_actual_pair':stats(rounded_bf16(he+hx),hj),
           'hp_actual_combined':stats(rounded_bf16(hj+hs),h['dx']),
           'readonly_hp_with_mf_shared':stats(rounded_bf16(hj+ms),m['dx']),
           'readonly_mf_with_hp_shared':stats(rounded_bf16(rounded_bf16(me+mx)+hs),h['dx']),
           'mf_round_expert_router_then_shared':stats(rounded_bf16(rounded_bf16(me+mx)+ms),m['dx']),
           'mf_round_router_shared_then_expert':stats(rounded_bf16(rounded_bf16(mx+ms)+me),m['dx']),
           'mf_fp32_expert_router_then_shared':stats(rounded_bf16((me+mx)+ms),m['dx']),
           'mf_fp32_expert_shared_then_router':stats(rounded_bf16((me+ms)+mx),m['dx'])}
    report={'complete':True,'fidelity':checks,'padding':hm['padding'],'comparisons':comparisons,'joins':joins,
            'scope':'Actual complete205 branch gradients; no intervention or reference injection.'}
    (ROOT/'evidence/moe205-branch-comparison.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report))


if __name__=='__main__':main()
