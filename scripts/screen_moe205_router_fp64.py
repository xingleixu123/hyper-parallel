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

"""Screen same-input FP64 router arithmetic against the observed MoE205 boundary."""
import hashlib
import json
from pathlib import Path

import numpy as np
from compare_vocab_optimizer_trace import fidelity,read_rows
from compare_head_selected_operands import stats
from compare_head_weight_boundaries import rounded_bf16

ROOT=Path('/alignment/phase12')
STATE=Path('/state/jt-recipe-20261014')
HP='hp-jtf637-adamw-all-moebranches206'
MF='mf-moe205-boundaries206'


def round_double_bf16(values):
    if not np.isfinite(values).all() or np.any((np.abs(values)<2.0**-126)&(values!=0)):
        raise ValueError('This exact mantissa rounding needs finite BF16-normal values')
    bits=np.ascontiguousarray(values,dtype=np.float64).view(np.uint64)
    rounded=(bits+np.uint64((1<<44)-1)+((bits>>np.uint64(45))&np.uint64(1)))&np.uint64(0xffffe00000000000)
    return rounded.view(np.float64).astype(np.float32)


def main():
    checks=fidelity(HP,'hp-jtf637-adamw-all-archive206',6)
    if not all(checks.values()):raise ValueError(checks)
    rows=json.loads((ROOT/'runs'/HP/'MOE_BRANCHES.json').read_text())['records']
    hm=json.loads((ROOT/'runs'/HP/'MOE_BRANCH_OBSERVER.json').read_text())
    def load(name,kind,width=5120):
        r=next(r for r in rows if r['name']==name and r['kind']==kind)
        a=np.load(STATE/HP/'branches'/r['file'],mmap_mode='r')
        if hashlib.sha256(a.tobytes()).hexdigest()!=r['sha256']:raise ValueError('Observed array changed')
        return a.reshape(-1,width)
    mr=next(r for r in read_rows(ROOT/'runs'/MF/'MOE_BOUNDARIES.jsonl') if r['step']==205)['records']
    r=next(r for r in mr if r['kind']=='dx')
    mf=np.load(STATE/MF/'deep'/r['file'],mmap_mode='r').reshape(-1,5120)
    dy=load('router','dy',8);weight=load('router','weight');native=load('router','dx')
    expert=load('pair','routed_dx')[hm['padding']:];shared=load('shared','dx');joint=load('main','dx')
    reference=dy.astype(np.float64)@weight.astype(np.float64)
    rounded=reference.astype(np.float32)
    normal=rounded_bf16(rounded_bf16(expert+native)+shared)
    candidates={'native':normal,'router_fp64_then_fp32':rounded_bf16(rounded_bf16(expert+rounded)+shared),
        'pair_fp64_round_direct_bf16':rounded_bf16(round_double_bf16(expert.astype(np.float64)+reference)+shared)}
    report={'complete':True,'fidelity':checks,'native_join_replay':stats(normal,joint),
        'router_vs_fp64_rounded':stats(native,rounded),
        'combined_vs_native_mf':{name:stats(value,mf) for name,value in candidates.items()},
        'scope':'Read-only same-HP-input router FP64 GEMM; MF router operands not yet compared. Conditional on actual routed/shared gradients; not whole-model correctness or acceptance.'}
    (ROOT/'evidence/moe205-router-fp64-screen.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report))


if __name__=='__main__':main()
