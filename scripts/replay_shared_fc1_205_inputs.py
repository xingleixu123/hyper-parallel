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

"""Screen independent NPU dinput GEMM layouts with actual shared-FC1 operands."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional
import torch_npu
from hyper_parallel.trainer.runtime.random import enable_full_determinism
from compare_head_selected_operands import stats

ROOT=Path('/alignment/phase12')
STATE=Path('/state/jt-recipe-20261014')
MF='mf-sharedfc1-205206'
HP='hp-jtf637-adamw-all-sharedfc1206'


def main():
    prior=json.loads((ROOT/'evidence/shared-fc1-205-full-comparison.json').read_text())
    if not prior['complete'] or not prior['comparisons']['weight']['exact'] or prior['comparisons']['dy']['different']!=1:
        raise ValueError('Actual same-input evidence is required before GEMM screening')
    hr=json.loads((ROOT/'runs'/HP/'SHARED_FC1_BOUNDARIES.json').read_text())['records']
    mr=next(json.loads(s) for s in (ROOT/'runs'/MF/'SHARED_FC1_BOUNDARIES.jsonl').read_text().splitlines()
            if json.loads(s)['step']==205)['records']
    def load(rows,folder,kind):
        r=next(r for r in rows if r['kind']==kind);a=np.load(folder/r['file'],mmap_mode='r')
        if hashlib.sha256(a.tobytes()).hexdigest()!=r['sha256']:raise ValueError('Operand hash changed')
        return a
    weight=load(hr,STATE/HP/'shared_fc1','weight')
    dy=load(hr,STATE/HP/'shared_fc1','dy').reshape(-1,weight.shape[0])
    hp=load(hr,STATE/HP/'shared_fc1','dx').reshape(-1,5120)
    mf=load(mr,STATE/MF/'deep','dx').reshape(-1,5120)
    mf_dy=load(mr,STATE/MF/'deep','dy').reshape(dy.shape)
    enable_full_determinism(1,close_matmul_k_shift=False)
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    a=torch.from_numpy(np.array(dy)).to(device='npu',dtype=torch.bfloat16)
    b=torch.from_numpy(np.array(weight)).to(device='npu',dtype=torch.bfloat16)
    other=torch.from_numpy(np.array(mf_dy)).to(device='npu',dtype=torch.bfloat16)
    variants=[('native_nn',lambda:torch.mm(a,b)),('native_nn_with_mf_dy',lambda:torch.mm(other,b))]
    records=[]
    for name,operation in variants:
        value=operation().float().cpu().numpy()
        record={'variant':name,'vs_native_hp':stats(value,hp),'vs_native_mf':stats(value,mf),
                'sha256':hashlib.sha256(value.tobytes()).hexdigest()}
        records.append(record);print(json.dumps(record),flush=True)
        if name=='native_nn' and not record['vs_native_hp']['exact']:
            raise ValueError('Replay does not reproduce actual HP dinput')
        del value
    report={'complete':True,'records':records,'runtime':{'torch':torch.__version__,
        'deterministic':torch.are_deterministic_algorithms_enabled(),
        'CLOSE_MATMUL_K_SHIFT':os.environ.get('CLOSE_MATMUL_K_SHIFT')},
        'scope':'Replay both actual upstream gradients with the same BF16 weight and full GEMM shape on CANN9.2. Diagnostic only; no training acceptance.'}
    (ROOT/'evidence/shared-fc1-205-input-replay.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
