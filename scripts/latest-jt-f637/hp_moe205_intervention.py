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

"""DIAGNOSTIC ONLY: replace exactly49 measured MoE205 dx coordinates, never acceptance."""
import hashlib
import json
from pathlib import Path
from types import MethodType

import numpy as np
import torch
from capture_hp_native import Observer

ROOT=Path('/alignment/phase12')
STATE=Path('/state/jt-recipe-20261014')
CONTEXT={}


class DiagnosticBoundary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value

    @staticmethod
    def backward(ctx, gradient):
        if CONTEXT['step']!=205:
            return gradient
        actual=gradient.detach().float().cpu().numpy().reshape(-1)
        before=hashlib.sha256(actual.tobytes()).hexdigest()
        if before!=CONTEXT['hp_sha256']:
            raise ValueError('Intervention input differs from independent observed control')
        result=gradient.clone()
        flat=result.reshape(-1)
        indices=torch.as_tensor(CONTEXT['indices'],dtype=torch.int64,device=gradient.device)
        values=torch.as_tensor(CONTEXT['values'],dtype=gradient.dtype,device=gradient.device)
        flat[indices]=values
        updated=result.detach().float().cpu().numpy().reshape(-1)
        changed=int(np.count_nonzero(updated!=actual))
        after=hashlib.sha256(updated.tobytes()).hexdigest()
        if changed!=49 or after!=CONTEXT['mf_sha256']:
            raise ValueError('Intervention did not replace exactly49 coordinates')
        (CONTEXT['output']/'MOE205_INTERVENTION.json').write_text(json.dumps({
            'diagnostic_only':True,'eligible_for_acceptance':False,'step':205,'changed_elements':changed,
            'input_sha256':before,'output_sha256':after,'native_mf_sha256':CONTEXT['mf_sha256'],
            'all_other_elements_unchanged':True,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2))
        return result


def install():
    original_begin=Observer.on_train_begin
    original_end=Observer.on_step_end

    def begin(self,state,**kwargs):
        original_begin(self,state,**kwargs)
        mf_run='mf-moe205-boundaries206';hp_run='hp-jtf637-adamw-all-moe205206'
        mr=next(json.loads(s) for s in (ROOT/'runs'/mf_run/'MOE_BOUNDARIES.jsonl').read_text().splitlines()
                if json.loads(s)['step']==205)['records']
        hr=json.loads((ROOT/'runs'/hp_run/'MOE_BOUNDARIES.json').read_text())['records']
        m=next(r for r in mr if r['kind']=='dx');h=next(r for r in hr if r['kind']=='dx')
        a=np.load(STATE/mf_run/'deep'/m['file'],mmap_mode='r').reshape(-1)
        b=np.load(STATE/hp_run/'moe'/h['file'],mmap_mode='r').reshape(-1)
        if hashlib.sha256(a.tobytes()).hexdigest()!=m['sha256'] or hashlib.sha256(b.tobytes()).hexdigest()!=h['sha256']:
            raise ValueError('Reference observation changed')
        indices=np.flatnonzero(a!=b)
        if len(indices)!=49:
            raise ValueError('Expected exactly49 measured mismatches')
        CONTEXT.update(step=int(state.global_step)+1,output=self.output,indices=indices,
                       values=np.array(a[indices],copy=True),hp_sha256=h['sha256'],mf_sha256=m['sha256'])
        module=dict(self.trainer.model.named_modules())['model.layers.1.mlp']
        original=module.forward
        def forward(self,values):
            del self
            return original(DiagnosticBoundary.apply(values))
        module.forward=MethodType(forward,module)
        (self.output/'DIAGNOSTIC_ONLY.json').write_text(json.dumps({'diagnostic_only':True,
            'eligible_for_acceptance':False,'factor':'Only49 actual MoE205 input-gradient coordinates from native MF',
            'teacher_values_used':True,'scope':'No model or optimizer state replacement; steps201-204 unchanged'},indent=2))

    def end(self,state,**kwargs):
        original_end(self,state,**kwargs)
        CONTEXT['step']=int(state.global_step)+1

    Observer.on_train_begin=begin
    Observer.on_step_end=end
