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

"""Observe real vocabulary optimizer values only at previously located coordinates."""
import json
from functools import wraps
from pathlib import Path
import torch
from capture_hp_native import Observer, leaves


def install():
    original = Observer.on_train_begin

    def begin(self, state, **kwargs):
        original(self, state, **kwargs)
        coordinates = json.loads(Path('/alignment/phase12/evidence/hp-denseadj-vocab204-coordinates-coordinates.json').read_text())
        selected = {r['name']: [c['flat_index'] for c in r['coordinates']] for r in coordinates['records']}
        names = {id(p): name for name, p in self.trainer.model.named_parameters()}
        output = self.output/'VOCAB_OPTIMIZER_TRACE.jsonl'
        metadata = {'teacher_values_used': False, 'coordinates': selected,
                    'scope': 'Actual leaf optimizer input and output at fixed coordinates; no tensor replacement'}
        (self.output/'VOCAB_OPTIMIZER_TRACE.json').write_text(json.dumps(metadata,indent=2))

        def sample(value, indices):
            if hasattr(value,'to_local'):
                value=value.to_local()
            index=torch.tensor(indices,device=value.device,dtype=torch.long)
            return value.detach().reshape(-1).index_select(0,index).float().cpu().tolist()

        def attach(leaf):
            tracked=[]
            for group in leaf.param_groups:
                for parameter in group['params']:
                    name=names[id(parameter)]
                    if name in selected:
                        tracked.append((group,parameter,name))
            if not tracked:
                return
            step=leaf.step
            calls=0

            @wraps(step)
            def observed(*args,**kw):
                nonlocal calls
                calls+=1
                if calls>5:
                    raise ValueError('Trace is limited to201-205')
                records=[]
                for group,parameter,name in tracked:
                    slots=leaf.state[parameter]
                    if parameter.grad is None:
                        raise ValueError('Missing vocabulary gradient')
                    record={'step':200+calls,'name':name,'indices':selected[name],
                            'leaf':type(leaf).__name__,'group_step_before':group.get('step'),
                            'lr':group['lr'],'betas':list(group['betas']),
                            'eps':group['eps'],'weight_decay':group['weight_decay'],
                            'gradient':sample(parameter.grad,selected[name]),
                            'before':{'weight':sample(parameter,selected[name]),
                                      'exp_avg':sample(slots['exp_avg'],selected[name]),
                                      'exp_avg_sq':sample(slots['exp_avg_sq'],selected[name])}}
                    records.append(record)
                result=step(*args,**kw)
                for record,(group,parameter,name) in zip(records,tracked):
                    slots=leaf.state[parameter]
                    record['group_step_after']=group.get('step')
                    record['after']={'weight':sample(parameter,selected[name]),
                                     'exp_avg':sample(slots['exp_avg'],selected[name]),
                                     'exp_avg_sq':sample(slots['exp_avg_sq'],selected[name])}
                    with output.open('a') as stream:
                        stream.write(json.dumps(record)+'\n')
                return result
            leaf.step=observed
        for leaf in leaves(self.trainer.optimizer):
            attach(leaf)
    Observer.on_train_begin=begin
