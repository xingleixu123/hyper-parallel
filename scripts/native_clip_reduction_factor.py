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

"""Use the measured archive reduction order on live single-card gradients only."""
import json
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
import capture_hp_native
import hyper_parallel.trainer.base as trainer_base
from hyper_parallel.core.utils.clip_grad import _get_local_grad, _clip_grads_with_norm_
from observe_hp_preclip_norm_adjacent import mf_gradient, dvm_square_sum, sequential


def install():
    order = json.loads(Path('/alignment/phase12/configs/mf-gradient-layout-order.json').read_text())
    context = {'output': None, 'step': 0}
    original_begin = capture_hp_native.Observer.on_train_begin

    def begin(observer, state, **kwargs):
        original_begin(observer, state, **kwargs)
        context['output'] = observer.output / 'CLIP_REDUCTION.jsonl'
        context['step'] = int(state.global_step)
        (observer.output / 'CLIP_REDUCTION_POLICY.json').write_text(json.dumps({
            'teacher_values_used': False, 'world_size': 1,
            'policy': 'MF logical layouts; FP32 square; native SumX tiles; half-row and adjacent SIMD lanes; 40strided cores sequential;46 tensors sequential',
            'scope': 'Archive arithmetic compatibility diagnostic, not a generic distributed clipping default'},indent=2))

    @torch.no_grad()
    def clip(model, max_norm, norm_type=2.0, error_if_nonfinite=False, foreach=None):
        if not isinstance(model, torch.nn.Module) or float(norm_type) != 2.0 or dist.get_world_size() != 1:
            raise ValueError('Measured clipping factor supports this single-card model and L2 only')
        parameters = dict(model.named_parameters())
        rows = []
        for entry in order:
            value = mf_gradient(parameters, entry['name'])
            if list(value.shape) != entry['shape']:
                raise ValueError('Unexpected gradient layout: '+entry['name'])
            rows.append(float(dvm_square_sum(value)))
            del value
        total = sequential(np.asarray(rows,dtype=np.float32))
        gradients = [_get_local_grad(parameter) for parameter in parameters.values()]
        if any(g is None or g.dtype != torch.float32 for g in gradients):
            raise ValueError('Expected complete FP32 local gradients')
        norm = torch.tensor(np.sqrt(total),dtype=torch.float32,device=gradients[0].device)
        if error_if_nonfinite and not np.isfinite(total):
            raise ValueError('Nonfinite gradient norm')
        _clip_grads_with_norm_(gradients,float(max_norm),norm,foreach)
        context['step'] += 1
        record = {'step':context['step'],'square_sum':float(total),'norm':float(np.sqrt(total)),
                  'per_parameter_square_sums':rows,'teacher_values_used':False}
        with context['output'].open('a') as stream:
            stream.write(json.dumps(record)+'\n')
        return norm

    capture_hp_native.Observer.on_train_begin = begin
    trainer_base.clip_grad_norm_ = clip
