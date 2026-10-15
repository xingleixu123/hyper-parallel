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

"""Controlled archive AdamW arithmetic; each run selects exactly one parameter scope."""
import hashlib
import json
import os
from pathlib import Path
import torch
import hyper_parallel.core.optimizer.adamw as adamw_module
import capture_hp_native


def install():
    scope = os.environ['P12_ADAMW_ARCHIVE_SCOPE']
    if scope not in ('embedding', 'head', 'all'):
        raise ValueError(scope)
    original = adamw_module.adamw
    original_begin = capture_hp_native.Observer.on_train_begin
    names = {}
    output = None
    coords = {}

    def begin(self, state, **kwargs):
        nonlocal output, coords
        original_begin(self, state, **kwargs)
        names.update({id(parameter): name for name, parameter in self.trainer.model.named_parameters()})
        output = self.output
        rows = json.loads(Path('/alignment/phase12/evidence/hp-denseadj-vocab204-coordinates-coordinates.json').read_text())['records']
        coords = {row['name']: [item['flat_index'] for item in row['coordinates']] for row in rows}
        (output/'ADAMW_ARCHIVE_ARITHMETIC.json').write_text(json.dumps({
            'scope': scope, 'teacher_values_used': False, 'chunk_elements': 1048576,
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'control': 'hp-cann92-jtf637-routercontig-denseadj-castall',
            'arithmetic': 'FP32 beta/lr/decay tensors; separate moment mul/add; addcmul variance; native archive update order'}, indent=2))

    def local(value):
        return value.to_local() if hasattr(value, 'to_local') else value

    def sample(value, indices):
        value = local(value)
        index = torch.tensor(indices, device=value.device, dtype=torch.long)
        return value.detach().reshape(-1).index_select(0, index).cpu().tolist()

    def selected(parameter):
        name = names.get(id(parameter))
        if name is None:
            raise ValueError('Unmapped AdamW parameter')
        return scope == 'all' or name == {'embedding': 'model.embed_tokens.weight', 'head': 'lm_head.weight'}.get(scope)

    @torch.no_grad()
    def archive(params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, step, **kwargs):
        keep = [index for index, parameter in enumerate(params) if not selected(parameter)]
        if keep:
            original([params[i] for i in keep], [grads[i] for i in keep], [exp_avgs[i] for i in keep],
                     [exp_avg_sqs[i] for i in keep], [max_exp_avg_sqs[i] for i in keep] if max_exp_avg_sqs else [],
                     step, **kwargs)
        for index, parameter in enumerate(params):
            if index in keep:
                continue
            if kwargs['amsgrad'] or kwargs['maximize']:
                raise ValueError('Ablation supports the fixed recipe only')
            values = [local(value) for value in (parameter, grads[index], exp_avgs[index], exp_avg_sqs[index])]
            if any(value.dtype != torch.float32 or not value.is_contiguous() for value in values):
                raise ValueError('Expected contiguous FP32 AdamW state')
            weight, grad, moment, variance = [value.view(-1) for value in values]
            scalar = lambda number: torch.tensor([number], device=weight.device, dtype=torch.float32)
            beta1, beta2 = scalar(kwargs['beta1']), scalar(kwargs['beta2'])
            lr, decay, eps = scalar(kwargs['lr']), scalar(kwargs['weight_decay']), scalar(kwargs['eps'])
            step_size = lr / (1 - beta1.pow(step))
            bc2 = 1 - beta2.pow(step)
            variance_scale = (1 - beta2).item()
            decay_scale = 1 - lr * decay
            for start in range(0, weight.numel(), 1048576):
                end = start + 1048576
                current_weight, current_grad = weight[start:end], grad[start:end]
                current_m, current_v = moment[start:end], variance[start:end]
                next_weight = current_weight * decay_scale
                current_m.copy_(current_m * beta1 + current_grad * (1 - beta1))
                current_v.copy_(torch.addcmul(current_v * beta2, current_grad, current_grad, value=variance_scale))
                denom = (current_v / bc2).sqrt() + eps
                current_weight.copy_(next_weight - (current_m / denom) * step_size)
            name = names[id(parameter)]
            if name in coords and step <= 206:
                record = {'step': step, 'name': name, 'indices': coords[name],
                          'gradient': sample(grads[index], coords[name]),
                          'after': {'weight': sample(parameter, coords[name]),
                                    'exp_avg': sample(exp_avgs[index], coords[name]),
                                    'exp_avg_sq': sample(exp_avg_sqs[index], coords[name])}}
                with (output/'VOCAB_OPTIMIZER_TRACE.jsonl').open('a') as stream:
                    stream.write(json.dumps(record)+'\n')
    adamw_module.adamw = archive
    capture_hp_native.Observer.on_train_begin = begin
