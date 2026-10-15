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

"""Observe selected complete head operand rows around all226 dinput mismatches."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

from capture_hp_native import Observer
import hp_head_gemm_legacy_nz_factor as factor

CONTEXT = {'step': 201, 'calls': 0, 'records': []}


def save(call, kind, tensor):
    """Retain only226 observations and preserve each original live tensor."""
    if CONTEXT['step'] != 226:
        return
    array = tensor.detach().float().cpu().numpy()
    name = f'call{call}_{kind}.npy'
    path = CONTEXT['directory'] / name
    if path.exists():
        if not np.array_equal(np.load(path, mmap_mode='r'), array):
            raise ValueError('Recomputed head operand differs')
        return
    np.save(path, array)
    CONTEXT['records'].append({'call': call, 'kind': kind, 'file': name,
                               'shape': list(array.shape), 'dtype': str(tensor.dtype),
                               'sha256': hashlib.sha256(array.tobytes()).hexdigest()})


def save_full(name, tensor):
    """Preserve actual BF16 bits for a subsequent full-shape operator replay."""
    if tensor.dtype != torch.bfloat16:
        raise ValueError('Expected actual BF16 matmul operands/output')
    array = tensor.detach().contiguous().view(torch.uint16).cpu().numpy()
    digest = hashlib.sha256(array.tobytes()).hexdigest()
    path = CONTEXT['directory'] / (name + '_bf16_bits.npy')
    previous = next((r for r in CONTEXT['full_records'] if r['name'] == name), None)
    if previous is not None:
        if previous['sha256'] != digest:
            raise ValueError('Shared consumed weight differs between branches')
        return
    np.save(path, array)
    CONTEXT['full_records'].append({'name': name, 'file': path.name, 'shape': list(array.shape),
                                    'sha256': digest, 'storage_dtype': 'uint16', 'logical_dtype': 'bfloat16'})


def install():
    """Wrap the already selected real head implementation without replacing its math."""
    original_begin = Observer.on_train_begin
    original_end = Observer.on_step_end
    original_forward = factor.SelectedHeadGradient.forward
    original_backward = factor.SelectedHeadGradient.backward

    def forward(ctx, values, weight, mode):
        result = original_forward(ctx, values, weight, mode)
        CONTEXT['forward_calls'] += 1
        if CONTEXT['step'] == 226:
            rows = CONTEXT['rows'].to(values.device)
            ctx.head226_logits = result.reshape(-1, 152064).index_select(0, rows).detach()
        return result

    def backward(ctx, gradient):
        values, weight = ctx.saved_tensors
        result = original_backward(ctx, gradient)
        CONTEXT['calls'] += 1
        call = CONTEXT['calls']
        if CONTEXT['step'] == 226:
            rows = CONTEXT['rows'].to(gradient.device)
            cols = CONTEXT['columns'].to(weight.device)
            save(call, 'x', values.reshape(-1, 5120).index_select(0, rows))
            save(call, 'logits', ctx.head226_logits)
            save(call, 'weight_columns', weight.index_select(1, cols))
            save(call, 'dy', gradient.reshape(-1, 152064).index_select(0, rows))
            save(call, 'dx', result[0].reshape(-1, 5120).index_select(0, rows))
            save_full('shared_weight', weight)
            save_full(f'call{call}_dy', gradient.reshape(-1, 152064))
            save_full(f'call{call}_dx', result[0].reshape(-1, 5120))
        return result

    def begin(self, state, **kwargs):
        original_begin(self, state, **kwargs)
        source = Path('/alignment/phase12/evidence/head226-selected-coordinates.json')
        selection = json.loads(source.read_text())
        directory = Path('/state/jt-recipe-20261014') / self.output.name / 'head'
        directory.mkdir(parents=True, exist_ok=False)
        CONTEXT.update(step=int(state.global_step) + 1, calls=0, forward_calls=0, records=[], full_records=[], output=self.output,
                       directory=directory, rows=torch.tensor(selection['token_rows']),
                       columns=torch.tensor(selection['hidden_columns']))
        (self.output / 'HEAD226_OBSERVER.json').write_text(json.dumps({
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'selection_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'selection': selection, 'teacher_values_used': False}, indent=2))

    def end(self, state, **kwargs):
        original_end(self, state, **kwargs)
        if CONTEXT['calls'] != 2:
            raise ValueError('Expected two original head backward calls: ' + str(CONTEXT['calls']))
        if int(state.global_step) == 226:
            if len(CONTEXT['records']) != 10:
                raise ValueError('Incomplete selected head observations')
            (self.output / 'HEAD226_OPERANDS.json').write_text(json.dumps({
                'step': 226, 'records': CONTEXT['records'], 'full_records': CONTEXT['full_records']}, indent=2))
        with (self.output / 'HEAD226_CALLS.jsonl').open('a') as stream:
            stream.write(json.dumps({'step': int(state.global_step), 'forward_calls': CONTEXT['forward_calls'],
                                     'backward_calls': CONTEXT['calls']}) + '\n')
        CONTEXT.update(step=int(state.global_step) + 1, calls=0, forward_calls=0)

    factor.SelectedHeadGradient.forward = staticmethod(forward)
    factor.SelectedHeadGradient.backward = staticmethod(backward)
    Observer.on_train_begin = begin
    Observer.on_step_end = end
