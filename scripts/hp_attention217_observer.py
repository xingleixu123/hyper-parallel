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

"""Observe the exact layer0 NPU attention invocation at independent step217."""
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch_npu

import capture_hp_native


def install() -> None:
    """Record live kernel operands and options without changing any return value."""
    original_begin = capture_hp_native.Observer.on_micro_step_begin
    original_kernel = torch_npu.npu_fusion_attention
    context = {'active': False, 'attached': False, 'calls': 0}

    def save(name: str, tensor: torch.Tensor) -> dict:
        array = np.ascontiguousarray(tensor.detach().float().cpu().numpy())
        filename = name + '.npy'
        np.save(context['directory'] / filename, array, allow_pickle=False)
        return {'name': name, 'file': filename, 'shape': list(array.shape), 'dtype': str(tensor.dtype),
                'stride': list(tensor.stride()), 'sha256': hashlib.sha256(array.tobytes()).hexdigest()}

    def kernel(query, key, value, head_num, input_layout, **kwargs):
        if not context['active']:
            return original_kernel(query, key, value, head_num, input_layout, **kwargs)
        context['calls'] += 1
        if context['calls'] != 1:
            raise ValueError('Expected one layer0 attention forward call')
        records = [save(name, tensor) for name, tensor in [('query', query), ('key', key), ('value', value)]]
        metadata = {'head_num': head_num, 'input_layout': input_layout}
        for name, argument in kwargs.items():
            if isinstance(argument, torch.Tensor):
                records.append(save(name, argument))
            else:
                metadata[name] = argument
        result = original_kernel(query, key, value, head_num, input_layout, **kwargs)
        records.append(save('output', result[0]))
        (context['output'] / 'ATTENTION217.json').write_text(json.dumps({
            'step': 217, 'records': records, 'options': metadata, 'readonly': True,
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2))
        return result

    def begin(self: Any, state: Any, micro_batch: Any, **kwargs: Any) -> None:
        original_begin(self, state, micro_batch, **kwargs)
        if int(state.global_step) != 216 or context['attached']:
            return
        context['attached'] = True
        directory = Path('/state/jt-recipe-20261014') / self.output.name / 'attention'
        directory.mkdir(parents=True, exist_ok=False)
        context.update(directory=directory, output=self.output)
        attention = dict(self.trainer.model.named_modules())['model.layers.0.self_attn']
        native = attention.attention_interface

        def observe(owner, query, key, value, mask, **options):
            context['active'] = True
            try:
                return native(owner, query, key, value, mask, **options)
            finally:
                context['active'] = False
        attention.attention_interface = observe

    torch_npu.npu_fusion_attention = kernel
    capture_hp_native.Observer.on_micro_step_begin = begin
