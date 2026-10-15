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

"""Read the actual BF16 head GEMM branches, accounting for chunk-loss recomputation at207."""
import hashlib
import json
from pathlib import Path
from types import MethodType

import numpy as np
import torch
from torch.nn import functional

import capture_hp_native
from backward_precision_builder import SharedHead

STATE = Path('/state/jt-recipe-20261014')
CONTEXT = {}


def save_tensor(value: torch.Tensor, call: int, kind: str) -> None:
    """Preserve exact BF16 storage bits without modifying the original tensor."""
    if value.dtype != torch.bfloat16:
        raise ValueError('Expected native BF16 head boundary')
    directory = STATE / CONTEXT['output'].name / 'branches207'
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ('call' + str(call) + '-' + kind + '.npy')
    if path.exists():
        raise FileExistsError(path)
    array = value.detach().cpu().contiguous().view(torch.int16).numpy().view(np.uint16)
    np.save(path, array, allow_pickle=False)
    record = {'step': 207, 'call': call, 'kind': kind, 'path': str(path), 'shape': list(array.shape),
              'logical_dtype': 'bfloat16', 'storage_dtype': 'uint16', 'bytes': path.stat().st_size,
              'sha256': hashlib.sha256(memoryview(array).cast('B')).hexdigest()}
    with (CONTEXT['output'] / 'HEAD_BRANCHES207.jsonl').open('a') as stream:
        stream.write(json.dumps(record) + '\n')


def install() -> None:
    """Retain the original separate BF16 casts and native F.linear/autograd calculation."""
    original_begin = capture_hp_native.Observer.on_train_begin
    original_end = capture_hp_native.Observer.on_step_end

    def begin(self, state, **kwargs):
        original_begin(self, state, **kwargs)
        CONTEXT.update(output=self.output, step=int(state.global_step) + 1, calls=0)
        module = self.trainer.model.lm_head
        if not isinstance(module, SharedHead) or module.bias is not None or module.alignment_native_gradient:
            raise ValueError('Unexpected native F.linear head configuration')

        def forward(head, values):
            weight = head.weight.to(values.dtype)
            if CONTEXT['step'] != 207:
                return functional.linear(values, weight)
            CONTEXT['calls'] += 1
            call = CONTEXT['calls']
            if call > 8:
                raise ValueError('Unexpected unbounded head recomputation')
            save_tensor(values.reshape(-1, values.shape[-1]), call, 'x')

            def weight_hook(gradient):
                save_tensor(gradient, call, 'dw')
                return gradient

            def output_hook(gradient):
                save_tensor(gradient.reshape(-1, gradient.shape[-1]), call, 'dy')
                return gradient

            weight.register_hook(weight_hook)
            result = functional.linear(values, weight)
            result.register_hook(output_hook)
            return result

        module.forward = MethodType(forward, module)
        report = {'step': 207, 'read_only_observation': True, 'formula': 'unchanged separate BF16 cast + F.linear',
                  'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'original_source_sha256': hashlib.sha256(
                      (Path(__file__).parent / 'backward_precision_builder.py').read_bytes()).hexdigest()}
        (self.output / 'HEAD_BRANCH_OBSERVER.json').write_text(json.dumps(report, indent=2))

    def end(self, state, **kwargs):
        original_end(self, state, **kwargs)
        if int(state.global_step) == 207:
            rows = [json.loads(line) for line in (self.output / 'HEAD_BRANCHES207.jsonl').read_text().splitlines()]
            counts = {kind: sum(row['kind'] == kind for row in rows) for kind in ['x', 'dy', 'dw']}
            (self.output / 'HEAD_BRANCH_CALLS.json').write_text(json.dumps(
                {'forward_calls': CONTEXT['calls'], 'saved_counts': counts}, indent=2))
            if counts['dy'] != 2 or counts['dw'] != 2:
                raise ValueError('Expected exactly two actual head gradient hooks')
        CONTEXT['step'] = int(state.global_step) + 1

    capture_hp_native.Observer.on_train_begin = begin
    capture_hp_native.Observer.on_step_end = end
