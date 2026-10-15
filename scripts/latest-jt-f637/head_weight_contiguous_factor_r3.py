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

"""Isolate the shared output head dweight GEMM storage on live operands."""
import hashlib
import json
from pathlib import Path
from types import MethodType
import numpy as np
import torch
from torch.nn import functional
from capture_hp_native import Observer
from backward_precision_builder import SharedHead


OUTPUT = None
CALLS = 0
SELECTED = None


class ContiguousHead(torch.autograd.Function):
    """Keep native forward and dinput; materialize only transposed dweight input."""
    @staticmethod
    def forward(ctx, values, weight):
        ctx.save_for_backward(values, weight)
        return functional.linear(values, weight)

    @staticmethod
    def backward(ctx, gradient):
        global CALLS
        CALLS += 1
        values, weight = ctx.saved_tensors
        grad = gradient.reshape(-1, gradient.shape[-1])
        inputs = values.reshape(-1, values.shape[-1])
        dx = torch.mm(grad, weight).reshape_as(values) if ctx.needs_input_grad[0] else None
        dw = torch.mm(grad.T.contiguous(), inputs) if ctx.needs_input_grad[1] else None
        indices = torch.tensor(SELECTED, device=dw.device, dtype=torch.long)
        record = {'call': CALLS, 'dy_shape': list(grad.shape), 'dy_stride': list(grad.stride()),
                  'dw_at_coordinates': dw.reshape(-1).index_select(0, indices).float().cpu().tolist()}
        with (OUTPUT/'HEAD_BACKWARD_COORDINATES.jsonl').open('a') as stream:
            stream.write(json.dumps(record)+'\n')
        if CALLS <= 2:
            vocab = sorted({index // inputs.shape[-1] for index in SELECTED})
            hidden = sorted({index % inputs.shape[-1] for index in SELECTED})
            np.savez_compressed(OUTPUT/f'head-operands-call{CALLS}.npz',
                vocabulary_rows=np.asarray(vocab), hidden_columns=np.asarray(hidden),
                dy=grad.index_select(1, torch.tensor(vocab, device=grad.device)).float().cpu().numpy(),
                x=inputs.index_select(1, torch.tensor(hidden, device=inputs.device)).float().cpu().numpy())
        return dx, dw


def forward(self, values):
    return ContiguousHead.apply(values, self.weight.to(values.dtype))


def install():
    original = Observer.on_train_begin
    def begin(self, state, **kwargs):
        global OUTPUT, SELECTED
        original(self, state, **kwargs)
        OUTPUT = self.output
        coordinates = json.loads(Path('/alignment/phase12/evidence/hp-denseadj-vocab204-coordinates-coordinates.json').read_text())
        SELECTED = [item['flat_index'] for row in coordinates['records'] if row['name'] == 'lm_head.weight' for item in row['coordinates']]
        module = self.trainer.model.lm_head
        if not isinstance(module, SharedHead) or module.bias is not None:
            raise ValueError('Unexpected output head '+repr(type(module)))
        module.forward = MethodType(forward, module)
        (self.output/'HEAD_WEIGHT_GRADIENT.json').write_text(json.dumps({
            'teacher_values_used': False, 'scope': 'head dweight only', 'runtime_class': str(type(module)),
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'forward': 'F.linear', 'dx': 'dy@weight', 'dw': 'dy.T.contiguous()@x'}, indent=2))
    Observer.on_train_begin = begin
