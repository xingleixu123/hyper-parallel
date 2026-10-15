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

"""Select a explicit NZ MatMulV2 output-head dweight route with a separate mm wrapper control."""
import hashlib
import json
import os
from pathlib import Path
from types import MethodType

import torch
import torch_npu
from torch.nn import functional

import capture_hp_native
from backward_precision_builder import SharedHead
from head_gemm_aclop_nz_v3 import matmul as legacy_matmul

CONTEXT = {}


class SelectedHeadGradient(torch.autograd.Function):
    """Keep F.linear forward and native dinput; select only the dweight matrix multiplication."""

    @staticmethod
    def forward(ctx, values, weight, mode):
        ctx.save_for_backward(values, weight)
        ctx.mode = mode
        return functional.linear(values, weight)

    @staticmethod
    def backward(ctx, gradient):
        values, weight = ctx.saved_tensors
        dy = gradient.reshape(-1, gradient.shape[-1])
        inputs = values.reshape(-1, values.shape[-1])
        dx = torch.mm(dy, weight).reshape_as(values) if ctx.needs_input_grad[0] else None
        dw = None
        if ctx.needs_input_grad[1]:
            if ctx.mode == 'mm':
                dw = torch.mm(dy.T, inputs)
            elif ctx.mode == 'legacy_nz':
                torch_npu.npu.set_compile_mode(jit_compile=True)
                torch.npu.config.allow_internal_format = True
                try:
                    dw, metadata = legacy_matmul(dy.contiguous(), inputs.contiguous())
                    CONTEXT['last_storage_metadata'] = metadata
                finally:
                    torch_npu.npu.set_compile_mode(jit_compile=False)
                    torch.npu.config.allow_internal_format = False
            else:
                raise ValueError('Unknown head gradient route')
        CONTEXT['calls'] += 1
        if dy.dtype != torch.bfloat16 or dw is None or dw.dtype != torch.bfloat16:
            raise ValueError('Expected both actual BF16 head gradient uses')
        return dx, dw, None


def install() -> None:
    """Use real live tensors at every resumed step, without teacher values or selected coordinates."""
    mode = os.environ['P12_HEAD_DWEIGHT_ROUTE']
    if mode not in ('mm', 'legacy_nz'):
        raise ValueError('Expected mm control or explicit legacy NZ candidate')
    original_begin = capture_hp_native.Observer.on_train_begin
    original_end = capture_hp_native.Observer.on_step_end

    def begin(self, state, **kwargs):
        original_begin(self, state, **kwargs)
        module = self.trainer.model.lm_head
        if not isinstance(module, SharedHead) or module.bias is not None or module.alignment_native_gradient:
            raise ValueError('Unexpected head configuration')
        CONTEXT.update(calls=0, initial_step=int(state.global_step), output=self.output)

        def forward(head, values):
            return SelectedHeadGradient.apply(values, head.weight.to(values.dtype), mode)

        module.forward = MethodType(forward, module)
        report = {'mode': mode, 'teacher_values_used': False, 'all_resumed_steps': True,
                  'coordinate_or_step_special_case': False, 'forward': 'unchanged F.linear',
                  'dinput': 'dy @ weight', 'dweight': 'dy.T @ x' if mode == 'mm' else 'explicit MatMulV2(NZ(dy),NZ(x))->NZ, transpose_x1=True, output->ND',
                  'scope': 'Independent head dweight arithmetic-route ablation; not a universal correctness fix',
                  'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'helper_sha256': {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                                    for name in ['head_gemm_aclop_nz_v3.py', 'head_gemm_aclop_v2.py']},
                  'validation': 'head207-gemm-aclop-v2-nz-replay-r3: both entire outputs exact MF',
                  'jit_and_internal_format_scope': 'Enabled only around dweight and restored False in finally'}
        (self.output / 'HEAD_GEMM_ROUTE.json').write_text(json.dumps(report, indent=2))

    def end(self, state, **kwargs):
        original_end(self, state, **kwargs)
        expected = (int(state.global_step) - CONTEXT['initial_step']) * 2
        if CONTEXT['calls'] != expected:
            raise ValueError('Expected exactly two real head backward calls per training step')
        with (self.output / 'HEAD_GEMM_ROUTE_CALLS.jsonl').open('a') as stream:
            stream.write(json.dumps({'step': int(state.global_step), 'mode': mode,
                                     'cumulative_backward_calls': CONTEXT['calls'],
                                     'storage_metadata': CONTEXT.get('last_storage_metadata')}) + '\n')

    capture_hp_native.Observer.on_train_begin = begin
    capture_hp_native.Observer.on_step_end = end
