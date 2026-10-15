"""Diagnostic only: deliver FP32 expert wgrad to FSDP's existing accumulation buffer.

Forward values and input-gradient kernels remain the original BF16 operations.
No production interface is added here. This deliberately narrow experiment
requires JT's direct transposed parameter view and fails if that contract changes.
"""

import importlib
import json
import os
import weakref
from pathlib import Path

import torch
import torch_npu

from hyper_parallel.core.fully_shard.hsdp_param import HSDPParamV2 as HSDPParam


def install_fp32_expert_wgrad() -> None:
    """Attach a diagnostic callback to stable unsharded parameters and the JT GMM."""
    original_init = HSDPParam.init_unsharded_param
    traced = set()
    output = Path(os.environ['JT_CASE_OUTPUT'])
    rank = int(os.environ['RANK'])

    def initialize_with_accumulator(self):
        original_init(self)
        state_reference = weakref.ref(self)

        @torch.no_grad()
        def accumulate(gradient):
            state = state_reference()
            if state is None:
                raise RuntimeError('FSDP parameter owner expired during backward')
            if gradient.dtype != torch.float32 or tuple(gradient.shape) != tuple(state._orig_size):
                raise ValueError('FP32 expert gradient does not match its FSDP local parameter')
            if state.reduce_comm_dtype(gradient) != torch.float32:
                raise ValueError('The diagnostic requires FP32 gradient reduction')
            if state.unsharded_accumulated_grad is None:
                state.unsharded_accumulated_grad = gradient
            else:
                state.unsharded_accumulated_grad.add_(gradient)
            identity = id(state)
            if identity not in traced:
                traced.add(identity)
                with (output / f'fp32-wgrad-delivery-rank{rank}.jsonl').open('a') as handle:
                    handle.write(json.dumps({'parameter': state._module_info.param_name,
                        'shape': list(gradient.shape), 'gradient_dtype': str(gradient.dtype),
                        'weight_dtype': str(state.unsharded_param.dtype),
                        'reduce_dtype': str(state.reduce_comm_dtype(gradient))}) + '\n')

        self.unsharded_param._jt_diagnostic_accumulate_fp32 = accumulate

    class FP32WeightGradient(torch.autograd.Function):
        """Preserve JT's original GMM forward/dgrad and change only wgrad storage."""

        @staticmethod
        def forward(ctx, inputs, weight, group_list):
            owner = weight._base
            accumulate = getattr(owner, '_jt_diagnostic_accumulate_fp32', None)
            if not callable(accumulate):
                raise ValueError('JT GMM weight is not a direct view of an observed FSDP parameter')
            if inputs.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
                raise ValueError('This diagnostic requires BF16 GMM inputs and weights')
            if tuple(owner.shape) != tuple(weight.transpose(-1, -2).shape):
                raise ValueError('JT expert weight view has an unexpected layout')
            ctx.accumulate = accumulate
            ctx.save_for_backward(inputs, weight, group_list)
            return torch_npu.npu_grouped_matmul(
                x=[inputs], weight=[weight], bias=[], group_list=group_list,
                split_item=3, group_type=0, group_list_type=0)[0]

        @staticmethod
        def backward(ctx, grad_output):
            inputs, weight, group_list = ctx.saved_tensors
            grad_output = grad_output.contiguous()
            grad_inputs = torch_npu.npu_grouped_matmul(
                x=[grad_output], weight=[weight.transpose(-1, -2)], bias=[],
                group_list=group_list, split_item=3, group_type=0, group_list_type=0)[0]
            gradient = torch.zeros(weight.shape, dtype=torch.float32, device=weight.device)
            if inputs.numel():
                torch_npu.npu_grouped_matmul_add_(
                    gradient, inputs, grad_output, group_list,
                    transpose_x=True, transpose_weight=False, group_type=2)
            ctx.accumulate(gradient.transpose(-1, -2).contiguous())
            # A zero autograd gradient preserves normal FSDP backward scheduling;
            # the real FP32 contribution is already in its accumulation buffer.
            return grad_inputs, torch.zeros_like(weight), None

    HSDPParam.init_unsharded_param = initialize_with_accumulator
    module = importlib.import_module('hyper_parallel.components.functional.npu_grouped_swiglu')
    module._NPU_GROUPED_MATMUL = FP32WeightGradient
