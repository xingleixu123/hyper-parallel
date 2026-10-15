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

"""Independent FP32 CE reduction preserving the measured archive DVM addition tree."""
import json

import torch
from torch.nn import functional

import capture_hp_native

ORIGINAL_CE = functional.cross_entropy


def tiled_sum(exponent: torch.Tensor) -> torch.Tensor:
    """Use the audited 22x6912, 108x64 tree; every add remains FP32 on the same device."""
    if exponent.ndim != 2 or exponent.shape[-1] != 152064 or exponent.dtype != torch.float32:
        raise ValueError('Measured DVM compatibility tree requires FP32 vocabulary152064')
    values = exponent.reshape(exponent.shape[0], 22, 108, 64)
    while values.shape[-2] > 1:
        count = values.shape[-2]
        half = count // 2
        result = values[..., :half, :] + values[..., half:half * 2, :]
        if count % 2:
            first = result[..., :1, :] + values[..., -1:, :]
            result = torch.cat((first, result[..., 1:, :]), dim=-2) if half > 1 else first
        values = result
    values = values.squeeze(-2)
    while values.shape[-1] > 1:
        values = values[..., 0::2] + values[..., 1::2]
    total = values[:, 0, :]
    for tile in range(1, 22):
        total = total + values[:, tile, :]
    return total


def log_probabilities(inputs: torch.Tensor) -> torch.Tensor:
    """Retain max/sub/exp/sum/log/sub order, with no reference-stage values."""
    shifted = inputs - inputs.max(dim=-1, keepdim=True).values
    total = tiled_sum(shifted.exp())
    return shifted - total.log()


class TreeBackwardCE(torch.autograd.Function):
    """Keep original forward loss and compute the derivative using independent graph arithmetic."""

    @staticmethod
    def forward(ctx, inputs, targets, ignore_index):
        ctx.save_for_backward(inputs, targets)
        ctx.ignore_index = ignore_index
        return ORIGINAL_CE(inputs, targets, reduction='sum', ignore_index=ignore_index)

    @staticmethod
    def backward(ctx, gradient):
        inputs, targets = ctx.saved_tensors
        result = log_probabilities(inputs).exp()
        valid = targets != ctx.ignore_index
        hot = torch.zeros_like(result).scatter_(1, targets.clamp_min(0)[:, None], 1)
        result = (result - hot) * (valid[:, None] * gradient)
        return result, None, None


def observed_shape_ce(inputs, targets, *args, **kwargs):
    """Guard the specific measured scalar-loss contract before choosing compatibility arithmetic."""
    if args or inputs.shape != (4096, 152064) or inputs.dtype != torch.float32:
        raise ValueError('CE compatibility factor requires the audited original loss shape/dtype')
    allowed = {'weight', 'reduction', 'ignore_index', 'label_smoothing'}
    if (set(kwargs) - allowed or kwargs.get('weight') is not None or kwargs.get('label_smoothing', 0) != 0
            or kwargs.get('reduction') != 'sum'):
        raise ValueError('Unexpected CE contract')
    return TreeBackwardCE.apply(inputs, targets, kwargs.get('ignore_index', -100))


def install() -> None:
    """Record an independent arithmetic factor, without imposing a one-batch reference restriction."""
    original = capture_hp_native.Observer.on_train_begin

    def begin(observer, state, **kwargs):
        original(observer, state, **kwargs)
        report = {'teacher_values_used': False, 'native_forward_loss_unchanged': True,
                  'sum_tree': '22sequential tiles; each108x64, raggedfirst-half fold, then adjacent64 reduction',
                  'runtime': 'PyTorch tensor arithmetic on existing CANN9.2 NPU',
                  'classification': 'Archive DVM numerical compatibility experiment, not a correctness default'}
        (observer.output/'CE_SUM_TREE.json').write_text(json.dumps(report, indent=2))
        print('[P12-CE-SUM-TREE] independent measured FP32 addition tree', flush=True)

    capture_hp_native.Observer.on_train_begin = begin
    functional.cross_entropy = observed_shape_ce
