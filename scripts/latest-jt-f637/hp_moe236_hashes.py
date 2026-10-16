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

"""Read-only MTP MoE branch and routing derivative hashes at236."""
import hashlib
import json
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any

import numpy as np
import torch

from capture_hp_native import Observer
from mindspore272_stat_hash import native_dump_sha1
from moe_backward_builder import RouterRoutedInputPair, WeightedProductGradient
from router_normalization_join_factor import RoutingNormalization
from router_sigmoid_arithmetic_factor import ArithmeticSigmoid
from router_weight_contiguous_factor import RouterLinear

CONTEXT: dict[str, Any] = {'active': False, 'records': {}, 'attached': False}


def save(name: str, value: torch.Tensor) -> None:
    """Hash a logical tensor without retaining any full array."""
    if value is None:
        return
    native = value.detach().contiguous().cpu()
    array = native.view(torch.uint16).numpy() if native.dtype == torch.bfloat16 else native.numpy()
    fp32 = np.ascontiguousarray(native.float().numpy())
    record = {'name': name, 'shape': list(value.shape), 'dtype': str(value.dtype),
              'native_dump_sha1': native_dump_sha1(array), 'fp32_native_dump_sha1': native_dump_sha1(fp32),
              'fp32_sha256': hashlib.sha256(fp32.tobytes()).hexdigest()}
    prior = CONTEXT['records'].get(name)
    if prior is not None and prior != record:
        raise ValueError('Recomputed MTP boundary changed: ' + name)
    CONTEXT['records'][name] = record


class InputBoundary(torch.autograd.Function):
    """Keep the branch input gradient separate from other uses of the same tensor."""

    @staticmethod
    def forward(ctx: Any, value: torch.Tensor, name: str) -> torch.Tensor:
        ctx.name = name
        return value

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        save(ctx.name, gradient)
        return gradient, None


def wrap_function(cls: Any, label: str, inputs: list[str], outputs: list[str],
                  upstream: list[str], downstream: list[str]) -> None:
    """Preserve original functions and unpack checkpoint saved tensors only once."""
    native_forward, native_backward = cls.forward, cls.backward

    def forward(ctx: Any, *args: Any) -> Any:
        ctx.moe236_active = CONTEXT['active']
        result = native_forward(ctx, *args)
        if ctx.moe236_active:
            for name, value in zip(inputs, args):
                if name:
                    save(label + '.' + name, value)
            for name, value in zip(outputs, result if isinstance(result, tuple) else (result,)):
                if name:
                    save(label + '.' + name, value)
        return result

    def backward(ctx: Any, *gradients: Any) -> Any:
        if not ctx.moe236_active:
            return native_backward(ctx, *gradients)
        saved = ctx.saved_tensors
        proxy = SimpleNamespace(**ctx.__dict__, saved_tensors=saved, needs_input_grad=ctx.needs_input_grad)
        result = native_backward(proxy, *gradients)
        for name, value in zip(upstream, gradients):
            if name:
                save(label + '.' + name, value)
        for name, value in zip(downstream, result if isinstance(result, tuple) else (result,)):
            if name:
                save(label + '.' + name, value)
        return result

    cls.forward, cls.backward = staticmethod(forward), staticmethod(backward)


def install() -> None:
    """Attach only at the requested step after all existing compatibility factors."""
    original_begin, original_end = Observer.on_micro_step_begin, Observer.on_step_end

    def begin(self: Any, state: Any, micro_batch: Any, **kwargs: Any) -> None:
        original_begin(self, state, micro_batch, **kwargs)
        if int(state.global_step) != 235 or CONTEXT['attached']:
            return
        CONTEXT['attached'] = True
        wrap_function(WeightedProductGradient, 'weighted', ['outputs', 'weights'], ['result'],
                      ['dy'], ['outputs_dy', 'weights_dy'])
        wrap_function(RouterLinear, 'router', ['x', 'weight'], ['logits'], ['dy'], ['dx', 'dw'])
        wrap_function(RoutingNormalization, 'normalization', ['scores', 'indices'], ['coefficients', 'auxiliary'],
                      ['coefficients_dy', 'auxiliary_dy'], ['dx'])
        wrap_function(ArithmeticSigmoid, 'sigmoid', ['logits'], ['scores'], ['dy'], ['dx'])
        wrap_function(RouterRoutedInputPair, 'pair', ['x'], ['expert_x', 'router_x'],
                      ['expert_dx', 'router_dx'], ['joined_dx'])
        module = dict(self.trainer.model.named_modules())['mtp.layers.0.transformer_layer.mlp']
        native = module.forward

        def forward(module_self: Any, values: Any) -> Any:
            del module_self
            values = InputBoundary.apply(values, 'moe.dx')
            save('moe.x', values)
            CONTEXT['active'] = True
            try:
                result = native(values)
            finally:
                CONTEXT['active'] = False
            save('moe.y', result)
            def gradient_hook(gradient: torch.Tensor) -> torch.Tensor:
                save('moe.dy', gradient)
                return gradient
            result.register_hook(gradient_hook)
            return result

        module.forward = MethodType(forward, module)
        shared = module.shared_experts
        native_shared = shared.forward

        def shared_forward(module_self: Any, values: Any) -> Any:
            del module_self
            return native_shared(InputBoundary.apply(values, 'shared.dx'))

        shared.forward = MethodType(shared_forward, shared)
        (self.output / 'MOE236_OBSERVER.json').write_text(json.dumps({
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'hash_helper_sha256': hashlib.sha256(Path(__file__).with_name('mindspore272_stat_hash.py').read_bytes()).hexdigest(),
            'raw_arrays_written': False, 'step': 236, 'scope': 'MTP MoE only; native returned values unchanged'}, indent=2))

    def end(self: Any, state: Any, **kwargs: Any) -> None:
        original_end(self, state, **kwargs)
        if int(state.global_step) == 236:
            required = ['moe.dx', 'moe.dy', 'shared.dx', 'router.dw', 'weighted.weights_dy',
                        'normalization.dx', 'sigmoid.dx', 'pair.joined_dx']
            if not all(name in CONTEXT['records'] for name in required):
                raise ValueError('Missing MTP branch observations: ' + str(CONTEXT['records'].keys()))
            (self.output / 'MOE236_HASHES.json').write_text(json.dumps({
                'complete': True, 'step': 236, 'records': list(CONTEXT['records'].values())}, indent=2))

    Observer.on_micro_step_begin, Observer.on_step_end = begin, end
