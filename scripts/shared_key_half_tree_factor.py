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

"""Isolate shared-key head reduction using live gradients and an FP32 half tree."""
import hashlib
import inspect
import json
import textwrap
from pathlib import Path

import torch

from capture_hp_native import Observer
from hyper_parallel.models.jt_deepseek_v3 import modeling_jt_deepseek_v3 as modeling


class SharedKeyHalfTree(torch.autograd.Function):
    """Preserve expand forward while using a balanced FP32 backward sum."""

    @staticmethod
    def forward(ctx, value, heads):
        if value.ndim != 4 or value.shape[1] != 1 or heads < 1 or heads & (heads - 1):
            raise ValueError('Expected singleton shared head and power-of-two head count')
        return value.expand(-1, heads, -1, -1)

    @staticmethod
    def backward(ctx, gradient):
        reduced = gradient.float()
        while reduced.shape[1] > 1:
            half = reduced.shape[1] // 2
            reduced = reduced[:, :half] + reduced[:, half:]
        return reduced.to(gradient.dtype), None


def install() -> None:
    """Replace only the expand in the actual MLA projection method."""
    owners = [value for value in vars(modeling).values()
              if isinstance(value, type) and value.__module__ == modeling.__name__
              and '_project_attention_inputs' in value.__dict__]
    needle = 'key_rope.expand(-1, self.num_heads, -1, -1)'
    matches = [(owner, inspect.getsource(owner._project_attention_inputs)) for owner in owners]
    matches = [(owner, source) for owner, source in matches if needle in source]
    if len(matches) != 1:
        raise ValueError('Expected one live shared-key projection method')
    owner, source = matches[0]
    if source.count(needle) != 1:
        raise ValueError('Ambiguous key expansion')
    scope = dict(owner._project_attention_inputs.__globals__)
    scope['shared_key_half_tree'] = SharedKeyHalfTree.apply
    replacement = textwrap.dedent(source).replace(needle, 'shared_key_half_tree(key_rope, self.num_heads)')
    exec(compile(replacement, __file__ + ':projection', 'exec'), scope)
    owner._project_attention_inputs = scope['_project_attention_inputs']
    previous = Observer.on_train_begin

    def begin(self, state, **kwargs):
        previous(self, state, **kwargs)
        report = {'teacher_values_used': False, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'owner': owner.__name__, 'scope': 'all MLA layers, all steps', 'forward': 'unchanged expand view',
                  'backward': 'FP32 half-tree head sum then original dtype'}
        (self.output / 'SHARED_KEY_REDUCTION.json').write_text(json.dumps(report, indent=2))

    Observer.on_train_begin = begin
