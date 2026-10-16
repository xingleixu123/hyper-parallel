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

"""Isolated compatibility ablation for the measured archived DVM reduction tail."""
import hashlib
import inspect
import json
from pathlib import Path

import torch

from capture_hp_native import Observer
import ce_sum_tree_factor as factor

CONTEXT = {'calls': 0}
VECTOR_LANES = 40


def tail_join(values: torch.Tensor, sequential: torch.Tensor) -> torch.Tensor:
    """Use the validated even/odd join on the incomplete40-row dispatch group."""
    if values.shape != (4096, 22, 1) or values.dtype != torch.float32:
        raise ValueError('Tail compatibility is validated only for4096x152064 FP32 CE')
    CONTEXT['calls'] += 1
    regular = values.shape[0] // VECTOR_LANES * VECTOR_LANES
    tail = values[regular:]
    even, odd = tail[:, 0, :], tail[:, 1, :]
    for tile in range(2, 22, 2):
        even = even + tail[:, tile, :]
        odd = odd + tail[:, tile + 1, :]
    return torch.cat((sequential[:regular], even + odd), dim=0)


def install() -> None:
    """Change only the tile join; retain every existing intra-tile operation and source file."""
    original_source = inspect.getsource(factor.tiled_sum)
    if original_source.count('    return total') != 1:
        raise ValueError('Frozen CE tree source changed')
    source = original_source.replace('    return total', '    return ce_tail226_join(values, total)')
    factor.__dict__['ce_tail226_join'] = tail_join
    exec(compile(source, __file__ + ':tiled_sum', 'exec'), factor.__dict__)
    original_begin, original_end = Observer.on_train_begin, Observer.on_step_end

    def begin(self, state, **kwargs):
        original_begin(self, state, **kwargs)
        CONTEXT['calls'] = 0
        (self.output / 'CE_TAIL226_FACTOR.json').write_text(json.dumps({
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'base_source_sha256': hashlib.sha256(Path(factor.__file__).read_bytes()).hexdigest(),
            'teacher_values_used': False, 'step_specific_behavior': False,
            'validated_shape': [4096, 152064], 'vector_dispatch_group': VECTOR_LANES,
            'ordinary_join': '22 tiles accumulated sequentially',
            'incomplete_group_join': 'even and odd tiles separately accumulated, then added',
            'classification': 'Archived MF graph/CANN8.5/910B3 compatibility experiment; not a universal default',
            'original_source': original_source, 'instrumented_source': source}, indent=2))

    def end(self, state, **kwargs):
        original_end(self, state, **kwargs)
        if CONTEXT['calls'] != 2:
            raise ValueError('Expected two independently computed CE backward calls')
        with (self.output / 'CE_TAIL226_CALLS.jsonl').open('a') as stream:
            stream.write(json.dumps({'step': int(state.global_step), 'calls': CONTEXT['calls']}) + '\n')
        CONTEXT['calls'] = 0

    Observer.on_train_begin, Observer.on_step_end = begin, end
