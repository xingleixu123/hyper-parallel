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

"""Independent same-shape router NS square-sum normalization factor."""
import functools
import hashlib
import json
from pathlib import Path

import torch_npu
import capture_hp_native
import muon_dense_adjacent_factor
import muon_jit_probe

CALLS = 0


def install():
    """Change only the live 8-by-5120 router NS normalization, at every resumed step."""
    original_install = muon_dense_adjacent_factor.install
    original_begin = capture_hp_native.Observer.on_train_begin
    original_end = capture_hp_native.Observer.on_step_end

    def begin(self, state, **kwargs):
        original_begin(self, state, **kwargs)
        report = {'teacher_values_used': False, 'target_shape': [8, 5120],
                  'scope': 'Both router matrices; square-sum-sqrt of actual FP32 NS input, then original final BF16 cast.',
                  'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        (self.output / 'ROUTER_NS_SQUARE_SUM_FACTOR.json').write_text(json.dumps(report, indent=2))

    def end(self, state, **kwargs):
        original_end(self, state, **kwargs)
        row = {'step': int(state.global_step), 'total_router_calls': CALLS}
        with (self.output / 'ROUTER_NS_SQUARE_SUM_CALLS.jsonl').open('a') as handle:
            handle.write(json.dumps(row) + '\n')

    def deferred_install():
        original_install()
        original = muon_jit_probe.graph_jit_ns

        @functools.wraps(original)
        def square_sum(value, steps, normalized=None):
            global CALLS
            if tuple(value.shape) == (8, 5120) and normalized is None:
                previous = torch_npu.npu.is_jit_compile_false()
                torch_npu.npu.set_compile_mode(jit_compile=False)
                try:
                    normalized = (value / ((value * value).sum().sqrt() + 1e-7)).bfloat16()
                finally:
                    torch_npu.npu.set_compile_mode(jit_compile=not previous)
                CALLS += 1
            return original(value, steps, normalized)

        muon_jit_probe.graph_jit_ns = square_sum

    capture_hp_native.Observer.on_train_begin = begin
    capture_hp_native.Observer.on_step_end = end
    muon_dense_adjacent_factor.install = deferred_install
