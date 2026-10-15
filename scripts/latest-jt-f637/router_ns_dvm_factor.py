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

"""Independent same-shape router NS fixed-tree normalization factor."""
import functools
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch_npu
from observe_hp_preclip_norm_adjacent import dvm_square_sum
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
                  'scope': 'Both router matrices; fixed DVM tree square-sum of actual FP32 NS input with FP32 CPU sqrt, then original final BF16 cast.',
                  'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'reduction_helper_sha256': hashlib.sha256((Path(__file__).parent / 'observe_hp_preclip_norm_adjacent.py').read_bytes()).hexdigest()}
        (self.output / 'ROUTER_NS_DVM_FACTOR.json').write_text(json.dumps(report, indent=2))

    def end(self, state, **kwargs):
        original_end(self, state, **kwargs)
        row = {'step': int(state.global_step), 'total_router_calls': CALLS}
        with (self.output / 'ROUTER_NS_DVM_CALLS.jsonl').open('a') as handle:
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
                    norm = np.float32(np.sqrt(dvm_square_sum(value)))
                    norm = torch.tensor(float(norm), dtype=value.dtype, device=value.device)
                    normalized = (value / (norm + 1e-7)).bfloat16()
                finally:
                    torch_npu.npu.set_compile_mode(jit_compile=not previous)
                CALLS += 1
            return original(value, steps, normalized)

        muon_jit_probe.graph_jit_ns = square_sum

    capture_hp_native.Observer.on_train_begin = begin
    capture_hp_native.Observer.on_step_end = end
    muon_dense_adjacent_factor.install = deferred_install
