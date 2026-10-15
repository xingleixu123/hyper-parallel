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

"""Separate two actual219 normalization causes without reading reference values."""
import functools
import hashlib
import inspect
import json
import os
from pathlib import Path

import torch
import torch_npu
import capture_hp_native
import muon_dense_adjacent_factor
import muon_expert_bmm_eager_factor
import muon_jit_probe


def install():
    """Apply static normalization choices to live values at every training step."""
    mode = os.environ['P12_NORM219_FACTOR']
    if mode not in ['dense', 'expert', 'both']:
        raise ValueError('Unknown normalization219 factor')
    counts = {'dense': 0, 'expert_pieces': 0}
    changes = {}
    if mode in ['dense', 'both']:
        original = inspect.getsource(muon_dense_adjacent_factor._normalized_dense)
        old = 'np.float32(np.sqrt(total) + np.float32(1.0011717677116394e-7))'
        assert original.count(old) == 1
        source = original.replace(old, 'torch.tensor(total, device=value.device).sqrt() + 1.0011717677116394e-7')
        source = source.replace('torch.tensor(denominator, device=value.device)', 'denominator')
        namespace = dict(muon_dense_adjacent_factor.__dict__)
        exec(compile(source, __file__ + ':dense-npu-sqrt', 'exec'), namespace)
        candidate = namespace['_normalized_dense']

        @functools.wraps(candidate)
        def dense(value):
            counts['dense'] += 1
            return candidate(value)

        muon_dense_adjacent_factor._normalized_dense = dense
        changes['dense'] = {'original_source_sha256': hashlib.sha256(original.encode()).hexdigest(),
                            'candidate_source_sha256': hashlib.sha256(source.encode()).hexdigest(),
                            'change': 'Preserve reduction tree; compute scalar square root on NPU.'}
    if mode in ['expert', 'both']:
        original_install = muon_expert_bmm_eager_factor.install

        def expert_install():
            original_install()
            original = muon_jit_probe.graph_jit_ns

            @functools.wraps(original)
            def expert(value, steps, normalized=None):
                if value.ndim == 3 and normalized is None:
                    counts['expert_pieces'] += 1
                    previous = torch_npu.npu.is_jit_compile_false()
                    torch_npu.npu.set_compile_mode(jit_compile=False)
                    try:
                        current = value.mT if value.shape[-2] > value.shape[-1] else value
                        norm = current.float().norm(dim=(-2, -1), keepdim=True)
                        denominator = (norm + 1e-7).bfloat16()
                        normalized = (current / denominator).bfloat16()
                    finally:
                        torch_npu.npu.set_compile_mode(jit_compile=not previous)
                return original(value, steps, normalized)

            muon_jit_probe.graph_jit_ns = expert

        muon_expert_bmm_eager_factor.install = expert_install
        changes['expert'] = {'change': 'Round norm plus epsilon to BF16 together, preserving BF16 input/division.'}
    original_begin = capture_hp_native.Observer.on_train_begin
    original_end = capture_hp_native.Observer.on_step_end

    def begin(self, state, **kwargs):
        original_begin(self, state, **kwargs)
        report = {'mode': mode, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'changes': changes, 'teacher_values_used': False, 'coordinate_or_step_special_case': False,
                  'classification': 'Archive compatibility arithmetic ablation; not an asserted default HP bug.'}
        (self.output / 'NORM219_FACTOR.json').write_text(json.dumps(report, indent=2))

    def end(self, state, **kwargs):
        original_end(self, state, **kwargs)
        with (self.output / 'NORM219_CALLS.jsonl').open('a') as stream:
            stream.write(json.dumps({'step': int(state.global_step), 'calls': dict(counts)}) + '\n')

    capture_hp_native.Observer.on_train_begin = begin
    capture_hp_native.Observer.on_step_end = end
