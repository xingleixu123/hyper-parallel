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

"""Match the archived 2D NS epsilon literal type without changing norms or iterations."""
import functools
import hashlib
import inspect
import json
from pathlib import Path

import torch

import capture_hp_native
import muon_jit_probe


EPSILON = float(torch.tensor(1e-7, dtype=torch.bfloat16).float())


def install() -> dict:
    """Change one live scalar conversion, preserving input values and all existing factors."""
    original = inspect.getsource(muon_jit_probe.graph_jit_ns)
    needle = 'current.norm(dim=(-2, -1), keepdim=True) + 1e-7'
    if original.count(needle) != 1:
        raise ValueError('Unexpected frozen NS normalization expression')
    source = original.replace(needle,
        'current.norm(dim=(-2, -1), keepdim=True) + (P12_EPSILON229 if current.ndim == 2 else 1e-7)')
    namespace = dict(muon_jit_probe.__dict__)
    namespace['P12_EPSILON229'] = EPSILON
    exec(compile(source, __file__ + ':epsilon-literal-type', 'exec'), namespace)
    candidate = namespace['graph_jit_ns']
    counts = {}

    @functools.wraps(candidate)
    def changed(value, steps, normalized=None):
        if value.ndim == 2 and normalized is None:
            shape = str(tuple(value.shape))
            counts[shape] = counts.get(shape, 0) + 1
        return candidate(value, steps, normalized)

    muon_jit_probe.graph_jit_ns = changed
    report = {'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'base_source_sha256': hashlib.sha256(Path(muon_jit_probe.__file__).read_bytes()).hexdigest(),
              'original_source': original, 'candidate_source': source, 'epsilon': EPSILON,
              'scope': 'Only2D default normalization; externally normalized inputs and3D retain prior behavior.',
              'teacher_values_used': False, 'step_specific_behavior': False,
              'classification': 'Archived MF compiled scalar-type compatibility; not a generic HP correctness claim.'}
    original_begin = capture_hp_native.Observer.on_train_begin
    original_end = capture_hp_native.Observer.on_step_end

    def begin(self, state, **kwargs):
        original_begin(self, state, **kwargs)
        (self.output / 'NS_EPSILON229_FACTOR.json').write_text(json.dumps(report, indent=2))

    def end(self, state, **kwargs):
        original_end(self, state, **kwargs)
        with (self.output / 'NS_EPSILON229_CALLS.jsonl').open('a') as stream:
            stream.write(json.dumps({'step': int(state.global_step), 'cumulative_calls': dict(counts)}) + '\n')

    capture_hp_native.Observer.on_train_begin = begin
    capture_hp_native.Observer.on_step_end = end
    return report
