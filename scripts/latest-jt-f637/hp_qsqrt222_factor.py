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

"""Isolate Q normalization square root placement on every live training step."""
import hashlib
import inspect
import json
from pathlib import Path

import capture_hp_native
import muon_q_adjacent_factor


def install():
    """Retain the full reduction tree and move only its scalar square root to NPU."""
    original = inspect.getsource(muon_q_adjacent_factor.normalized_q)
    old = 'np.float32(np.sqrt(total) + np.float32(1.0011717677116394e-7))'
    if original.count(old) != 1:
        raise ValueError('Unexpected Q normalization source')
    source = original.replace('def normalized_q(', 'def normalized_q_npusqrt222(')
    source = source.replace(old, 'torch.tensor(total, device=value.device).sqrt() + 1.0011717677116394e-7')
    source = source.replace('torch.tensor(denominator, device=value.device)', 'denominator')
    exec(compile(source, __file__ + ':q-npu-sqrt', 'exec'), muon_q_adjacent_factor.__dict__)
    muon_q_adjacent_factor.normalized_q = muon_q_adjacent_factor.normalized_q_npusqrt222
    original_begin = capture_hp_native.Observer.on_train_begin

    def begin(self, state, **kwargs):
        original_begin(self, state, **kwargs)
        report = {'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'original_function_sha256': hashlib.sha256(original.encode()).hexdigest(),
                  'candidate_function_sha256': hashlib.sha256(source.encode()).hexdigest(),
                  'teacher_values_used': False, 'coordinate_or_step_special_case': False,
                  'scope': 'All three Q matrices, every step; preserve reduction, epsilon and BF16 division output.',
                  'classification': 'Archive arithmetic compatibility ablation; no asserted CPU sqrt defect.'}
        (self.output / 'Q_NPU_SQRT222_FACTOR.json').write_text(json.dumps(report, indent=2))

    capture_hp_native.Observer.on_train_begin = begin
