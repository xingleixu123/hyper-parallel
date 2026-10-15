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

"""Independent Q normalization: change only the24-lane partial-sum reduction tree."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import capture_hp_native
import muon_q_native_tile_factor
from muon_q_norm_probe import half_tree

CALLS = 0


def normalized_q(value):
    """Use adjacent24-lane pairing of actual values; retain all other control operations."""
    global CALLS
    if tuple(value.shape) != (1536, 5120) or value.dtype != torch.float32:
        raise ValueError('Unexpected live Q input')
    flat = value.reshape(-1)
    padded = value.new_zeros(1127 * 6984)
    padded[:flat.numel()] = flat * flat
    lanes = half_tree(padded.reshape(1127, 291, 24).transpose(-1, -2))
    buffer = value.new_zeros((1127, 32))
    buffer[:, :24] = lanes
    while buffer.shape[-1] > 1:
        buffer = buffer[..., ::2] + buffer[..., 1::2]
    partials = buffer[..., 0].cpu().numpy()
    accumulators = np.zeros(40, dtype=np.float32)
    for index, partial in enumerate(partials):
        core = index % 40
        accumulators[core] = np.float32(accumulators[core] + partial)
    total = np.float32(0)
    for partial in accumulators:
        total = np.float32(total + partial)
    denominator = np.float32(np.sqrt(total) + np.float32(1.0011717677116394e-7))
    CALLS += 1
    return (value / torch.tensor(denominator, device=value.device)).bfloat16()


def install():
    """Apply to all three actual Q matrices at every step, without recorded tensor values."""
    original_begin = capture_hp_native.Observer.on_train_begin
    original_end = capture_hp_native.Observer.on_step_end
    muon_q_native_tile_factor.normalized_q = normalized_q

    def begin(self, state, **kwargs):
        original_begin(self, state, **kwargs)
        report = {'teacher_values_used': False, 'coordinate_or_step_special_case': False,
                  'scope': 'All1536x5120 Q NS inputs; only24-lane reduction uses adjacent pairing.',
                  'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'control_wrapper_sha256': hashlib.sha256(Path(muon_q_native_tile_factor.__file__).read_bytes()).hexdigest(),
                  'half_tree_helper_sha256': hashlib.sha256((Path(__file__).parent / 'muon_q_norm_probe.py').read_bytes()).hexdigest()}
        (self.output / 'Q_NS_ADJACENT_FACTOR.json').write_text(json.dumps(report, indent=2))

    def end(self, state, **kwargs):
        original_end(self, state, **kwargs)
        with (self.output / 'Q_NS_ADJACENT_CALLS.jsonl').open('a') as stream:
            stream.write(json.dumps({'step': int(state.global_step), 'total_q_calls': CALLS}) + '\n')

    capture_hp_native.Observer.on_train_begin = begin
    capture_hp_native.Observer.on_step_end = end
