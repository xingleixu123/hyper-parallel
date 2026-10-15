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

"""Separate archived MF position-reset and MTP document-boundary semantics."""
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch

import capture_hp_native
from hyper_parallel.models.jt_deepseek_v3 import modeling_jt_deepseek_v3 as jt
from hyper_parallel.components.modules.mtp import MultiTokenPrediction


def install() -> None:
    """Apply one declared semantics factor to live inputs on every resumed step."""
    mode = os.environ['P12_PACKED_SEMANTICS_FACTOR']
    if mode not in ['absolute_positions', 'global_mtp', 'both']:
        raise ValueError('Unknown packed semantics factor')
    original_compute = jt.JTDeepseekV3ForCausalLM.compute_jt_losses
    original_mtp = MultiTokenPrediction.forward
    original_targets = jt.iter_mtp_targets
    original_begin = capture_hp_native.Observer.on_train_begin
    original_end = capture_hp_native.Observer.on_step_end
    counts = {'compute': 0, 'positions': 0, 'mtp_forward': 0, 'mtp_targets': 0}
    active = False

    def compute(self: Any, input_ids: Any, labels: Any, loss_mask: Any, **kwargs: Any):
        nonlocal active
        if active or kwargs.get('sequence_start', 0) != 0 or input_ids.shape[0] != 1:
            raise ValueError('Single-card batch-one nonnested observation required')
        counts['compute'] += 1
        if mode in ['absolute_positions', 'both']:
            kwargs['position_ids'] = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
            counts['positions'] += 1
        active = True
        try:
            return original_compute(self, input_ids, labels, loss_mask, **kwargs)
        finally:
            active = False

    def mtp(self: Any, *args: Any, **kwargs: Any):
        if active and mode in ['global_mtp', 'both']:
            kwargs['sequence_end_mask'] = None
            counts['mtp_forward'] += 1
        return original_mtp(self, *args, **kwargs)

    def targets(*args: Any, **kwargs: Any):
        if active and mode in ['global_mtp', 'both']:
            kwargs['sequence_end_mask'] = None
            counts['mtp_targets'] += 1
        return original_targets(*args, **kwargs)

    def begin(self: Any, state: Any, **kwargs: Any) -> None:
        original_begin(self, state, **kwargs)
        (self.output / 'PACKED_SEMANTICS_FACTOR.json').write_text(json.dumps({
            'mode': mode, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'teacher_values_used': False, 'coordinate_or_step_special_case': False,
            'classification': 'Archived-MF semantic compatibility ablation, not a proposed default correctness fix.'}, indent=2))

    def end(self: Any, state: Any, **kwargs: Any) -> None:
        original_end(self, state, **kwargs)
        with (self.output / 'PACKED_SEMANTICS_CALLS.jsonl').open('a') as stream:
            stream.write(json.dumps({'step': int(state.global_step), 'calls': dict(counts)}) + '\n')

    jt.JTDeepseekV3ForCausalLM.compute_jt_losses = compute
    MultiTokenPrediction.forward = mtp
    jt.iter_mtp_targets = targets
    capture_hp_native.Observer.on_train_begin = begin
    capture_hp_native.Observer.on_step_end = end
