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

"""Isolate FP32 update packing in the requested single-card sharded Muon path."""
import functools
import hashlib
import inspect
import json
import textwrap
from pathlib import Path

from hyper_parallel.core.optimizer import muon as muon_module


def install():
    original = muon_module.Muon._fused_broadcast_and_apply
    source = textwrap.dedent(inspect.getsource(original))
    if source.count('dtype=torch.bfloat16') != 2:
        raise ValueError('Unexpected pinned broadcast implementation')
    altered = source.replace('dtype=torch.bfloat16', 'dtype=torch.float32')
    namespace = dict(original.__globals__)
    exec(compile(altered, __file__ + ':single-factor-method', 'exec'), namespace)
    candidate = namespace[original.__name__]
    policy = {'scope': 'single-card diagnostic only', 'teacher_values_used': False,
              'original_source_sha256': hashlib.sha256(source.encode()).hexdigest(),
              'candidate_source_sha256': hashlib.sha256(altered.encode()).hexdigest(),
              'change': 'Only alignment element size and pack buffer dtype BF16 -> FP32; arithmetic and inputs unchanged'}

    @functools.wraps(original)
    def checked(self, params, updates, coords, group, assignment):
        if self._get_shard_info(assignment)[3] != 1:
            raise ValueError('Single-card buffer precision diagnostic cannot run multi-card')
        if any(value.dtype != muon_module.torch.float32 for value in updates.values()):
            raise ValueError('This experiment requires the measured FP32 scaled updates')
        return candidate(self, params, updates, coords, group, assignment)

    muon_module.Muon._fused_broadcast_and_apply = checked
    print('[P12-BROADCAST-FP32]', json.dumps(policy), flush=True)
    return policy
