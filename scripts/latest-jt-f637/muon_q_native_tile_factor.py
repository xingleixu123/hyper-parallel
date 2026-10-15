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

"""Test the actual full-model DVM tile schedule with forty strided vector-core accumulators."""
import functools
import numpy as np
import torch
import muon_jit_probe
from muon_q_norm_probe import half_tree


def normalized_q(value: torch.Tensor) -> torch.Tensor:
    if tuple(value.shape)!=(1536,5120) or value.dtype!=torch.float32:
        raise ValueError('Measured Q normalization shape/dtype required')
    flat=value.reshape(-1);padded=value.new_zeros(1127*6984)
    padded[:flat.numel()]=flat*flat
    tiled=padded.reshape(1127,291,24).transpose(-1,-2)
    partials=half_tree(half_tree(tiled)).cpu().numpy()
    accumulators=np.zeros(40,dtype=np.float32)
    for index,partial in enumerate(partials):
        core=index%40
        accumulators[core]=np.float32(accumulators[core]+partial)
    total=np.float32(0)
    for partial in accumulators:
        total=np.float32(total+partial)
    denominator=np.float32(np.sqrt(total)+np.float32(1.0011717677116394e-7))
    return (value/torch.tensor(denominator,device=value.device)).bfloat16()


def install() -> None:
    original=muon_jit_probe.graph_jit_ns
    @functools.wraps(original)
    def q_norm(value,steps,normalized=None):
        if tuple(value.shape)==(1536,5120) and normalized is None:
            normalized=normalized_q(value)
        return original(value,steps,normalized)
    muon_jit_probe.graph_jit_ns=q_norm
    print('[P12-Q-NATIVE-TILE] live tiled normalization; no recorded values used',flush=True)
