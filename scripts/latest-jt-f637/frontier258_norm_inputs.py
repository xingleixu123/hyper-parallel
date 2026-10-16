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

"""Produce deterministic actual and synthetic BF16 normalization holdouts."""
from pathlib import Path

import numpy as np

STATE = Path('/state/jt-recipe-20261014')
CASES = ['actual258_nope', 'actual258_rope', 'actual229_nope', 'actual229_rope',
         'synthetic_narrow_nope', 'synthetic_wide_nope', 'synthetic_wide_rope']


def load_case(name: str) -> np.ndarray:
    """Return FP32 representations of BF16 inputs, without writing tensor files."""
    if name.startswith('actual'):
        step = name[6:9]
        label = 'main1_q_up' if step == '258' else 'mtp_q_up'
        piece = 0 if name.endswith('nope') else 1
        return np.load(STATE / ('hp-frontier' + step + '-muon-replay') /
                       (label + '-' + str(piece)) / 'pre_norm.npy')
    if name not in CASES:
        raise ValueError('Unknown holdout')
    size = 16384 if name.endswith('nope') else 8192
    generator = np.random.RandomState(25801 if 'narrow' in name else 25802)
    lower, upper = (110, 121) if 'narrow' in name else (95, 135)
    exponent = generator.randint(lower, upper, size=(1536, size), dtype=np.uint32)
    mantissa = generator.randint(0, 128, size=exponent.shape, dtype=np.uint32)
    sign = generator.randint(0, 2, size=exponent.shape, dtype=np.uint32)
    bits = (sign << 31) | (exponent << 23) | (mantissa << 16)
    return bits.view(np.float32)
