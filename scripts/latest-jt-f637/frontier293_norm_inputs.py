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

"""Construct deterministic dense-fc1 holdouts without writing full tensors."""
import numpy as np

CASES = ['synthetic_narrow', 'synthetic_wide', 'synthetic_tail']


def load_case(name: str) -> np.ndarray:
    """Cover ordinary magnitudes, wider exponents and the final reduction tile."""
    if name not in CASES:
        raise ValueError('Unknown holdout')
    generator = np.random.RandomState({'synthetic_narrow': 29301,
                                      'synthetic_wide': 29302, 'synthetic_tail': 29303}[name])
    lower, upper = (95, 135) if name == 'synthetic_wide' else (110, 121)
    exponent = generator.randint(lower, upper, size=(5120, 12288), dtype=np.uint32)
    mantissa = generator.randint(0, 128, size=exponent.shape, dtype=np.uint32)
    sign = generator.randint(0, 2, size=exponent.shape, dtype=np.uint32)
    bits = (sign << 31) | (exponent << 23) | (mantissa << 16)
    value = bits.view(np.float32)
    if name == 'synthetic_tail':
        value.reshape(-1)[:-6000] = 0
    return value
