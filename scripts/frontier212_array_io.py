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

"""Bounded-memory fingerprints and FP64 norm references for private gradient arrays."""
import hashlib
import math
from typing import Any

import numpy as np


def describe(array: np.ndarray) -> dict[str, Any]:
    """Hash complete values and reduce FP64 squares in bounded chunks."""
    if array.dtype != np.float32:
        raise ValueError('Expected original FP32 gradients')
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    flat = array.reshape(-1)
    sums = []
    for start in range(0, flat.size, 1_048_576):
        chunk = flat[start:start + 1_048_576]
        digest.update(memoryview(chunk).cast('B'))
        sums.append(float(np.sum(np.square(chunk, dtype=np.float64), dtype=np.float64)))
    return {'shape': list(array.shape), 'dtype': str(array.dtype), 'sha256': digest.hexdigest(),
            'square_sum_fp64': math.fsum(sums), 'array_bytes': int(array.nbytes)}
