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

"""Match the installed MindSpore2.7.2 native Dump hash composition."""
import hashlib

import numpy as np

ELEMENTS_PER_SLICE = 20_480
SOURCE_COMMIT = '970b12dc1a89d57d703e7a1bae412212aad95887'


def native_dump_sha1(value: np.ndarray) -> str:
    """Hash every logical element using native chunk-and-hex composition.

    The source contract is TensorSummary::TensorStatistics in the installed
    MindSpore build, rather than an ordinary single digest over all bytes.
    """
    array = np.ascontiguousarray(value).reshape(-1)
    if array.size <= ELEMENTS_PER_SLICE:
        return hashlib.sha1(memoryview(array).cast('B')).hexdigest()
    combined = hashlib.sha1()
    for start in range(0, array.size, ELEMENTS_PER_SLICE):
        chunk = array[start:start + ELEMENTS_PER_SLICE]
        combined.update(hashlib.sha1(memoryview(chunk).cast('B')).hexdigest().encode('ascii'))
    return combined.hexdigest()
