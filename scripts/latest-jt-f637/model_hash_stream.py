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

"""Hash exact FP32 model storage and its BF16 round-to-nearest-even view in bounded chunks."""
import hashlib
import numpy as np


def model_record(name: str, array: np.ndarray) -> dict:
    """Hash model values without storing another model checkpoint."""
    if array.dtype != np.float32:
        raise ValueError('Expected actual FP32 model storage: ' + name + ' ' + str(array.dtype))
    flat = np.ascontiguousarray(array).reshape(-1)
    fp32, bf16 = hashlib.sha256(), hashlib.sha256()
    for start in range(0, flat.size, 1_000_000):
        piece = flat[start:start + 1_000_000]
        if not np.isfinite(piece).all():
            raise ValueError('Nonfinite model value: ' + name)
        fp32.update(piece.tobytes())
        bits = piece.view(np.uint32)
        rounded = (bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) & np.uint32(0xffff0000)
        bf16.update(rounded.tobytes())
    return {'name': name, 'shape': list(array.shape), 'size': int(array.size),
            'fp32_sha256': fp32.hexdigest(), 'bf16_as_fp32_sha256': bf16.hexdigest()}
