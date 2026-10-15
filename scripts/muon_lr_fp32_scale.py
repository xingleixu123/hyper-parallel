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

"""Represent the archived FP32 scalar order without recorded future values."""
import math

import numpy as np


class FP32Scale(float):
    """Round the subsequent learning-rate product once in FP32."""
    def __imul__(self, other):
        return float(np.float32(np.float32(self) * np.float32(other)))


def fp32_archive_scale(value, matched_adamw_rms, zero_rms_scale_mode='zero'):
    if not matched_adamw_rms or zero_rms_scale_mode != 'zero':
        raise ValueError('Expected observed archived RMS scaling recipe')
    dimension = 8192 if tuple(value.shape) == (16384, 1536) else max(value.shape[-2:])
    root = np.float32(math.sqrt(float(dimension)))
    return FP32Scale(np.float32(root * np.float32(matched_adamw_rms)))
