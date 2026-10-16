# Copyright 2026 Huawei Technologies Co., Ltd
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

"""Exercise the device-selection boundary without constructing a model or allocating tensors."""

import json
import os
from pathlib import Path

import torch

from hyper_parallel.models.jt_deepseek_v3.adapter import jt_builder
from hyper_parallel.models import build_options


def main() -> None:
    """Record discovery facts and fail on the observed noninteger NPU index symptom."""
    result = {
        'source_file': build_options.__file__,
        'torch_file': torch.__file__,
        'torch_version': torch.__version__,
        'cached_npu_available': build_options.IS_NPU_AVAILABLE,
        'cached_cuda_available': build_options.IS_CUDA_AVAILABLE,
        'device_type': build_options.get_device_type(),
        'npu_device_count': torch.npu.device_count(),
        'npu_available_now': torch.npu.is_available(),
        'environment': {name: os.environ.get(name) for name in (
            'ASCEND_RT_VISIBLE_DEVICES', 'ASCEND_VISIBLE_DEVICES', 'ASCEND_HOME_PATH',
            'LD_LIBRARY_PATH', 'TORCH_DEVICE_BACKEND_AUTOLOAD', 'CLOSE_MATMUL_K_SHIFT')},
        'scope': 'Device discovery and torch.device construction only; no model, tensors or training',
    }
    try:
        index = jt_builder.get_device_id()
        result.update(device_index=repr(index), device_index_type=type(index).__name__)
        device = torch.device('npu', index)
        result['constructed_device'] = str(device)
        result['status'] = 'PASS' if build_options.IS_NPU_AVAILABLE and isinstance(index, int) else 'FAIL'
    except (RuntimeError, TypeError, ValueError) as error:
        result.update(status='FAIL', error_type=type(error).__name__, error=str(error))
    result['loaded_driver_libraries'] = sorted({line.split()[-1] for line in
        Path('/proc/self/maps').read_text().splitlines() if 'libascend_hal' in line or 'libdrvdsmi' in line})
    print(json.dumps(result, ensure_ascii=False), flush=True)
    if result['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
