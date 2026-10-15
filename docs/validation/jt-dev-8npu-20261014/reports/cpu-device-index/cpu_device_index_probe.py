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

"""Reproduce the CPU device-index failure without opening an accelerator."""

import json
from unittest.mock import patch

import torch

from hyper_parallel.models import build_options


def main() -> None:
    """Exercise the model builders' device construction against the CPU backend."""
    with patch.object(build_options, 'IS_CUDA_AVAILABLE', False), \
            patch.object(build_options, 'IS_NPU_AVAILABLE', False):
        device_type = build_options.get_device_type()
        index = build_options.get_device_id()
        result = {'device_type': device_type, 'index': index, 'index_type': type(index).__name__,
                  'scope': 'CPU only; accelerator availability patched false; no NPU devices mounted'}
        try:
            actual = torch.device(device_type, index)
        except TypeError as error:
            result.update(status='REPRODUCED', error=str(error))
            print(json.dumps(result))
            raise SystemExit(1) from error
        result.update(status='PASS' if actual == torch.device('cpu') else 'FAIL', actual=str(actual))
        print(json.dumps(result))
        raise SystemExit(0 if result['status'] == 'PASS' else 1)


if __name__ == '__main__':
    main()
