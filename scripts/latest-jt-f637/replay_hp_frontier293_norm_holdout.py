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

"""Apply the independent dense-fc1 reduction to actual and synthetic holdouts."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch_npu

from frontier293_norm_inputs import CASES, load_case
from hp_fc1_norm293_factor import normalized_fc1

ROOT = Path('/alignment/phase12')


def main() -> None:
    """Write only complete input/output hashes and computed live scalars."""
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    records = []
    for name in CASES:
        value = load_case(name)
        actual, scalars = normalized_fc1(torch.from_numpy(value).npu().mT)
        output = actual.float().cpu().numpy()
        record = {'case': name, 'shape': list(value.shape), 'scalars': scalars,
                  'input_sha256': hashlib.sha256(value.tobytes()).hexdigest(),
                  'normalized_sha256': hashlib.sha256(output.tobytes()).hexdigest(),
                  'fp64_square_sum': float(np.square(value, dtype=np.float64).sum())}
        records.append(record)
        print(json.dumps(record), flush=True)
    report = {'complete': True, 'records': records, 'arrays_written': False,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'inputs_source_sha256': hashlib.sha256((ROOT / 'tools-f637/frontier293_norm_inputs.py').read_bytes()).hexdigest()}
    (ROOT / 'evidence/frontier293-hp-norm-holdout.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
