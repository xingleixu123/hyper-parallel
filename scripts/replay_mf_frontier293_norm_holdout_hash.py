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

"""Validate native normalization on independent inputs with statistics-only Dump."""
import hashlib
import json
from pathlib import Path

import mindspore as ms
from mindspore import nn, ops
import numpy as np

from frontier293_norm_inputs import CASES, load_case

ROOT = Path('/alignment/phase12')


class Normalization(nn.Cell):
    """Retain the same FP32 cast output used by the native fused reduction."""
    def construct(self, value):
        current = value.astype(ms.float32)
        square_sum = ops.reduce_sum(current * current, (0, 1))
        normalized = (current / (ops.sqrt(square_sum) + ms.Tensor(1.0011717677116394e-7, ms.float32)))
        return current, normalized.astype(ms.bfloat16)


def main() -> None:
    """Record complete output hashes, not sampled values or full arrays."""
    ms.set_context(mode=ms.GRAPH_MODE, device_target='Ascend', device_id=0, deterministic='ON',
                   jit_config={'jit_level': 'O1'}, ascend_config={'parallel_speed_up_json_path':
                   '/workspace/mindformers/configs/qwen3/parallel_speed_up.json'})
    model = Normalization()
    records = []
    for name in CASES:
        value = load_case(name)
        raw, normalized = model(ms.Tensor(value, ms.bfloat16))
        actual = normalized.astype(ms.float32).asnumpy()
        record = {'case': name, 'shape': list(value.shape),
                  'input_sha256': hashlib.sha256(value.tobytes()).hexdigest(),
                  'cast_output_exact': np.array_equal(raw.asnumpy(), value),
                  'normalized_sha256': hashlib.sha256(actual.tobytes()).hexdigest()}
        records.append(record)
        print(json.dumps(record), flush=True)
    report = {'complete': True, 'records': records, 'arrays_written': False,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'inputs_source_sha256': hashlib.sha256((ROOT / 'tools/frontier293_norm_inputs.py').read_bytes()).hexdigest()}
    (ROOT / 'evidence/frontier293-mf-norm-holdout-hash.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
