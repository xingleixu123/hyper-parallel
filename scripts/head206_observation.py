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

"""Save only the actual FP32 output-head state at the measured frontier."""
import hashlib
import json
from pathlib import Path

import numpy as np
from model_hash_stream import model_record


def save_head(output, array, source):
    """Keep one model tensor, plus complete FP32/BF16 hashes, without any state substitution."""
    if array.shape != (152064, 5120) or array.dtype != np.float32:
        raise ValueError('Unexpected output-head shape or dtype')
    directory = Path('/state/jt-recipe-20261014') / output.name
    directory.mkdir(exist_ok=False)
    np.save(directory / 'head206.npy', array, allow_pickle=False)
    record = model_record('lm_head.weight', array)
    report = {'step': 206, 'record': record, 'file': 'head206.npy',
              'source_sha256': hashlib.sha256(Path(source).read_bytes()).hexdigest(),
              'helper_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'hash_helper_sha256': hashlib.sha256((Path(__file__).parent / 'model_hash_stream.py').read_bytes()).hexdigest(),
              'scope': 'One actual postupdate model tensor; no optimizer state changed or additional checkpoint saved.'}
    (output / 'HEAD206_STATE.json').write_text(json.dumps(report, indent=2))
