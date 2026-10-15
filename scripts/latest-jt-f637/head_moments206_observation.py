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

"""Save only the actual output-head AdamW moments at the fixed206 boundary."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path('/alignment/phase12')
STATE = Path('/state/jt-recipe-20261014')
SHAPE = (152064, 5120)


def digest(array: np.ndarray) -> str:
    """Hash exact FP32 storage without making another full array copy."""
    if array.shape != SHAPE or array.dtype != np.float32 or not array.flags.c_contiguous:
        raise ValueError('Expected contiguous152064x5120 FP32 head state')
    return hashlib.sha256(memoryview(array).cast('B')).hexdigest()


def save(output: Path, kind: str, array: np.ndarray) -> dict:
    """Keep one moment array and its exact data checksum."""
    if kind not in ('exp_avg', 'exp_avg_sq'):
        raise ValueError('Unexpected moment kind')
    directory = STATE / output.name / 'moments206'
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / (kind + '.npy')
    if target.exists():
        raise FileExistsError(target)
    sha = digest(array)
    np.save(target, array, allow_pickle=False)
    return {'kind': kind, 'file': 'moments206/' + target.name, 'sha256': sha,
            'shape': list(array.shape), 'dtype': str(array.dtype), 'bytes': target.stat().st_size}


def finish(output: Path, family: str, weight: np.ndarray, records: list,
           optimizer_step: int, source: str, **extra) -> None:
    """Require the same complete model state as the previous independent observation."""
    expected = json.loads((ROOT / 'evidence/head206-model-comparison.json').read_text())
    actual = digest(weight)
    if actual != expected['model_records'][family]['fp32_sha256'] or optimizer_step != 206:
        raise ValueError('Head model or optimizer step did not reproduce the independent control')
    if [row['kind'] for row in records] != ['exp_avg', 'exp_avg_sq']:
        raise ValueError('Exactly two ordered AdamW moments required')
    report = {'step': 206, 'family': family, 'name': 'lm_head.weight', 'records': records,
              'optimizer_step': optimizer_step, 'model_fp32_sha256': actual,
              'source_sha256': hashlib.sha256(Path(source).read_bytes()).hexdigest(),
              'helper_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'read_only_observation': True, 'extra': extra}
    (output / 'HEAD_MOMENTS206.json').write_text(json.dumps(report, indent=2))
    print('[P12-HEAD-MOMENTS206]', family, 'complete', flush=True)
