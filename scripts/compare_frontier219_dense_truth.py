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

"""Compare CPU/NPU dense219 square roots against complete real-input FP64 accumulation."""
import gzip
import hashlib
import json
import math
from pathlib import Path
import numpy as np
ROOT = Path('/alignment/phase12')
STATE = Path('/state/jt-recipe-20261014')


def main():
    relative = Path('hp-frontier219-muon-replay/down-0/pre_ns.npy')
    source = STATE / relative
    if source.exists():
        array = np.load(source, mmap_mode='r')
    else:
        with gzip.open(ROOT / 'private-archives/frontier219-replays' / relative.with_suffix('.npy.gz'), 'rb') as stream:
            array = np.load(stream, allow_pickle=False)
    report = json.loads((ROOT / 'evidence/frontier219-hp-muon-replay.json').read_text())
    row = next(r for r in report['records'] if r['label'] == 'down')
    expected = next(r for r in row['arrays'] if r['name'] == 'pre_ns')
    assert hashlib.sha256(array.tobytes()).hexdigest() == expected['sha256']
    array = array.reshape(-1)
    total = math.fsum(float(np.sum(np.square(array[i:i+1048576], dtype=np.float64)))
                      for i in range(0, array.size, 1048576))
    truth = math.sqrt(total)
    rows = {}
    for name in ['frontier219-hp-muon-replay', 'frontier219-dense-npusqrt-replay']:
        report = json.loads((ROOT / 'evidence' / (name + '.json')).read_text())
        scalar = next(r for r in report['records'] if r['label'] == 'down')['scalar']
        rows[name] = {**scalar, 'norm_abs_error_vs_fp64_input': abs(scalar['norm'] - truth),
                      'denominator_abs_error_vs_fp64_input_plus_exact_epsilon': abs(scalar['denominator'] - (truth + 1e-7))}
    output = {'input_square_sum_fp64': total, 'input_norm_fp64': truth,
              'scalar_sqrt_of_reduced_fp32_sum_fp64': math.sqrt(rows['frontier219-hp-muon-replay']['square_sum']),
              'rows': rows, 'scope': 'CPU and NPU square roots compared against FP64; reference agreement is not automatically superior numerical accuracy.'}
    path = ROOT / 'evidence/frontier219-dense-fp64-truth.json'
    prior = json.loads(path.read_text())
    assert all(prior[k] == v for k, v in output.items())
    output['source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    path.write_text(json.dumps(output, indent=2))
    print('FP64 truth reproduced from complete hash-verified real input')


if __name__ == '__main__':
    main()
