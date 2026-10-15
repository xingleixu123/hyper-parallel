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

"""Check whether the complete MTP embedding difference is confined to packed document tails."""
import hashlib
import json
from pathlib import Path

import numpy as np

from compare_vocab_optimizer_trace import read_rows

ROOT = Path('/alignment/phase12')
STATE = Path('/state/jt-recipe-20261014')


def main() -> None:
    """Compare all embedding elements and record only counts and boundary membership."""
    hp = json.loads((ROOT / 'runs/hp-jtf637-weighted-dvm-forward217/FORWARD217.json').read_text())['records']
    mf = read_rows(ROOT / 'runs/mf-forward217/FORWARD217.jsonl')[-1]['records']
    left = next(row for row in hp if row['name'] == 'mtp.layers.0.enorm' and row['kind'] == 'input')
    right = next(row for row in mf if row['name'] == 'model.mtp.layers.0.enorm' and row['kind'] == 'input')
    arrays = []
    for row, directory in [(left, 'hp-jtf637-weighted-dvm-forward217/forward'), (right, 'mf-forward217/deep')]:
        array = np.load(STATE / directory / row['file'], mmap_mode='r')
        if hashlib.sha256(array.tobytes()).hexdigest() != row['sha256']:
            raise ValueError('Full native embedding observation changed')
        arrays.append(array.reshape(4096, 5120))
    unequal = arrays[0] != arrays[1]
    changed_rows = np.flatnonzero(np.any(unequal, axis=1))
    attention = json.loads((ROOT / 'runs/hp-jtf637-weighted-dvm-attention217/ATTENTION217.json').read_text())
    tails = np.asarray(attention['options']['actual_seq_qlen'][:-1]) - 1
    report = {'complete': True, 'elements': int(unequal.size), 'different_elements': int(unequal.sum()),
              'different_token_rows': int(changed_rows.size), 'internal_document_tails': int(tails.size),
              'changed_rows_exactly_internal_document_tails': bool(np.array_equal(changed_rows, tails)),
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'scope': 'Full actual MTP embedding inputs; no raw token IDs or tensor values published.'}
    (ROOT / 'evidence/packed217-mtp-document-tail.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == '__main__':
    main()
