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

"""Select every observed main-head dinput mismatch for bounded operand diagnosis."""
import json
from pathlib import Path
import numpy as np

ROOT = Path('/alignment/phase12')
STATE = Path('/state/jt-recipe-20261014')


def main():
    """Find all226 dy mismatches without changing any training values."""
    audit = json.loads((ROOT / 'evidence/backward226-comparison.json').read_text())
    if not audit['complete'] or not all(all(v.values()) for v in audit['fidelity'].values()):
        raise ValueError('Missing native-fidelity audit')
    hp_name = 'hp-jtf637-qsqrt222-backward226'
    mf_name = 'mf-backward226'
    hp = json.loads((ROOT / 'runs' / hp_name / 'BACKWARD226_BOUNDARIES.json').read_text())
    mf = json.loads((ROOT / 'runs' / mf_name / 'BACKWARD226_BOUNDARIES.jsonl').read_text().splitlines()[-1])
    left = next(r for r in hp['records'] if r['name'] == 'model.norm' and r['kind'] == 'dy')
    right = next(r for r in mf['records'] if r['name'] == 'model.decoder.final_layernorm' and r['kind'] == 'dy')
    a = np.load(STATE / hp_name / 'norms' / left['file'], mmap_mode='r').reshape(4096, 5120)
    b = np.load(STATE / mf_name / 'deep' / right['file'], mmap_mode='r').reshape(4096, 5120)
    coordinates = np.argwhere(a != b).tolist()
    if len(coordinates) != 9:
        raise ValueError('Unexpected frontier')
    rows = sorted({row for row, _ in coordinates})
    controls = [i for i in range(4096) if i not in rows][:2]
    result = {'complete': True, 'step': 226, 'coordinates': coordinates,
              'token_rows': sorted(rows + controls), 'affected_token_rows': rows,
              'control_token_rows': controls, 'hidden_columns': sorted({col for _, col in coordinates}),
              'scope': 'Every observed main-head dinput mismatch; all vocabulary entries in affected token rows; observation only.'}
    (ROOT / 'evidence/head226-selected-coordinates.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
