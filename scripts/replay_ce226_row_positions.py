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

"""Test whether identical CE rows receive position-dependent SumJoin results."""
import hashlib
import json
from pathlib import Path

import numpy as np
import mindspore as ms

from compare_ce226 import load_arrays
from replay_cross_entropy_mf_o1_stages import CEStages

ROOT = Path('/alignment/phase12')
STATE = Path('/state/jt-recipe-20261014')


def main() -> None:
    """Repeat identical real rows across all positions with the validated full CE graph."""
    audit = json.loads((ROOT / 'evidence/ce226-comparison.json').read_text())
    if not audit['complete']:
        raise ValueError('Actual CE operands have not passed fidelity')
    records = [json.loads(line) for line in (ROOT / 'runs/mf-ce226/CE226_OPERANDS.jsonl').read_text().splitlines()]
    mf = load_arrays(records[-1]['records'], STATE / 'mf-ce226/deep', 'name')
    hp = load_arrays(json.loads((ROOT / 'runs/hp-jtf637-qsqrt222-ce226/CE226_OPERANDS.json').read_text())['records'],
                     STATE / 'hp-jtf637-qsqrt222-ce226/ce', 'call')
    output = STATE / 'ce226-row-positions'
    output.mkdir(exist_ok=False)
    ms.set_context(mode=ms.GRAPH_MODE, device_target='Ascend', device_id=0, deterministic='ON',
                   jit_config={'jit_level': 'O1'})
    results = []
    for name, mapping in audit['mapping'].items():
        call = mapping['ce_call']
        model = CEStages(0.3 if call == 1 else 1.0)
        labels = ms.Tensor(mf[name, 'targets'].reshape(-1).astype(np.int32))
        mask = ms.Tensor(mf[name, 'mask'].reshape(-1))
        for selected_slot, token in [(2, 4080), (3, 4095)]:
            values = np.repeat(mf[name, 'logits'][selected_slot:selected_slot + 1], 4096, axis=0)
            gradient, total, logarithm, log_probability, maximum = model(ms.Tensor(values), labels, mask)
            actual = total.asnumpy().reshape(-1)
            filename = f'call{call}-token{token}-sums.npy'
            np.save(output / filename, actual)
            groups = []
            for value in np.unique(actual):
                positions = np.flatnonzero(actual == value)
                groups.append({'count': int(positions.size), 'first_position': int(positions[0]),
                               'last_position': int(positions[-1]),
                               'matches_hp': bool(value == hp[call, 'sum'][selected_slot, 0]),
                               'matches_mf': bool(value == mf[name, 'sum'][selected_slot, 0])})
            row = {'name': name, 'source_token': token, 'unique_results': len(groups), 'groups': groups,
                   'original_position_matches_training': bool(actual[token] == mf[name, 'sum'][selected_slot, 0]),
                   'output': filename, 'sha256': hashlib.sha256(actual.tobytes()).hexdigest()}
            results.append(row)
            print(json.dumps(row), flush=True)
            del values, gradient, total, logarithm, log_probability, maximum
    report = {'complete': True, 'results': results, 'teacher_values_used_for_training': False,
              'scope': 'Diagnostic replicated rows; not a training candidate or an alignment result.',
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (ROOT / 'evidence/ce226-row-positions.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
