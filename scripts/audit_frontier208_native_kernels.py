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

"""Verify unchanged native graph-kernel operands and normalization outputs at208."""
import hashlib
import json
from pathlib import Path

import numpy as np
from compare_vocab_optimizer_trace import fidelity

ROOT = Path('/alignment/phase12')
STATE = Path('/state/jt-recipe-20261014')


def main():
    run = 'mf-frontier208-native-normalized'
    checks = fidelity(run, 'mf-resume', 8)
    directory = ROOT / 'runs' / run
    baseline = ROOT / 'runs/mf-frontier208-state'
    checks['exit_zero'] = (ROOT / 'runs' / (run + '.exit')).read_text().strip() == '0'
    for step in [207, 208]:
        name = 'FRONTIER_STATE-' + str(step) + '.json'
        checks['state_' + str(step)] = json.loads((directory / name).read_text()) == json.loads((baseline / name).read_text())
    canonical = lambda rows: [{key: value for key, value in row.items() if key != 'file'} for row in rows]
    checks['gradient208'] = canonical(json.loads((directory / 'FRONTIER_GRADIENTS-208.json').read_text())) == canonical(
        json.loads((baseline / 'FRONTIER_GRADIENTS-208.json').read_text()))
    observations = {row['prefix']: row for row in json.loads((ROOT / 'runs/mf-frontier208-ns-train/NS_BOUNDARIES-208.json').read_text())}
    pre_hashes = {observations[label + '-pre_ns']['sha256']: label for label in ['q', 'router']}
    files = list((STATE / (run + '-kernels')).rglob('*.npy'))
    checks['exact15_native_arrays'] = len(files) == 15
    kernels = []
    for op in [1, 12, 19, 25, 32]:
        records, arrays = {}, {}
        for label, pattern in [('square_sum', '.input.0.'), ('ns_input', '.input.1.'), ('normalized', '.output.0.')]:
            paths = [p for p in files if ('-op' + str(op) + '.') in p.name and pattern in p.name]
            if len(paths) != 1:
                raise ValueError('Native kernel operand not unique')
            value = np.load(paths[0])
            arrays[label] = value
            records[label] = {'file': str(paths[0].relative_to(STATE)), 'shape': list(value.shape),
                              'sha256': hashlib.sha256(value.tobytes()).hexdigest()}
        target = pre_hashes.get(records['ns_input']['sha256'])
        if target:
            checks[target + '_native_normalized_exact_tap'] = records['normalized']['sha256'] == observations[target + '-normalized']['sha256']
        kernels.append({'op': op, 'matched_target_by_complete_input_hash': target,
                        'square_sum': float(arrays['square_sum']), 'records': records})
    checks['both_targets_found'] = sorted(row['matched_target_by_complete_input_hash'] for row in kernels if row['matched_target_by_complete_input_hash']) == ['q', 'router']
    report = {'complete': True, 'fidelity': checks, 'kernels': kernels,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'scope': 'Original compiled fused Sqrt+Add+Div+Cast kernel; FP32 square-sum and NS input, final BF16 result. '
              'The separate TensorDump norm scalars are materialized BF16 views and must not be used as fused FP32 denominators.'}
    (ROOT / 'evidence/frontier208-native-kernel-audit.json').write_text(json.dumps(report, indent=2))
    if not all(checks.values()):
        raise ValueError(checks)
    print(json.dumps({'fidelity': checks, 'kernels': [{k: v for k, v in row.items() if k != 'records'} for row in kernels]}))


if __name__ == '__main__':
    main()
