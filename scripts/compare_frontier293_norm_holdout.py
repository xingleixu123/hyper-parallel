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

"""Audit holdout kernel identity, complete input hashes and exact scalar reductions."""
import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from frontier293_norm_inputs import CASES, load_case
from mindspore272_stat_hash import native_dump_sha1

ROOT = Path('/alignment/phase12')
STATE = Path('/state/jt-recipe-20261014')


def blocks(path: Path) -> dict:
    """Normalize only the graph-local operator index, preserving all kernel instructions."""
    result = {}
    for text in re.findall(r'\[dvm kernel\] GraphKernel_Cast_Mul_ReduceSum_split.*?(?=\[func graph\]|\Z)',
                           path.read_text(), re.S):
        text = re.sub(r'-op[0-9]+', '-opN', text).strip()
        size = int(re.search(r'tile_num=([0-9]+)', text)[1])
        result[size] = hashlib.sha256(text.encode()).hexdigest()
    return result


def main() -> None:
    """Validate all three holdouts without retaining or sampling tensor values."""
    hp = json.loads((ROOT / 'evidence/frontier293-hp-norm-holdout.json').read_text())
    mf = json.loads((ROOT / 'evidence/frontier293-mf-norm-holdout-hash.json').read_text())
    directory = STATE / 'frontier293-mf-norm-holdout-hash'
    tables = []
    for path in directory.rglob('statistic.csv'):
        with path.open() as stream:
            rows = list(csv.DictReader(stream))
        tables.append((min(int(row['Timestamp']) for row in rows), path, rows))
    tables.sort()
    if len(tables) != len(CASES):
        raise ValueError('Unexpected native invocation count')
    checks = {
        'hp_complete': hp['complete'], 'mf_complete': mf['complete'],
        'all_cast_outputs_exact': all(row['cast_output_exact'] for row in mf['records']),
        'same_inputs_source': hp['inputs_source_sha256'] == mf['inputs_source_sha256'],
        'no_raw_arrays': not any(directory.rglob('*.npy')),
    }
    native_path = ROOT / 'evidence/frontier293-native-dvm.txt'
    kernels = list((STATE / 'work-frontier293-mf-norm-holdout-hash/graph_kernel_dump').glob('dvm*txt'))
    if len(kernels) != 1:
        raise ValueError('Expected one holdout kernel text')
    checks['native_reduction_kernel_identity'] = blocks(native_path) == blocks(kernels[0])
    records = []
    for name, first, second, (_, path, table) in zip(CASES, hp['records'], mf['records'], tables):
        value = load_case(name)
        bf16 = (value.view(np.uint32) >> 16).astype(np.uint16)
        inputs = [row for row in table if row['Op Type'] == 'GraphKernel_Cast_Mul_ReduceSum_split'
                  and row['IO'] == 'input' and row['Slot'] == '0']
        sums = [row for row in table if row['Op Type'] == 'GraphKernel_Cast_Mul_ReduceSum_split'
                and row['IO'] == 'output' and row['Slot'] == '1']
        if len(inputs) != 1 or len(sums) != 1:
            raise ValueError('Ambiguous native reduction')
        row = {'case': name, 'elements': int(value.size),
               'case_order': first['case'] == second['case'] == name,
               'same_input': first['input_sha256'] == second['input_sha256'],
               'complete_native_input_hash': native_dump_sha1(bf16) == inputs[0]['SHA1'],
               'complete_normalized_output_hash': first['normalized_sha256'] == second['normalized_sha256'],
               'square_sum_bit_exact': native_dump_sha1(np.asarray(first['scalars']['square_sum'], dtype=np.float32))
                                       == sums[0]['SHA1'],
               'square_sum': first['scalars']['square_sum'], 'native_sum_sha1': sums[0]['SHA1'],
               'native_csv': str(path.relative_to(STATE))}
        records.append(row)
    checks['all_holdouts'] = all(all(row[key] for key in ['case_order', 'same_input',
        'complete_native_input_hash', 'complete_normalized_output_hash', 'square_sum_bit_exact']) for row in records)
    for run in ['frontier293-hp-norm-holdout', 'frontier293-mf-norm-holdout-hash']:
        checks[run + '_exit_zero'] = (ROOT / 'runs' / (run + '.exit')).read_text().strip() == '0'
    report = {'complete': True, 'fidelity': checks, 'records': records,
              'kernel_digests': blocks(native_path),
              'scope': 'Observed scalar/output equivalence on three inputs; internal tile partials are not exposed.',
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (ROOT / 'evidence/frontier293-norm-holdout-audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
    if not all(checks.values()):
        raise ValueError(checks)


if __name__ == '__main__':
    main()
