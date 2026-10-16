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

"""Distinguish inherited expert weights from an actual329 normalization or NS discrepancy."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from compare_frontier219_state import load
from compare_preclip217 import bf16_bits, compare_arrays

ROOT = Path('/alignment/phase12')
STATE = Path('/state/jt-recipe-20261014')
HP = 'hp-jte213-fc1norm-qupboth-state329'
MF = 'mf-frontier329-state'


def main() -> None:
    """Require faithful native replays and compare complete operator-boundary hashes."""
    hp = json.loads((ROOT / 'evidence/frontier329-hp-control-replay.json').read_text())
    native = json.loads((ROOT / 'evidence/frontier329-mf-builtin-stat.json').read_text())
    scalar = json.loads((ROOT / 'evidence/frontier329-mf-normalization-scalars.json').read_text())
    observation = json.loads((ROOT / 'evidence/frontier329-state-comparison.json').read_text())
    checks = {'hp_replay': hp['complete'] and all(row['weight_exact_training'] and row['momentum_exact_training']
                                                for row in hp['checks']),
              'mf_native_replay': native['complete'] and all(row['exact_actual_training']
                                                            for row in native['native_state_fidelity'].values()),
              'mf_scalar_dump': scalar['complete'] and all(row['exact_actual_training']
                                                           for row in scalar['native_state_fidelity'].values()),
              'state_capture': observation['complete'] and all(all(row.values())
                                                               for row in observation['fidelity'].values())}
    rows = list(csv.DictReader((ROOT / 'evidence/frontier329-native-update-stat.csv').open()))

    def native_hash(kernel, io, slot):
        selected = [row for row in rows if row['Op Name'] == 'Default/' + kernel
                    and row['IO'] == io and row['Slot'] == str(slot)]
        if len(selected) != 1:
            raise ValueError('Native boundary not unique')
        return selected[0]['SHA1']

    record = hp['records'][0]
    checks.update({
        'pre_norm_exact': record['pre_norm']['native_dump_sha1'] == native_hash(
            'GraphKernel_Cast_Mul_ReduceSum_split-op0', 'input', 0),
        'normalized_exact': record['control_normalized']['native_dump_sha1'] == native_hash(
            'GraphKernel_Div_fusion-op0', 'output', 0),
        'ns_output_exact': record['control_ns_mf_layout']['native_dump_sha1'] == native_hash(
            'GraphKernel_Mul_Mul_Cast_Mul_Sub_Assign_fusion-op0', 'input', 2),
    })
    scalar_values = json.loads((ROOT / 'evidence/frontier329-native-normalization-scalars-values.json').read_text())
    output = next(row for row in scalar_values['records'] if '.output.0.' in row['file'])
    represented = np.asarray(output['values'], dtype=np.float32)
    checks['denominator_exact'] = record['denominator_bf16'] == output['values']
    checks['denominator_native_hash_exact'] = hashlib.sha1(bf16_bits(represented).tobytes()).hexdigest() == native_hash(
        'GraphKernel_Sqrt_Add_Cast_split-op0', 'output', 0)
    if not all(checks.values()):
        raise ValueError(checks)
    states = {}
    for step in [328, 329]:
        arrays = {}
        for family, run in [('hp', HP), ('mf', MF)]:
            report = json.loads((ROOT / 'runs' / run / ('FRONTIER_STATE-' + str(step) + '.json')).read_text())
            row = next(item for item in report['records'] if item['kind'] == 'weight')
            array = load(STATE / run / ('state-' + str(step)), row)
            arrays[family] = array if family == 'hp' else array.reshape(8, 1536, 5120).transpose(0, 2, 1)
        states[step] = [compare_arrays(np.ascontiguousarray(arrays['hp'][index]),
                                       np.ascontiguousarray(arrays['mf'][index])) for index in range(8)]
    gradients = json.loads((ROOT / 'runs' / HP / 'FRONTIER_GRADIENT-329.json').read_text())
    row = next(item for item in gradients['records'] if item['kind'] == 'postclip')
    gradient = load(STATE / HP / 'gradient-329', row).reshape(8, 1536, 5120)
    active = [{'expert': index, 'nonzero_gradient_elements': int(np.count_nonzero(value))}
              for index, value in enumerate(gradient)]
    report = {'complete': True, 'fidelity': checks, 'weight_differences_by_expert': states,
              'clipped_gradient_activity': active, 'raw_scalar_payload_bytes': sum(
                  item['payload_bytes'] for item in scalar_values['records']),
              'conclusion': '329 pre-normalization, represented denominator, normalized inputs and full NS output '
                            'are identical. Expert weight differences already exist at328; '
                            'the observed329 weight mismatch is not proof of a new329 optimizer defect.',
              'next_interval': [307, 328],
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (ROOT / 'evidence/frontier329-inherited-weight-comparison.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'fidelity': checks, 'different_experts328': [index for index, row in enumerate(states[328])
                   if row['fp32_bitwise_different']], 'gradient_activity': active,
                     'raw_scalar_payload_bytes': report['raw_scalar_payload_bytes']}))


if __name__ == '__main__':
    main()
