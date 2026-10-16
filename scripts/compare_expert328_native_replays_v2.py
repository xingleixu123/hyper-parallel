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

"""Audit an actual first-update intervention and a neighboring normalization negative control."""
import csv
import hashlib
import json
from pathlib import Path
import struct

ROOT = Path('/home/xxl/jt-adamw-align-20261013/phase12')


def main() -> None:
    """Require native state fidelity, source provenance and complete operator-boundary equality."""
    results = []
    for step in [328, 329]:
        prefix = 'frontier' + str(step)
        load = lambda suffix: json.loads((ROOT / 'evidence' / (prefix + suffix)).read_text())
        hp = load('-hp-expert-native-replay.json')
        native = load('-mf-builtin-stat.json')
        scalars = load('-native-normalization-scalars-values.json')
        states = load('-state-comparison.json')
        checks = {
            'exit_zero': (ROOT / 'runs' / (prefix + '-hp-expert-native-replay.exit')).read_text().strip() == '0',
            'hp_complete': hp['complete'] and not hp['teacher_values_used'],
            'hp_control_exact': hp['checks'][0]['weight_exact_training'],
            'all_momentum_exact': all(row['momentum_exact_training'] for row in hp['checks']),
            'native_fidelity': native['complete'] and all(row['exact_actual_training']
                                                         for row in native['native_state_fidelity'].values()),
            'state_capture_fidelity': states['complete'] and all(all(row.values())
                                                                for row in states['fidelity'].values()),
        }
        launch = load('-hp-expert-native-replay-launch.json')
        checks['sources_and_launcher_unchanged'] = all(hashlib.sha256((ROOT / row['name']).read_bytes()).hexdigest()
                                                       == row['sha256'] for row in launch['files'])
        rows = list(csv.DictReader((ROOT / 'evidence' / (prefix + '-native-update-stat.csv')).open()))

        def native_hash(kernel: str, io: str, slot: int) -> str:
            selected = [row for row in rows if row['Op Name'] == 'Default/' + kernel
                        and row['IO'] == io and row['Slot'] == str(slot)]
            if len(selected) != 1:
                raise ValueError('Native boundary is not unique')
            return selected[0]['SHA1']

        sums = next(row['values'] for row in scalars['records'] if '.input.0.' in row['file'])
        denominator = next(row['values'] for row in scalars['records'] if '.output.0.' in row['file'])
        sum_bytes = struct.pack('<8f', *sums)
        denominator_bits = struct.unpack('<8I', struct.pack('<8f', *denominator))
        checks['native_scalar_hashes'] = (
            hashlib.sha1(sum_bytes).hexdigest() == native_hash('GraphKernel_Sqrt_Add_Cast_split-op0', 'input', 0)
            and all(value & 0xffff == 0 for value in denominator_bits)
            and hashlib.sha1(struct.pack('<8H', *(value >> 16 for value in denominator_bits))).hexdigest()
            == native_hash('GraphKernel_Sqrt_Add_Cast_split-op0', 'output', 0))
        checks['all_prenorm_exact'] = all(row['pre_norm']['native_dump_sha1'] == native_hash(
            'GraphKernel_Cast_Mul_ReduceSum_split-op0', 'input', 0) for row in hp['records'])
        candidates = []
        for record in hp['records'][1:]:
            exact = {
                'all_sums_exact': record['candidate_stats']['squared_sum'] == sums,
                'all_denominators_exact': record['denominator_bf16'] == denominator,
                'normalized_exact': record['normalized']['native_dump_sha1'] == native_hash(
                    'GraphKernel_Div_fusion-op0', 'output', 0),
                'ns_exact': record['ns_mf_layout']['native_dump_sha1'] == native_hash(
                    'GraphKernel_Mul_Mul_Cast_Mul_Sub_Assign_fusion-op0', 'input', 2),
            }
            candidates.append({'mode': record['mode'], 'exact': exact})
        checks['all_candidate_boundaries_exact'] = all(all(row['exact'].values()) for row in candidates)
        checks['expected_weight_intervention'] = all(
            row['weight_exact_mf_training'] if step == 328 else row['weight_exact_training']
            for row in hp['checks'][1:])
        if not all(checks.values()):
            raise ValueError(checks)
        results.append({'step': step, 'fidelity': checks, 'candidates': candidates,
                        'weight_checks': hp['checks']})
    screen = json.loads((ROOT / 'evidence/frontier328-hp-normalization-screen.json').read_text())
    negative = screen['complete'] and all(row['weight_exact_training'] and row['momentum_exact_training']
                                          for row in screen['checks'])
    if not negative:
        raise ValueError('Simple epsilon/sum/sqrt negative control failed')
    report = {'complete': True, 'records': results, 'simple_negative_controls_exact_hp': negative,
              'classification': 'First328 expert-down normalized input differs because of the archived reduction schedule. '
                                'Live schedule reproduction fixes the whole328 weight update; independent200 validation pending.',
              'holdout_limitation': '329 starts from already divergent328 weights. Matching329 NS output does not repair inherited weights.',
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (ROOT / 'evidence/frontier328-native-cause-audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == '__main__':
    main()
