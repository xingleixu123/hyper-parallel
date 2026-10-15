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

"""Quantify the222 Q square-root intervention after full training replay fidelity."""
import hashlib
import json
from pathlib import Path

import numpy as np
from compare_preclip217 import compare_arrays

ROOT = Path('/alignment/phase12')
STATE = Path('/state/jt-recipe-20261014')


def main():
    reports = {mode: json.loads((ROOT / 'evidence' / ('frontier222-q-' + mode + '-replay.json')).read_text())
               for mode in ['control', 'npu']}
    for mode, report in reports.items():
        assert report['complete']
        assert all(v['momentum_exact_training'] and v[
            'weight_exact_training' if mode == 'control' else 'weight_exact_mf_training'] for v in report['checks'])
    rows = {}
    for index, label in enumerate(['main', 'mtp']):
        arrays = {}
        for name in ['pre_ns', 'normalized', 'ns_output']:
            pair = {}
            for mode in reports:
                value = np.load(STATE / ('hp-frontier222-q-' + mode + '-replay') / label / (name + '.npy'))
                record = next(v for v in reports[mode]['records'][index]['arrays'] if v['name'] == name)
                assert hashlib.sha256(value.tobytes()).hexdigest() == record['sha256']
                pair[mode] = value
            arrays[name] = compare_arrays(pair['control'], pair['npu'])
        control = reports['control']['records'][index]
        candidate = reports['npu']['records'][index]
        assert control['scalar']['square_sum'] == candidate['scalar']['square_sum']
        assert control['norm_fp64'] == candidate['norm_fp64']
        reference = control['norm_fp64'] + float(np.float32(1.0011717677116394e-7))
        rows[label] = {'boundaries': arrays, 'fp64_input_norm': control['norm_fp64'],
                       'fp64_input_norm_plus_actual_epsilon': reference,
                       'fp64_sqrt_of_rounded_fp32_sum': float(np.sqrt(np.float64(control['scalar']['square_sum']))),
                       'denominators': {mode: reports[mode]['records'][index]['scalar']['denominator'] for mode in reports},
                       'denominator_abs_error_vs_fp64_input': {mode: abs(reports[mode]['records'][index]['scalar']['denominator'] - reference) for mode in reports}}
    report = {'complete': True, 'diagnostic_only': True, 'records': rows,
              'interpretation': 'CPU/NPU placement reproduces different rounding. A closer result for one tensor does not establish general numerical superiority or a product defect.',
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (ROOT / 'evidence/frontier222-q-ns-comparison.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == '__main__':
    main()
