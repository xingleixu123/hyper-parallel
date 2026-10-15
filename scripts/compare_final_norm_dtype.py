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

"""Normalize observer dtype boundaries without changing the training computation."""
from pathlib import Path
import json
import numpy as np
from compare_head_weight_boundaries import rounded_bf16
from compare_head_selected_operands import stats


def main():
    root = Path('/alignment/phase12')
    state = Path('/state/jt-recipe-20261014')
    mf_run = 'mf-finalnorm-boundaries206'
    hp_run = 'hp-jtf637-adamw-all-finalnorm206'
    mf = next(json.loads(line) for line in (root/'runs'/mf_run/'FINAL_NORM_BOUNDARIES.jsonl').read_text().splitlines()
              if json.loads(line)['step'] == 205)
    hp = json.loads((root/'runs'/hp_run/'FINAL_NORM_BOUNDARIES.json').read_text())
    records = []
    for mf_name, hp_name in [('model.decoder.final_layernorm', 'model.norm'),
                             ('model.mtp.layers.0.final_layernorm', 'mtp.layers.0.final_layernorm')]:
        mf_record = next(r for r in mf['records'] if r['name'] == mf_name and r['kind'] == 'dx')
        hp_record = next(r for r in hp['records'] if r['name'] == hp_name and r['kind'] == 'dx')
        actual = np.load(state/hp_run/'norms'/hp_record['file']).reshape(-1, 5120)
        native = np.load(state/mf_run/'deep'/mf_record['file']).reshape(-1, 5120)
        records.append({'name': hp_name, 'hp_actual_gradient_dtype': hp_record['dtype'],
                        'mf_values_all_bf16_representable': bool(np.array_equal(native, rounded_bf16(native))),
                        'hp_dx_after_bf16_round_vs_mf': stats(rounded_bf16(actual), native)})
    report = {'complete': True, 'records': records,
              'scope': 'Boundary dtype normalization for diagnosis only; the FP32 branch must still be compared at the actual shared join before changing arithmetic.'}
    (root/'evidence/finalnorm205-boundary-dtype-normalization.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == '__main__':
    main()
