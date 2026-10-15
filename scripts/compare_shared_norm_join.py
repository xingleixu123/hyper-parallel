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

"""Audit both native final-norm observers and compare full205 operator boundaries."""
import hashlib
import json
from pathlib import Path

import numpy as np

from compare_head_selected_operands import stats
from compare_head_weight_boundaries import rounded_bf16
from compare_vocab_optimizer_trace import fidelity, read_rows

ROOT = Path('/alignment/phase12')
STATE = Path('/state/jt-recipe-20261014')
MF = 'mf-sharednorm-join206'
HP = 'hp-jtf637-adamw-all-sharednorm206-r2'
PAIRS = [('model.decoder.final_layernorm', 'model.norm'),
         ('model.mtp.layers.0.hnorm', 'mtp.layers.0.hnorm')]


def main():
    """Use no comparison unless six-step trajectory and actual observer source pass."""
    checks = {}
    for label, run, baseline in [('mf', MF, 'mf-resume'), ('hp', HP, 'hp-jtf637-adamw-all-archive206')]:
        checks[label + '_exit_zero'] = (ROOT/'runs'/(run+'.exit')).read_text().strip() == '0'
        if not checks[label + '_exit_zero']:
            raise ValueError('Incomplete probe cannot be compared: ' + run)
        checks.update({label + '_' + key: value for key, value in fidelity(run, baseline, 6).items()})
    checks['hp_selected_optimizer_trace_exact'] = (ROOT/'runs'/HP/'VOCAB_OPTIMIZER_TRACE.jsonl').read_bytes() == (
        ROOT/'runs/hp-jtf637-adamw-all-archive206/VOCAB_OPTIMIZER_TRACE.jsonl').read_bytes()
    mf_manifest = json.loads((STATE/MF/'deep/manifest.json').read_text())
    hp_manifest = json.loads((ROOT/'runs'/HP/'FINAL_NORM_OBSERVER.json').read_text())
    checks['mf_source_hash'] = mf_manifest['source_sha256'] == hashlib.sha256(
        (ROOT/'tools/mf_shared_norm_join_tap.py').read_bytes()).hexdigest()
    checks['hp_source_hash'] = hp_manifest['source_sha256'] == hashlib.sha256(
        (ROOT/'tools-f637/hp_shared_norm_join_factor.py').read_bytes()).hexdigest()
    target = ROOT/'evidence/sharednorm205-full-boundary-comparison.json'
    if not all(checks.values()):
        target.write_text(json.dumps({'complete': False, 'fidelity': checks}, indent=2))
        raise ValueError(checks)
    mf_rows = next(r for r in read_rows(ROOT/'runs'/MF/'FINAL_NORM_BOUNDARIES.jsonl') if r['step'] == 205)
    hp_rows = json.loads((ROOT/'runs'/HP/'FINAL_NORM_BOUNDARIES.json').read_text())
    mf_map = {(r['name'], r['kind']): r for r in mf_rows['records']}
    hp_map = {(r['name'], r['kind']): r for r in hp_rows['records']}
    results = []
    for mf_name, hp_name in PAIRS:
        for kind in ['x', 'weight', 'inverse_rms', 'y_fp32', 'dy', 'dx']:
            mf_record = mf_map[mf_name, kind]
            hp_record = hp_map[hp_name, kind]
            mf = np.load(STATE/MF/'deep'/mf_record['file'], mmap_mode='r')
            hp = np.load(STATE/HP/'norms'/hp_record['file'], mmap_mode='r')
            for array, record in [(mf, mf_record), (hp, hp_record)]:
                if hashlib.sha256(array.tobytes()).hexdigest() != record['sha256']:
                    raise ValueError('Norm dump changed')
            width = 1 if kind == 'inverse_rms' else 5120
            mf = mf.reshape(-1, width)
            hp = hp.reshape(-1, width)
            result = stats(hp, mf)
            if kind == 'dx':
                result['after_bf16_cast'] = stats(rounded_bf16(hp), mf)
            indices = np.argwhere(hp != mf)
            samples = [{'row': int(i), 'column': int(j), 'hp': float(hp[i,j]), 'mf': float(mf[i,j])}
                       for i,j in indices[:20]]
            results.append({'name': hp_name, 'mf_name': mf_name, 'kind': kind,
                            'comparison': result, 'first_different_examples': samples})
    def load_mf(name, kind):
        return np.load(STATE/MF/'deep'/mf_map[name,kind]['file'], mmap_mode='r').reshape(-1,5120)
    def load_hp(name, kind):
        return np.load(STATE/HP/'norms'/hp_map[name,kind]['file'], mmap_mode='r').reshape(-1,5120)
    mf_y = load_mf('model.decoder', 'y')
    hp_y = load_hp('model.layers.1', 'y')
    mf_joint = load_mf('model.decoder', 'dy')
    hp_joint = load_hp('model.layers.1', 'dy')
    hp_sum = load_hp('model.norm', 'dx') + load_hp('mtp.layers.0.hnorm', 'dx')
    mf_early_sum = rounded_bf16(load_mf('model.decoder.final_layernorm', 'dx') +
                              load_mf('model.mtp.layers.0.hnorm', 'dx'))
    joins = {'shared_forward': stats(hp_y,mf_y),
             'hp_joint_dtype': hp_map['model.layers.1','dy']['dtype'],
             'hp_sum_vs_actual_hp_joint': stats(hp_sum,hp_joint),
             'hp_joint_after_bf16_vs_actual_mf_joint': stats(rounded_bf16(hp_joint),mf_joint),
             'mf_early_branch_sum_vs_actual_mf_joint': stats(mf_early_sum,mf_joint),
             'hp_late_sum_vs_actual_mf_joint': stats(rounded_bf16(hp_sum),mf_joint)}
    report = {'joins': joins, 'complete': True, 'step': 205, 'fidelity': checks, 'records': results,
              'scope': 'Complete actual finalRMSNorm operatorinputs/outputs andbranchderivatives; no reference injection.'}
    target.write_text(json.dumps(report, indent=2))
    print(json.dumps({'joins': joins, 'fidelity': checks, 'records': [{k:v for k,v in row.items() if k!='first_different_examples'} for row in results]}))


if __name__ == '__main__':
    main()
