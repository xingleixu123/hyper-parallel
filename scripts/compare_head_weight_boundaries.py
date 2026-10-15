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

"""Audit native head derivative boundaries and compare each branch with FP64 dots."""
import hashlib
import json
from pathlib import Path

import numpy as np

from compare_vocab_optimizer_trace import fidelity, read_rows
from compare_head_selected_operands import stats

ROOT = Path('/alignment/phase12')
STATE = Path('/state/jt-recipe-20261014')
MF = 'mf-head-weight-boundaries206-r2'
HP = 'hp-jtf637-adamw-all-headcontig206-r4'


def rounded_bf16(value):
    """Round finite FP32 samples to BF16 using nearest-even integer rounding."""
    bits = np.asarray(value, dtype=np.float32).view(np.uint32)
    rounded = (bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) & np.uint32(0xffff0000)
    return rounded.view(np.float32)


def main():
    """Reject perturbed observers before interpreting branch boundaries."""
    checks = {}
    for label, run, baseline in [('mf', MF, 'mf-resume'), ('hp', HP, 'hp-jtf637-adamw-all-archive206')]:
        checks[label + '_exit_zero'] = (ROOT/'runs'/(run+'.exit')).read_text().strip() == '0'
        checks.update({label + '_' + key: value for key, value in fidelity(run, baseline, 6).items()})
    target = ROOT/'evidence/head-weight-boundaries-six-step-comparison.json'
    if not all(checks.values()):
        target.write_text(json.dumps({'complete': False, 'fidelity': checks}, indent=2))
        raise ValueError(checks)
    directory = STATE/MF/'deep'
    manifest = json.loads((directory/'manifest.json').read_text())
    indices = np.asarray(json.loads((directory/'weight-observer.json').read_text())['indices'])
    hp_calls = read_rows(ROOT/'runs'/HP/'HEAD_BACKWARD_COORDINATES.jsonl')
    hp_optim = {r['step']: r for r in read_rows(ROOT/'runs'/HP/'VOCAB_OPTIMIZER_TRACE.jsonl')
                if r['name'] == 'lm_head.weight'}
    mf_rows = read_rows(ROOT/'runs'/MF/'HEAD_WEIGHT_BOUNDARIES.jsonl')
    mf_operands = read_rows(ROOT/'runs'/MF/'HEAD_OPERANDS.jsonl')
    mf_native = {r['step']: r for r in read_rows(ROOT/'runs/mf-vocab-optimizer-coords-r2/VOCAB_OPTIMIZER_TRACE.jsonl')
                 if r['name'] == 'lm_head.weight'}
    hp_metrics = {r['step']: r for r in read_rows(ROOT/'runs'/HP/'metrics.jsonl')}
    checks['hp_selected_optimizer_trace_exact'] = (ROOT/'runs'/HP/'VOCAB_OPTIMIZER_TRACE.jsonl').read_bytes() == (
        ROOT/'runs/hp-jtf637-adamw-all-archive206/VOCAB_OPTIMIZER_TRACE.jsonl').read_bytes()
    checks['mf_weight_observer_source_hash'] = json.loads((directory/'weight-observer.json').read_text())['source_sha256'] == hashlib.sha256(
        (ROOT/'tools/mf_head_weight_boundaries_tap_r2.py').read_bytes()).hexdigest()
    if not all(checks.values()):
        raise ValueError(checks)
    output = []
    for step in range(201, 207):
        mf = {r['name']: np.asarray(r['values'], dtype=np.float32)
              for r in next(row for row in mf_rows if row['step'] == step)['records']}
        operand_record = next(row for row in mf_operands if row['step'] == step)
        arrays = {(r['name'], r['kind']): np.load(directory/r['file']) for r in operand_record['records']}
        branches = []
        hp_sum = np.zeros(len(indices), dtype=np.float32)
        mf_sum = np.zeros(len(indices), dtype=np.float32)
        for branch, call in [('model.mtp.output_layer', (step-201)*2+1), ('model.output_layer', (step-201)*2+2)]:
            native_mf = mf[branch.replace('.', '_') + '_dw']
            native_hp = np.asarray(hp_calls[call-1]['dw_at_coordinates'], dtype=np.float32)
            with np.load(ROOT/'runs'/HP/f'head-operands-call{call}.npz') as hp:
                x_comparison = stats(hp['x'], arrays[branch, 'x'])
                dy_comparison = stats(hp['dy'], arrays[branch, 'dy_matmul'])
                vocab_slots = np.searchsorted(hp['vocabulary_rows'], indices // 5120)
                hidden_slots = np.searchsorted(hp['hidden_columns'], indices % 5120)
                reference = np.sum(hp['dy'][:, vocab_slots].astype(np.float64) *
                                   hp['x'][:, hidden_slots].astype(np.float64), axis=0)
            ideal = rounded_bf16(reference.astype(np.float32))
            hp_error = np.abs(native_hp.astype(np.float64) - reference)
            mf_error = np.abs(native_mf.astype(np.float64) - reference)
            accuracy = {'reference_input': 'HP actual operands', 'both_inputs_exact': x_comparison['exact'] and dy_comparison['exact'],
                        'hp_matches_fp64_then_fp32_then_bf16': int(np.count_nonzero(native_hp == ideal)),
                        'mf_matches_fp64_then_fp32_then_bf16': int(np.count_nonzero(native_mf == ideal)),
                        'hp_closer': int(np.count_nonzero(hp_error < mf_error)),
                        'mf_closer': int(np.count_nonzero(mf_error < hp_error)),
                        'tie': int(np.count_nonzero(hp_error == mf_error)),
                        'hp_max_abs_error': float(hp_error.max()), 'mf_max_abs_error': float(mf_error.max())}
            branches.append({'branch': branch, 'hp_call': call, 'x': x_comparison, 'dy': dy_comparison,
                             'dw': stats(native_hp, native_mf), 'accuracy': accuracy})
            hp_sum += native_hp
            mf_sum += native_mf
        record = {'step': step, 'branches': branches,
                  'mf_branch_fp32_sum_vs_raw': stats(mf_sum, mf['shared_head_raw']),
                  'hp_branch_fp32_sum_vs_mf_raw': stats(hp_sum, mf['shared_head_raw']),
                  'clipped_gradient': stats(np.asarray(hp_optim[step]['gradient'], dtype=np.float32), mf['shared_head_clipped']),
                  'mf_norm': mf['shared_head_norm'].tolist()}
        hp_coefficient = np.float32(1) / (np.float32(hp_metrics[step]['grad_norm']) + np.float32(1e-6))
        mf_coefficient = np.float32(1) / (mf['shared_head_norm'][0] + np.float32(1e-6))
        record['hp_norm'] = hp_metrics[step]['grad_norm']
        record['mf_probe_gradient_vs_independent_native'] = stats(mf['shared_head_clipped'], np.asarray(mf_native[step]['gradient'], dtype=np.float32))
        record['hp_native_clip_replay'] = stats(hp_sum * hp_coefficient, np.asarray(hp_optim[step]['gradient'], dtype=np.float32))
        record['mf_native_clip_replay'] = stats(mf['shared_head_raw'] * mf_coefficient, mf['shared_head_clipped'])
        record['hp_sum_with_mf_coefficient_vs_mf'] = stats(hp_sum * mf_coefficient, mf['shared_head_clipped'])
        if not record['mf_probe_gradient_vs_independent_native']['exact']:
            raise ValueError('Native MF selected gradient changed under probe')
        output.append(record)
    report = {'complete': True, 'fidelity': checks, 'records': output,
              'scope': '541 selected head coordinates; FP64 reference accuracy only compares frameworks when actual operands match.'}
    target.write_text(json.dumps(report, indent=2))
    print(json.dumps({'complete': True, 'fidelity': checks, 'summary': [
        {'step': row['step'], 'branches': [{'branch': b['branch'], 'x_different': b['x']['different'],
         'dy_different': b['dy']['different'], 'dw_different': b['dw']['different'], 'accuracy': b['accuracy']} for b in row['branches']],
         'mf_join_different': row['mf_branch_fp32_sum_vs_raw']['different'],
         'raw_different': row['hp_branch_fp32_sum_vs_mf_raw']['different'],
         'clipped_different': row['clipped_gradient']['different']} for row in output]}))


if __name__ == '__main__':
    main()
