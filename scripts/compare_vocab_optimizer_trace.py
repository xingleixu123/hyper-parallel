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

"""Compare actual vocabulary optimizer coordinates against independent native state anchors."""
import json
from pathlib import Path
import numpy as np

ROOT = Path('/alignment/phase12')
STATE = Path('/state/jt-recipe-20261014')
HP = 'hp-denseadj-vocab-optimizer-coords'
MF = 'mf-vocab-optimizer-coords-r2'
BASELINE = 'hp-cann92-jtad1-pr180-192-routercontig-denseadj-castall'
PAIR = {'model.embed_tokens.weight': 'embedding.word_embeddings.weight', 'lm_head.weight': 'output_layer.weight'}
ANCHORS = {200: 'mf-warmup', 201: 'mf-probe201-r2', 202: 'mf-grad202-state',
           203: 'mf-grad203-state', 204: 'mf-grad204-state-r2', 205: 'mf-grad205-state'}


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def stats(left, right):
    left, right = np.asarray(left, dtype=np.float32), np.asarray(right, dtype=np.float32)
    if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError('Invalid compared coordinates')
    delta = np.abs(left.astype(np.float64) - right.astype(np.float64))
    equal = left == right
    def ordered(values):
        bits = values.view(np.uint32).astype(np.int64)
        return np.where(bits & 0x80000000, 0xffffffff - bits, bits + 0x80000000)
    ulps = np.abs(ordered(left) - ordered(right))
    worst = int(delta.argmax())
    return {'exact': bool(equal.all()), 'equal_count': int(equal.sum()), 'size': int(left.size),
            'max_abs': float(delta.max()), 'max_ulp': int(ulps.max()), 'worst_coordinate_slot': worst,
            'left_at_worst': float(left[worst]), 'right_at_worst': float(right[worst])}


def fidelity(run, baseline, count):
    rows = read_rows(ROOT/'runs'/run/'metrics.jsonl')
    ref = {row['step']: row for row in read_rows(ROOT/'runs'/baseline/'metrics.jsonl')}
    metrics = len(rows) == count and [r['step'] for r in rows] == list(range(201, 201 + count))
    metrics = metrics and all({k: v for k, v in row.items() if k != 'wall_time'} ==
                              {k: v for k, v in ref[row['step']].items() if k != 'wall_time'} for row in rows)
    inputs = []
    for step in range(201, 201 + count):
        filename = f'{step-1:06d}.npz'
        with np.load(ROOT/'runs'/run/'batches'/filename) as actual, np.load(ROOT/'runs'/baseline/'batches'/filename) as original:
            inputs.append(set(actual.files) == set(original.files) and all(np.array_equal(actual[k], original[k]) for k in actual.files))
    restore = json.loads((ROOT/'runs'/run/'RESTORE.json').read_text()) == json.loads((ROOT/'runs'/baseline/'RESTORE.json').read_text())
    return {'all_metrics_grad_norm_exact': metrics, 'all_inputs_exact': all(inputs), 'restore_exact': restore}


def native_state(step, name, indices):
    directory = STATE/ANCHORS[step]/f'step-{step:06d}'
    model = next(row for row in json.loads((directory/'model/manifest.json').read_text()) if row['parameter_name'] == PAIR[name])
    optimizer = next(row for row in json.loads((directory/'optimizer/manifest.json').read_text()) if row['parameter_name'] == PAIR[name])
    files = {'weight': directory/'model'/model['file']}
    files.update({key: directory/'optimizer'/optimizer['state'][key]['file'] for key in ['exp_avg', 'exp_avg_sq']})
    return {key: np.load(path, mmap_mode='r').reshape(-1)[indices].tolist() for key, path in files.items()}


def main():
    hp_rows = read_rows(ROOT/'runs'/HP/'VOCAB_OPTIMIZER_TRACE.jsonl')
    if len(hp_rows) != 10:
        raise ValueError('Expected ten HP coordinate records')
    hp_fidelity = fidelity(HP, BASELINE, 5)
    if not all(hp_fidelity.values()):
        raise ValueError(hp_fidelity)
    native = {}
    comparisons = []
    for row in hp_rows:
        step, name, indices = row['step'], row['name'], row['indices']
        record = {'step': step, 'name': name}
        for side, anchor in [('before', step-1), ('after', step)]:
            key = (anchor, name)
            if key not in native:
                native[key] = native_state(anchor, name, indices)
            record[side] = {field: stats(row[side][field], native[key][field]) for field in row[side]}
        comparisons.append(record)
    report = {'hp_fidelity': hp_fidelity, 'hp_vs_native_anchors': comparisons,
              'mf_probe_ready': False, 'scope': 'Selected 13 embedding and 541 head coordinates only; no whole-tensor equivalence claim.'}
    (ROOT/'evidence/vocab-native-coordinate-anchors.json').write_text(json.dumps([
        {'step': key[0], 'name': key[1], 'state': value} for key, value in native.items()]))
    mf_exit = ROOT/'runs'/(MF+'.exit')
    if mf_exit.exists() and mf_exit.read_text().strip() == '0':
        mf_fidelity = fidelity(MF, 'mf-resume', 6)
        mf_rows = {(r['step'], r['name']): r for r in read_rows(ROOT/'runs'/MF/'VOCAB_OPTIMIZER_TRACE.jsonl')}
        anchor_checks = []
        for row in hp_rows:
            key = (row['step'], row['name'])
            mf = mf_rows[key]
            if mf['indices'] != row['indices']:
                raise ValueError('Coordinates differ')
            anchor_checks.extend(stats(mf['after'][field], native[key][field])['exact'] for field in mf['after'])
        report['mf_fidelity'] = mf_fidelity
        report['mf_selected_native_state_exact'] = all(anchor_checks)
        report['mf_probe_ready'] = all(mf_fidelity.values()) and all(anchor_checks) and len(mf_rows) == 12
        if report['mf_probe_ready']:
            for row, record in zip(hp_rows, comparisons):
                mf = mf_rows[(row['step'], row['name'])]
                record['gradient'] = stats(row['gradient'], mf['gradient'])
    (ROOT/'evidence/vocab-optimizer-coordinate-comparison.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
