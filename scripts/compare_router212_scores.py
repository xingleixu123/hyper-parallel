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

"""Audit native routing observations and compare expert-aligned derivative boundaries."""
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from compare_head206_interventions import source_sha
from compare_preclip212 import compare_arrays
from compare_vocab_optimizer_trace import fidelity, read_rows

ROOT = Path('/alignment/phase12')
STATE = Path('/state/jt-recipe-20261014')
HP = 'hp-jtf637-norm208-routerscores212'
MF = 'mf-routerscores212'
BASE = 'hp-jtf637-norm208-q-router-independent226'


def branch_hashes(run: str, side: str) -> dict[tuple[str, str], str]:
    """Return observed native branch hashes, retaining their semantic names."""
    directory = ROOT / 'runs' / run
    if side == 'hp':
        rows = json.loads((directory / 'MOE_BRANCHES.json').read_text())['records']
    else:
        rows = next(row for row in read_rows(directory / 'MOE_BRANCHES.jsonl') if row['step'] == 212)['records']
    return {(row['name'], row['kind']): row['sha256'] for row in rows}


def load_scores(run: str, side: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Verify every retained dump before canonicalizing its leading dimensions."""
    report = json.loads((ROOT / 'runs' / run / 'ROUTER_SCORES-212.json').read_text())
    directory = STATE / run / ('scores' if side == 'hp' else 'deep')
    values = {}
    for row in report['records']:
        value = np.load(directory / row['file'], mmap_mode='r')
        if (hashlib.sha256(value.tobytes()).hexdigest() != row['sha256']
                or list(value.shape) != row['shape'] or str(value.dtype) != row['dtype']):
            raise ValueError('Changed routing dump: ' + row['name'])
        value = value.reshape(4096, -1)
        if row['name'] in values and not np.array_equal(value, values[row['name']]):
            raise ValueError('Recomputed routing boundary changed')
        values[row['name']] = value
    return values, report


def main() -> None:
    """Reject changed trajectories before locating the first unequal actual boundary."""
    checks = {}
    for side, run, baseline, previous in [
            ('hp', HP, BASE, 'hp-jtf637-norm208-moebranches212'),
            ('mf', MF, 'mf-resume', 'mf-moebranches212')]:
        checks[side + '_exit_zero'] = (ROOT / 'runs' / (run + '.exit')).read_text().strip() == '0'
        checks.update({side + '_' + key: value for key, value in fidelity(run, baseline, 12).items()})
        checks[side + '_all_main_branches_exact'] = branch_hashes(run, side) == branch_hashes(previous, side)
    directory, baseline = ROOT / 'runs' / HP, ROOT / 'runs' / BASE
    for name in ['SOURCE_REVISION.json', 'PROBE_SOURCE_MANIFEST.json', 'HEAD_GEMM_ROUTE.json',
                 'SHARED_UP_FP32_FACTOR.json', 'ADAMW_ARCHIVE_ARITHMETIC.json',
                 'Q_NS_ADJACENT_FACTOR.json', 'ROUTER_NS_NPU_SQRT_FACTOR.json']:
        checks['unchanged_' + name] = (directory / name).read_bytes() == (baseline / name).read_bytes()
    for name in ['HEAD_GEMM_ROUTE_CALLS.jsonl', 'Q_NS_ADJACENT_CALLS.jsonl',
                 'ROUTER_NS_NPU_SQRT_CALLS.jsonl', 'CLIP_REDUCTION.jsonl']:
        checks['exact12_' + name] = read_rows(directory / name) == read_rows(baseline / name)[:12]
    manifest = json.loads((directory / 'PROBE_SOURCE_MANIFEST.json').read_text())
    checks['49_frozen_sources'] = len(manifest['files']) == 49 and all(
        row['sha256'] == source_sha(row['name']) for row in manifest['files'])
    for step in [211, 212]:
        actual = json.loads((directory / ('PRECLIP-' + str(step) + '.json')).read_text())
        expected = json.loads((ROOT / 'runs/hp-jtf637-norm208-preclip212' / (
            'PRECLIP-' + str(step) + '.json')).read_text())
        checks['hp_46_preclip_exact_' + str(step)] = len(actual['records']) == 46 and {
            row['name']: row['sha256'] for row in actual['records']} == {
                row['name']: row['sha256'] for row in expected['records']}
        checks['hp_preclip_source_' + str(step)] = actual['source_sha256'] == source_sha('hp_preclip212_hashes.py')
    hp, hp_report = load_scores(HP, 'hp')
    mf, mf_report = load_scores(MF, 'mf')
    checks['hp_13_boundaries'] = len(hp) == 13
    checks['mf_8_boundaries'] = len(mf) == 8
    checks['hp_scores_source'] = hp_report['source_sha256'] == source_sha('hp_router212_scores_observer.py')
    checks['mf_scores_source'] = mf_report['source_sha256'] == hashlib.sha256(
        (ROOT / 'tools/mf_router212_scores_tap.py').read_bytes()).hexdigest()
    hp_order, mf_order = np.argsort(hp['indices'], axis=-1), np.argsort(mf['indices'], axis=-1)
    checks['same_selected_experts'] = np.array_equal(
        np.take_along_axis(hp['indices'], hp_order, -1), np.take_along_axis(mf['indices'], mf_order, -1))
    if not all(checks.values()):
        raise ValueError(checks)
    comparisons = {name: compare_arrays(hp[name], mf[name]) for name in [
        'logits', 'scores', 'scores_dy', 'auxiliary', 'auxiliary_dy']}
    for name in ['coefficients', 'coefficients_dy']:
        comparisons[name + '_expert_aligned'] = compare_arrays(
            np.take_along_axis(hp[name], hp_order, -1), np.take_along_axis(mf[name], mf_order, -1))
    comparisons['normalization_dx_vs_mf_scores_dy'] = compare_arrays(hp['normalization_dx'], mf['scores_dy'])
    comparisons['hp_normalization_to_sigmoid_exact'] = compare_arrays(hp['normalization_dx'], hp['scores_dy'])
    report = {
        'complete': True, 'fidelity': checks, 'comparisons': comparisons,
        'indices_original_order_exact': np.array_equal(hp['indices'], mf['indices']),
        'runs': {'hp': HP, 'mf': MF}, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope': 'Readonly12: metrics/norm/input/restore and all main MoE branches exact prior verified runs; '
                 'HP also verifies46 preclip hashes211/212. MF score observer did not dump46 preclip gradients.'}
    (ROOT / 'evidence/router212-scores-comparison.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'complete': True, 'all_fidelity': all(checks.values()), 'comparisons': comparisons}))


if __name__ == '__main__':
    main()
