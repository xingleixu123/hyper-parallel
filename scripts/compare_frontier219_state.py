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

"""Separate first Muon weight differences from actual gradient and momentum differences."""
import hashlib
import json
from pathlib import Path

import numpy as np

from compare_head206_interventions import source_sha
from compare_preclip217 import compare_arrays
from compare_vocab_optimizer_trace import fidelity, read_rows
from model_hash_stream import model_record

ROOT = Path('/alignment/phase12')
STATE = Path('/state/jt-recipe-20261014')
HP = 'hp-jtf637-packed217-state219'
MF = 'mf-frontier219-state'
BASE = 'hp-jtf637-packed217-both226'
TARGETS = {'down': 'model.layers.0.mlp.down_proj.weight', 'expert': 'model.layers.1.mlp.experts.gate_up_proj'}
MF_TARGETS = {'down': 'decoder.layers.0.mlp.linear_fc2.weight', 'expert': 'decoder.layers.1.mlp.experts.weight1'}


def load(directory: Path, row: dict) -> np.ndarray:
    """Verify complete native snapshots before using them."""
    array = np.load(directory / row['file'], mmap_mode='r')
    if array.dtype != np.float32 or list(array.shape) != row['shape']:
        raise ValueError('Unexpected native state dtype or shape')
    if hashlib.sha256(array.tobytes()).hexdigest() != row['sha256']:
        raise ValueError('Native snapshot changed')
    return array


def hp_layout(label: str, array: np.ndarray) -> np.ndarray:
    """Map native MF expert flattened-input layout to the HP expert parameter layout."""
    return np.ascontiguousarray(array.reshape(8, 5120, 3072).transpose(0, 2, 1)) if label == 'expert' else array


def main() -> None:
    """Require exact observer trajectories and independent model-hash linkage."""
    checks = {'hp': fidelity(HP, BASE, 19), 'mf': fidelity(MF, 'mf-resume', 19)}
    for family, run, prior in [('hp', HP, 'hp-jtf637-packed217-preclip219'), ('mf', MF, 'mf-preclip219')]:
        checks[family]['exit_zero'] = (ROOT / 'runs' / (run + '.exit')).read_text().strip() == '0'
        checks[family]['same_config'] = (ROOT / 'configs' / (prior + '.yaml')).read_text().replace(
            prior, run) == (ROOT / 'configs' / (run + '.yaml')).read_text()
    for name in ['SOURCE_REVISION.json', 'PROBE_SOURCE_MANIFEST.json', 'HEAD_GEMM_ROUTE.json',
                 'SHARED_UP_FP32_FACTOR.json', 'ADAMW_ARCHIVE_ARITHMETIC.json',
                 'Q_NS_ADJACENT_FACTOR.json', 'ROUTER_NS_NPU_SQRT_FACTOR.json',
                 'WEIGHTED_DVM_FACTOR.json', 'PACKED_SEMANTICS_FACTOR.json']:
        checks['hp']['unchanged_' + name] = (ROOT / 'runs' / HP / name).read_bytes() == (
            ROOT / 'runs' / BASE / name).read_bytes()
    for name in ['HEAD_GEMM_ROUTE_CALLS.jsonl', 'Q_NS_ADJACENT_CALLS.jsonl',
                 'ROUTER_NS_NPU_SQRT_CALLS.jsonl', 'WEIGHTED_DVM_CALLS.jsonl',
                 'PACKED_SEMANTICS_CALLS.jsonl', 'CLIP_REDUCTION.jsonl']:
        checks['hp']['same19_' + name] = read_rows(ROOT / 'runs' / HP / name) == read_rows(
            ROOT / 'runs' / BASE / name)[:19]
    frozen = json.loads((ROOT / 'runs' / HP / 'PROBE_SOURCE_MANIFEST.json').read_text())
    checks['hp']['49_frozen_sources'] = len(frozen['files']) == 49 and all(
        row['sha256'] == source_sha(row['name']) for row in frozen['files'])
    models = {family: {row['step']: {item['name']: item for item in row['records']} for row in read_rows(
        ROOT / 'runs' / run / 'MODEL_HASHES.jsonl')}
              for family, run in [('hp', 'hp-jtf637-packed217-both-hash226'), ('mf', 'mf-modelhash226')]}
    states, links = {}, {}
    for step in [218, 219]:
        arrays = {}
        for family, run in [('hp', HP), ('mf', MF)]:
            report = json.loads((ROOT / 'runs' / run / ('FRONTIER_STATE-' + str(step) + '.json')).read_text())
            source = ROOT / ('tools-f637' if family == 'hp' else 'tools') / (
                'hp_frontier219_state_probe.py' if family == 'hp' else 'mf_frontier219_state_tap.py')
            checks[family]['observer_source_' + str(step)] = report['source_sha256'] == hashlib.sha256(
                source.read_bytes()).hexdigest()
            checks[family]['readonly_' + str(step)] = report['read_only'] and not report['teacher_values_used']
            checks[family]['all4_states_' + str(step)] = len(report['records']) == 4
            if family == 'mf':
                checks[family]['optimizer_step_' + str(step)] = report['optimizer_step'] == step
            for row in report['records']:
                array = load(STATE / run / ('state-' + str(step)), row)
                if family == 'mf':
                    array = hp_layout(row['label'], array)
                arrays[family, row['label'], row['kind']] = array
                if row['kind'] == 'weight':
                    observed = model_record(TARGETS[row['label']], array)
                    expected = models[family][step][TARGETS[row['label']]]
                    links[family + '_' + str(step) + '_' + row['label']] = all(
                        observed[key] == expected[key] for key in ['fp32_sha256', 'bf16_as_fp32_sha256'])
        states[step] = {label: {kind: compare_arrays(arrays['hp', label, kind], arrays['mf', label, kind])
                                for kind in ['weight', 'muon_m']} for label in TARGETS}
        del arrays
    native = json.loads((ROOT / 'runs' / MF / 'FRONTIER_GRADIENTS-219.json').read_text())
    checks['mf']['two_gradient_targets'] = len(native) == 2 and {row['name'] for row in native} == set(
        MF_TARGETS.values())
    gradients = json.loads((ROOT / 'runs' / HP / 'FRONTIER_GRADIENT-219.json').read_text())
    checks['hp']['gradient_observer_source'] = gradients['source_sha256'] == source_sha('hp_frontier219_state_probe.py')
    comparisons = {}
    for label, name in MF_TARGETS.items():
        hp = next(row for row in gradients['records'] if row['label'] == label and row['kind'] == 'postclip')
        mf = next(row for row in native if row['name'] == name)
        comparisons[label] = compare_arrays(load(STATE / HP / 'gradient-219', hp), load(STATE / MF / 'gradients', mf))
    if not all(all(check.values()) for check in checks.values()) or not all(links.values()):
        raise ValueError({'fidelity': checks, 'model_hash_links': links})
    report = {'complete': True, 'fidelity': checks, 'model_hash_links': links, 'states': states,
              'postclip_gradients219': comparisons, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'scope': 'Two actual Muon target weights/buffers at218/219 and full clipped gradients; no unobserved-slot claim.'}
    (ROOT / 'evidence/frontier219-state-comparison.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'fidelity': True, 'states': states, 'postclip_gradients219': comparisons}))


if __name__ == '__main__':
    main()
