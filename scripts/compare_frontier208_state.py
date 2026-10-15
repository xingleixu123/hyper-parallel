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

"""Locate208 target differences after full native/independent observer fidelity checks."""
import hashlib
import json
from pathlib import Path

import numpy as np

from compare_vocab_optimizer_trace import fidelity, read_rows
from model_hash_stream import model_record

ROOT = Path('/alignment/phase12')
STATE = Path('/state/jt-recipe-20261014')
RUNS = {'mf': 'mf-frontier208-state', 'hp': 'hp-jtf637-headrouter-state208'}
MF_TARGETS = {'q': 'decoder.layers.0.self_attention.linear_q_down_proj.weight',
              'kv': 'decoder.layers.0.self_attention.linear_kv_down_proj.weight',
              'router': 'decoder.layers.1.mlp.router.weight'}
HP_TARGETS = {'qkv': 'model.layers.0.self_attn.linear_qkv.weight', 'router': 'model.layers.1.mlp.gate.weight'}


def load(folder, row):
    value = np.load(folder / row['file'], allow_pickle=False)
    if value.dtype != np.float32 or not np.isfinite(value).all():
        raise ValueError('Expected finite FP32 actual tensor')
    if hashlib.sha256(value.tobytes()).hexdigest() != row['sha256']:
        raise ValueError('Observed array changed')
    return value


def bf16_words(value):
    words = np.ascontiguousarray(value).view(np.uint32)
    return (words + np.uint32(0x7fff) + ((words >> 16) & 1)) >> 16


def stats(left, right):
    if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError('Invalid full arrays')
    delta = left.astype(np.float64) - right.astype(np.float64)
    return {'size': int(left.size), 'exact': bool(np.array_equal(left, right)),
            'different': int(np.count_nonzero(left != right)), 'max_abs': float(np.abs(delta).max()),
            'mean_abs': float(np.abs(delta).mean()),
            'rel_l2': float(np.linalg.norm(delta) / max(np.linalg.norm(right.astype(np.float64)), 1e-300)),
            'bf16_different': int(np.count_nonzero(bf16_words(left) != bf16_words(right))),
            'hp_sha256': hashlib.sha256(left.tobytes()).hexdigest(),
            'mf_sha256': hashlib.sha256(right.tobytes()).hexdigest()}


def model_hashes(run):
    return {row['step']: {item['name']: item for item in row['records']}
            for row in read_rows(ROOT / 'runs' / run / 'MODEL_HASHES.jsonl')}


def main():
    checks = {}
    for label, run in RUNS.items():
        base = 'mf-resume' if label == 'mf' else 'hp-jtf637-sharedup-headlegacynz-routerdvm226'
        checks[label] = fidelity(run, base, 8)
        checks[label]['exit_zero'] = (ROOT / 'runs' / (run + '.exit')).read_text().strip() == '0'
        checks[label]['no_teacher_marker'] = not (ROOT / 'runs' / run / 'DIAGNOSTIC_ONLY.json').exists()
    hp_directory = ROOT / 'runs' / RUNS['hp']
    base_directory = ROOT / 'runs/hp-jtf637-sharedup-headlegacynz-routerdvm226'
    for marker in ['SOURCE_REVISION.json', 'HEAD_GEMM_ROUTE.json', 'ROUTER_NS_DVM_FACTOR.json',
                   'SHARED_UP_FP32_FACTOR.json', 'ADAMW_ARCHIVE_ARITHMETIC.json']:
        checks['hp']['unchanged_' + marker] = json.loads((hp_directory / marker).read_text()) == json.loads((base_directory / marker).read_text())
    manifest = json.loads((hp_directory / 'PROBE_SOURCE_MANIFEST.json').read_text())
    checks['hp']['49_sources'] = len(manifest['files']) == 49 and all(row['sha256'] == hashlib.sha256(
        (ROOT / 'tools-f637' / row['name']).read_bytes()).hexdigest() for row in manifest['files'])
    checks['hp']['16_head_calls'] = [row['cumulative_backward_calls'] for row in read_rows(hp_directory / 'HEAD_GEMM_ROUTE_CALLS.jsonl')] == list(range(2, 17, 2))
    checks['hp']['16_router_calls'] = read_rows(hp_directory / 'ROUTER_NS_DVM_CALLS.jsonl') == [
        {'step': step, 'total_router_calls': 2 * (step - 200)} for step in range(201, 209)]
    mf_hashes = model_hashes('mf-modelhash226')
    hp_hashes = model_hashes('hp-jtf637-sharedup-headlegacynz-routerdvm226')
    hp_hashes.update(model_hashes('hp-jtf637-headrouter-frontier210'))
    state_values, state_comparisons, model_checks = {}, {}, {}
    for step in [207, 208]:
        by_family = {}
        for family, run in RUNS.items():
            directory = ROOT / 'runs' / run
            report = json.loads((directory / ('FRONTIER_STATE-' + str(step) + '.json')).read_text())
            source_dir = 'tools' if family == 'mf' else 'tools-f637'
            source = 'mf_frontier208_state_tap.py' if family == 'mf' else 'hp_frontier208_state_probe.py'
            checks[family]['observer_source'] = report['source_sha256'] == hashlib.sha256((ROOT / source_dir / source).read_bytes()).hexdigest()
            checks[family]['read_only'] = report['read_only'] and not report['teacher_values_used']
            if family == 'mf':
                checks[family]['optimizer_step_' + str(step)] = report['optimizer_step'] == step
            arrays = {(row['label'], row['kind']): load(STATE / run / ('state-' + str(step)), row) for row in report['records']}
            if family == 'mf':
                arrays = {**arrays, **{('qkv', kind): np.concatenate([arrays['q', kind], arrays['kv', kind]], axis=0)
                                     for kind in ['weight', 'muon_m']}}
            by_family[family] = arrays
            expected = mf_hashes if family == 'mf' else hp_hashes
            for label, name in HP_TARGETS.items():
                record = model_record(name, arrays[label, 'weight'])
                model_checks[family + '_' + str(step) + '_' + label] = all(
                    record[key] == expected[step][name][key] for key in ['fp32_sha256', 'bf16_as_fp32_sha256'])
        state_values[step] = by_family
        state_comparisons[str(step)] = {label: {kind: stats(by_family['hp'][label, kind], by_family['mf'][label, kind])
                                               for kind in ['weight', 'muon_m']} for label in HP_TARGETS}
        hp_qkv, mf_qkv = by_family['hp']['qkv', 'weight'], by_family['mf']['qkv', 'weight']
        state_comparisons[str(step)]['qkv_splits'] = {label: stats(hp_qkv[part], mf_qkv[part])
                                                    for label, part in [('q', slice(0, 1536)), ('kv', slice(1536, 2112))]}
    native_rows = json.loads((ROOT / 'runs' / RUNS['mf'] / 'FRONTIER_GRADIENTS-208.json').read_text())
    checks['mf']['only_three_gradient_targets'] = {row['name'] for row in native_rows} == set(MF_TARGETS.values()) and len(native_rows) == 3
    hp_report = json.loads((hp_directory / 'FRONTIER_GRADIENT-208.json').read_text())
    gradient_comparisons = {}
    for label, name in MF_TARGETS.items():
        native = next(row for row in native_rows if row['name'] == name)
        hp_row = next(row for row in hp_report['records'] if row['label'] == label and row['kind'] == 'postclip')
        mf = load(STATE / RUNS['mf'] / 'gradients', native)
        hp = load(STATE / RUNS['hp'] / 'gradient-208', hp_row)
        gradient_comparisons[label] = stats(hp, mf)
    if not all(all(row.values()) for row in checks.values()) or not all(model_checks.values()):
        raise ValueError({'fidelity': checks, 'model_hashes': model_checks})
    report = {'complete': True, 'fidelity': checks, 'model_hashes_exact_prior_observers': model_checks,
              'states': state_comparisons, 'postclip_gradients208': gradient_comparisons,
              'hp_grad_norm208': hp_report['norm'], 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'scope': 'Two selected model weights and their actual Muon buffers at207/208; three complete native gradient arrays at208. No inference about unobserved optimizer slots.'}
    (ROOT / 'evidence/frontier208-gradient-state-comparison.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == '__main__':
    main()
