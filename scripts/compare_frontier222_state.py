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

"""Audit selected222 QKV weights, complete Muon buffers and clipped gradients."""
import hashlib
import json
from pathlib import Path
import numpy as np
from compare_preclip217 import compare_arrays
from compare_vocab_optimizer_trace import fidelity, read_rows
from compare_head206_interventions import source_sha
from model_hash_stream import model_record
from compare_frontier219_state import load
ROOT = Path('/alignment/phase12')
STATE = Path('/state/jt-recipe-20261014')
HP = 'hp-jtf637-norm219-state222'
MF = 'mf-frontier222-state'
BASE = 'hp-jtf637-norm219-both226'
TARGETS = {'main': 'model.layers.0.self_attn.linear_qkv.weight',
           'mtp': 'mtp.layers.0.transformer_layer.self_attn.linear_qkv.weight'}
MF_TARGETS = {'q_main': 'decoder.layers.0.self_attention.linear_q_down_proj.weight',
              'kv_main': 'decoder.layers.0.self_attention.linear_kv_down_proj.weight',
              'q_mtp': 'mtp.layers.0.transformer_layer.self_attention.linear_q_down_proj.weight',
              'kv_mtp': 'mtp.layers.0.transformer_layer.self_attention.linear_kv_down_proj.weight'}


def main():
    checks = {'hp': fidelity(HP, BASE, 22), 'mf': fidelity(MF, 'mf-resume', 22)}
    for family, run in [('hp', HP), ('mf', MF)]:
        checks[family]['exit_zero'] = (ROOT / 'runs' / (run + '.exit')).read_text().strip() == '0'
    checks['hp']['same_config'] = (ROOT / 'configs' / (BASE + '.yaml')).read_text().replace(
        BASE, HP).replace('train_iters: 226', 'train_iters: 222') == (ROOT / 'configs' / (HP + '.yaml')).read_text()
    checks['mf']['same_config'] = (ROOT / 'configs/mf-frontier219-state.yaml').read_text().replace(
        'mf-frontier219-state', MF).replace('  stop_step: 19', '  stop_step: 22') == (
        ROOT / 'configs' / (MF + '.yaml')).read_text()
    for name in ['SOURCE_REVISION.json', 'PROBE_SOURCE_MANIFEST.json', 'HEAD_GEMM_ROUTE.json',
                 'SHARED_UP_FP32_FACTOR.json', 'ADAMW_ARCHIVE_ARITHMETIC.json',
                 'Q_NS_ADJACENT_FACTOR.json', 'ROUTER_NS_NPU_SQRT_FACTOR.json',
                 'WEIGHTED_DVM_FACTOR.json', 'PACKED_SEMANTICS_FACTOR.json', 'NORM219_FACTOR.json']:
        checks['hp']['unchanged_' + name] = (ROOT / 'runs' / HP / name).read_bytes() == (
            ROOT / 'runs' / BASE / name).read_bytes()
    for name in ['HEAD_GEMM_ROUTE_CALLS.jsonl', 'Q_NS_ADJACENT_CALLS.jsonl',
                 'ROUTER_NS_NPU_SQRT_CALLS.jsonl', 'WEIGHTED_DVM_CALLS.jsonl',
                 'PACKED_SEMANTICS_CALLS.jsonl', 'NORM219_CALLS.jsonl', 'CLIP_REDUCTION.jsonl']:
        checks['hp']['same22_' + name] = read_rows(ROOT / 'runs' / HP / name) == read_rows(
            ROOT / 'runs' / BASE / name)[:22]
    manifest = json.loads((ROOT / 'runs' / HP / 'PROBE_SOURCE_MANIFEST.json').read_text())
    checks['hp']['49_frozen_sources'] = len(manifest['files']) == 49 and all(
        row['sha256'] == source_sha(row['name']) for row in manifest['files'])
    models = {family: {row['step']: {item['name']: item for item in row['records']} for row in read_rows(
        ROOT / 'runs' / run / 'MODEL_HASHES.jsonl')} for family, run in [('hp', BASE), ('mf', 'mf-modelhash226')]}
    states, links, pieces = {}, {}, {}
    for step in [221, 222]:
        arrays = {}
        for family, run in [('hp', HP), ('mf', MF)]:
            report = json.loads((ROOT / 'runs' / run / ('FRONTIER_STATE-' + str(step) + '.json')).read_text())
            source = ROOT / ('tools-f637' if family == 'hp' else 'tools') / (
                'hp_frontier222_state_probe.py' if family == 'hp' else 'mf_frontier222_state_tap.py')
            checks[family]['source_' + str(step)] = report['source_sha256'] == hashlib.sha256(source.read_bytes()).hexdigest()
            checks[family]['readonly_' + str(step)] = report['read_only'] and not report['teacher_values_used']
            checks[family]['all_states_' + str(step)] = len(report['records']) == (4 if family == 'hp' else 8)
            if family == 'mf':
                checks[family]['optimizer_step_' + str(step)] = report['optimizer_step'] == step
            for row in report['records']:
                arrays[family, row['label'], row['kind']] = load(STATE / run / ('state-' + str(step)), row)
        for label in TARGETS:
            for kind in ['weight', 'muon_m']:
                arrays['mf', label, kind] = np.concatenate([
                    arrays['mf', 'q_' + label, kind], arrays['mf', 'kv_' + label, kind]], axis=0)
            for family in ['hp', 'mf']:
                observed = model_record(TARGETS[label], arrays[family, label, 'weight'])
                expected = models[family][step][TARGETS[label]]
                links[family + '_' + str(step) + '_' + label] = all(
                    observed[key] == expected[key] for key in ['fp32_sha256', 'bf16_as_fp32_sha256'])
        states[step] = {label: {kind: compare_arrays(arrays['hp', label, kind], arrays['mf', label, kind])
                                for kind in ['weight', 'muon_m']} for label in TARGETS}
        pieces[step] = {label: {name: compare_arrays(arrays['hp', label, 'weight'][start:stop],
                                                     arrays['mf', label, 'weight'][start:stop])
                               for name, start, stop in [('q', 0, 1536), ('kv', 1536, 2048), ('rope', 2048, 2112)]}
                        for label in TARGETS}
        del arrays
    native = json.loads((ROOT / 'runs' / MF / 'FRONTIER_GRADIENTS-222.json').read_text())
    checks['mf']['four_gradient_targets'] = len(native) == 4 and {r['name'] for r in native} == set(MF_TARGETS.values())
    hp_grad = json.loads((ROOT / 'runs' / HP / 'FRONTIER_GRADIENT-222.json').read_text())
    checks['hp']['gradient_source'] = hp_grad['source_sha256'] == source_sha('hp_frontier222_state_probe.py')
    gradients = {}
    for label, name in MF_TARGETS.items():
        hp_row = next(r for r in hp_grad['records'] if r['label'] == label and r['kind'] == 'postclip')
        mf_row = next(r for r in native if r['name'] == name)
        gradients[label] = compare_arrays(load(STATE / HP / 'gradient-222', hp_row),
                                          load(STATE / MF / 'gradients', mf_row))
    assert all(all(v.values()) for v in checks.values()) and all(links.values()), checks
    report = {'complete': True, 'fidelity': checks, 'model_hash_links': links, 'states': states,
              'weight_pieces': pieces, 'clipped_gradients': gradients,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (ROOT / 'evidence/frontier222-state-comparison.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'states': states, 'weight_pieces': pieces, 'clipped_gradients': gradients}))


if __name__ == '__main__':
    main()
