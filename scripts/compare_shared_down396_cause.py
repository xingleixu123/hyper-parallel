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

"""Join the actual state, both native replays and one-factor update evidence without loss claims."""
import hashlib
import json
from pathlib import Path

ROOT = Path('/alignment/phase12')


def digest(path: Path) -> str:
    """Hash a bounded source or evidence file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Require full output fidelity before attributing the396 weight difference to normalization."""
    names = {'state': 'frontier396-state-comparison', 'mf': 'frontier396-mf-builtin-stat',
             'hp': 'frontier396-hp-muon-replay', 'candidate': 'frontier396-hp-shared-down-replay',
             'fixture': 'frontier396-normalization-fixture'}
    evidence = {key: json.loads((ROOT / 'evidence' / (name + '.json')).read_text())
                for key, name in names.items()}
    state, mf, hp, candidate, fixture = (evidence[key] for key in ['state', 'mf', 'hp', 'candidate', 'fixture'])
    hp_row, candidate_row = hp['checks'][0], candidate['checks'][0]
    boundaries = hp['records'][0]['boundaries']
    checks = {
        'all_reports_complete': all(row['complete'] for row in evidence.values()),
        'training_observer_fidelity': all(all(row.values()) for row in state['fidelity'].values()),
        'previous_selected_weight_and_momentum_exact': all(
            state['states']['395'][key]['fp32_bitwise_different'] == 0 for key in ['weight', 'muon_m']),
        'current_postclip_gradient_exact': state['clipped_gradient']['fp32_bitwise_different'] == 0,
        'current_momentum_exact': state['states']['396']['muon_m']['fp32_bitwise_different'] == 0,
        'mf_replay_full_outputs_exact': all(row['exact_actual_training'] for row in mf['native_state_fidelity'].values()),
        'hp_replay_full_outputs_exact': hp_row['weight_exact_training'] and hp_row['momentum_exact_training'],
        'hp_explicit_normalization_exact': hp['records'][0]['explicit_normalization_exact'],
        'same_pre_normalization_input': fixture['input_exact_native'] and
            boundaries['pre_norm']['native_dump_sha1'] == fixture['input']['native_dump_sha1'],
        'hp_actual_normalization_matches_fixture': boundaries['normalized']['native_dump_sha1']
            == fixture['usual_normalized']['native_dump_sha1'],
        'scheduled_square_sum_exact_native': fixture['scheduled_square_sum_exact_native'],
        'scheduled_normalization_exact_native': fixture['scheduled_normalized_exact_native'],
        'candidate_full_weight_exact_mf': candidate_row['weight_exact_mf_training'] and
            candidate_row['weight_sha256'] == mf['native_state_fidelity']['weight']['actual_sha256'],
        'candidate_momentum_unchanged': candidate_row['momentum_exact_training'] and
            candidate_row['momentum_sha256'] == hp_row['momentum_sha256'],
        'factor_source_unchanged': candidate['candidate_policy']['source_sha256']
            == digest(ROOT / 'tools-f637/hp_shared_down396_factor.py'),
        'epsilon_unchanged': candidate['candidate_policy']['epsilon'] == hp['epsilon_policy']['epsilon'],
        'no_teacher_or_step_factor': not (candidate['candidate_policy']['teacher_values_used']
                                          or candidate['candidate_policy']['step_specific_behavior']),
    }
    for key in ['mf', 'hp', 'candidate', 'fixture']:
        checks[key + '_exit_zero'] = (ROOT / 'runs' / (names[key] + '.exit')).read_text().strip() == '0'
    if not all(checks.values()):
        raise ValueError(checks)
    report = {
        'complete': True, 'checks': checks, 'target': 'model.layers.1.mlp.shared_experts.down_proj.weight',
        'shape': [5120, 3072], 'step': 396, 'before_weight_difference': state['states']['396']['weight'],
        'after_weight_difference': {'fp32_bitwise_different': 0, 'bf16_different': 0},
        'normalized_bf16_different_elements': fixture['normalized_different_elements'],
        'norm': {'hp': fixture['usual_norm'], 'mf': fixture['scheduled_norm'], 'fp64': fixture['norm_fp64']},
        'fp64_normalization_distance': {'hp_closer_elements': fixture['usual_closer_to_fp64_normalized'],
                                        'mf_closer_elements': fixture['scheduled_closer_to_fp64_normalized']},
        'classification': 'Archived DVM reduction compatibility; no generic HP mathematical defect established.',
        'loss_acceptance': 'Still requires independent full200 total/raw-MTP/raw-Aux simultaneous acceptance.',
        'sources': {key: {'file': name + '.json', 'sha256': digest(ROOT / 'evidence' / (name + '.json'))}
                    for key, name in names.items()},
        'source_sha256': digest(Path(__file__)),
    }
    (ROOT / 'evidence/frontier396-causal-chain.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == '__main__':
    main()
