# Copyright 2026 Huawei Technologies Co., Ltd
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

"""Screen a complete CP2 diagnostic against its separately audited partial FSDP reference."""

import hashlib
import json
from pathlib import Path

from audit_interrupted_dgrad_run import audit as audit_partial_reference
from audit_long_dgrad_run import audit as audit_complete_candidate
from strict_compare_runs import audit as strict_audit


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = 'r129-f637-dgrad-fsdp8-fp32-200-v1'
CANDIDATE = 'r129-f637-dgrad-cp2-fp32-200-v2'


def expected_incomplete_findings() -> set[str]:
    """Name each expected completeness failure without suppressing unrelated integrity findings."""
    expected = {f'Run is not successfully complete: {REFERENCE}',
                f'Missing collected input: {REFERENCE}/EXIT_CODE',
                f'Execution did not finish cleanly: {REFERENCE}',
                'Final acceptance requires exactly 200 consecutive steps'}
    for rank in range(8):
        expected.update({f'Missing completion marker: {REFERENCE}, rank {rank}',
                         f'Missing collected input: {REFERENCE}/complete-rank{rank}.json',
                         f'Candidate window differs from requested window at rank {rank}'})
    return expected


def analyze() -> dict:
    """Recompute both evidence audits and preserve the strict comparator's failed final verdict."""
    reference_manifest = ROOT / 'evidence/f637-dgrad-fsdp8-fp32-200-v1-evidence-manifest.json'
    candidate_manifest = ROOT / 'evidence/f637-dgrad-cp2-fp32-200-v2-evidence-manifest.json'
    partial = audit_partial_reference()
    if partial['observed_steps_per_rank'] != 163 or not partial['three_step_prefix_exact']:
        raise ValueError('Expected the verified 163-step FSDP prefix with exact short-arm agreement')
    complete = audit_complete_candidate(ROOT, CANDIDATE, candidate_manifest)
    if complete['status'] != 'PASS_RUN_AND_PREFIX_ONLY':
        raise ValueError('Complete CP2 execution or its short-arm prefix did not verify')
    comparison = strict_audit(ROOT / 'runs' / REFERENCE, ROOT / 'runs' / CANDIDATE, 1, 163,
                              reference_manifest=reference_manifest, candidate_manifest=candidate_manifest)
    if (set(comparison['findings']) != expected_incomplete_findings()
            or comparison['input_mismatch_count'] or comparison['initial_mismatches']):
        raise ValueError('Unexpected evidence mismatch beyond the known incomplete reference')
    helpers = ('capture_dgrad_200_training.py', 'linear_dgrad_long_observer.py',
               'linear_dgrad_trace_experiment.py', 'linear_dgrad_experiment.py')
    helper_hashes = {}
    for name in helpers:
        left = (ROOT / 'runs' / REFERENCE / name).read_bytes()
        right = (ROOT / 'runs' / CANDIDATE / name).read_bytes()
        if left != right:
            raise ValueError('Numerical helper differs across layouts: ' + name)
        helper_hashes[name] = hashlib.sha256(left).hexdigest()
    for case in (REFERENCE, CANDIDATE):
        receipt = json.loads((ROOT / 'scheduling' / f'{case}.receipt.json').read_bytes())
        if receipt['host'] != '129':
            raise ValueError('This comparison requires the same physical host')
    native = strict_audit(ROOT / 'runs/r129-f637-fp32-fsdp8-0-200-r4',
                          ROOT / 'runs/r129-f637-fp32-cp2-200', 1, 200,
                          reference_manifest=ROOT / 'evidence/f637-fsdp8-r4-complete-evidence-manifest.json',
                          candidate_manifest=ROOT / 'evidence/f637-cp2-200-complete-evidence-manifest.json')
    if native['findings'] or native['input_mismatch_count'] or native['initial_mismatches']:
        raise ValueError('Native control integrity failed')
    errors = comparison['step_errors']['training/foundation_loss/lm']
    first_over = next((index + 1 for index, value in enumerate(errors) if value > 0.005), None)
    return {'status': 'REJECT_MAIN_LM_CANDIDATE_IN_OBSERVED_PREFIX' if first_over else
            'INCONCLUSIVE_PREFIX_WITHIN_TOLERANCE', 'observed_comparison_steps': [1, 163],
            'main_lm_max_absolute_error': max(errors), 'main_lm_first_step_above_0_005': first_over,
            'native_main_lm_max_over_matching_prefix': max(native['step_errors']['training/foundation_loss/lm'][:163]),
            'matching_runtime_helper_sha256': helper_hashes, 'partial_reference_audit': partial,
            'complete_candidate_audit': complete, 'strict_comparator_result': comparison,
            'native_control_200_step_result': native, 'acceptance_200_steps': False,
            'production_performance_claim': False, 'review': 'Primary-agent self-audit; no independent review claimed',
            'scope': 'A threshold violation can reject this numerical candidate early. '
                     'A prefix within tolerance cannot establish the requested 200-step acceptance. '
                     'MTP/aux remain reported diagnostics; the current optimization target is main LM loss.'}


def main() -> None:
    """Save one non-overwriting analysis outside all captured run directories."""
    result = analyze()
    path = ROOT / 'evidence/f637-dgrad-long-cp2-v2-prefix-analysis.json'
    with path.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print(json.dumps({key: result[key] for key in ('status', 'main_lm_max_absolute_error',
                                                  'main_lm_first_step_above_0_005', 'acceptance_200_steps')}))


if __name__ == '__main__':
    main()
