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

"""Audit an interrupted dense-wgrad candidate without treating its prefix as 200-step acceptance."""

import hashlib
import json
import math
from pathlib import Path

from all_rank_resources import summarize
from audit_dense_wgrad_run import HELPERS, audit as audit_complete, verify_coverage
from audit_long_dgrad_run import verify_collection
from compare_runs import LOSSES, read_rows
from strict_compare_runs import audit as strict_audit


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = 'r129-f637-dense-wgrad-fsdp8-200-v1'
CANDIDATE = 'r129-f637-dense-wgrad-cp2-200-v1'
END = 159


def expected_findings() -> set[str]:
    """List only the known interruption findings; all other differences remain errors."""
    findings = {f'Run is not successfully complete: {CANDIDATE}',
                f'Missing collected input: {CANDIDATE}/EXIT_CODE',
                f'Execution did not finish cleanly: {CANDIDATE}',
                'Final acceptance requires exactly 200 consecutive steps'}
    for rank in range(8):
        findings.update({f'Missing completion marker: {CANDIDATE}, rank {rank}',
                         f'Missing collected input: {CANDIDATE}/complete-rank{rank}.json'})
    return findings


def audit_candidate(manifest: Path, reference_audit: dict) -> dict:
    """Check every retained byte, actual branch, input identity and independent container ownership."""
    directory = ROOT / 'runs' / CANDIDATE
    native = ROOT / 'runs/r129-f637-fp32-cp2-200'
    files = verify_collection(ROOT, manifest)
    native_files = verify_collection(ROOT, ROOT / 'evidence/f637-cp2-200-complete-evidence-manifest.json')
    receipt = json.loads((ROOT / 'scheduling' / (CANDIDATE + '.receipt.json')).read_bytes())
    event = json.loads((ROOT / receipt['ownership_event']).read_bytes())
    if (receipt['reason'] != 'foreign-job-appeared-on-selected-devices' or receipt['exit_code'] != 137
            or receipt['released'] is not True or receipt['remaining_owned_pids'] or receipt['host'] != '129'
            or not receipt['foreign_pids'] or event['container_id'] != receipt['container_id']):
        raise ValueError('Expected verified own-container release after a foreign task appeared')
    foreign = {}
    for pid in receipt['foreign_pids']:
        identity = event['identities_after_refresh'][str(pid)]
        cgroups = {line.split(':', 2)[-1] for line in identity['cgroup'].splitlines()}
        if (pid in event['owned_pids_seen'] or not cgroups
                or any(not group.startswith('/docker/') or receipt['container_id'] in group for group in cgroups)):
            raise ValueError('Foreign process ownership is not established')
        foreign[pid] = {'cgroups': sorted(cgroups), 'pid_namespace': identity['pid_namespace']}
    if (directory / 'EXIT_CODE').exists() or list(directory.glob('complete-rank*.json')):
        raise ValueError('Unexpected completed-run markers in the interrupted candidate')
    request = json.loads((ROOT / 'scheduling' / (CANDIDATE + '.request.json')).read_bytes())
    for name in ('source', 'image', 'patch_sha256'):
        if receipt[name] != request[name]:
            raise ValueError('Execution receipt differs from pinned request')
    for name, digest in request['files'].items():
        runtime = directory / Path(name).name
        if runtime.exists() and hashlib.sha256(runtime.read_bytes()).hexdigest() != digest:
            raise ValueError('Captured dependency differs from the request: ' + name)
    hashes = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in HELPERS}
    if hashes != reference_audit['runtime_helper_sha256']:
        raise ValueError('The matched numerical helpers differ')
    expected_config = json.loads((native / 'config.resolved.json').read_bytes())
    expected_config['checkpoint'].update(save_ckpt=False, checkpoint_dir='/work/checkpoints/' + CANDIDATE)
    if json.loads((directory / 'config.resolved.json').read_bytes()) != expected_config:
        raise ValueError('Candidate config differs from the native control beyond disabled checkpoint output')
    if (directory / 'source-status.txt').read_text().strip():
        raise ValueError('Candidate source was dirty')
    coverage, forward_errors = [], {key: 0.0 for key in LOSSES}
    for rank in range(8):
        rows, native_rows = read_rows(directory, rank), read_rows(native, rank)
        if list(rows) != list(range(1, END + 1)):
            raise ValueError('Incomplete or unordered observed prefix')
        mode = json.loads((directory / f'dense-wgrad-long-experiment-rank{rank}.json').read_bytes())
        if (mode['modules'] != reference_audit['modules'] or mode['numeric_recipe_changed'] is not True
                or mode['steps'] != 200 or mode['cp_size'] != 2 or mode['forward_and_dgrad'] != 'native BF16'):
            raise ValueError('Unexpected actual candidate numerical mode')
        for step, row in rows.items():
            if (row['rank'] != rank or row['input_identity'] != native_rows[step]['input_identity']
                    or row['training/lr'] != native_rows[step]['training/lr']
                    or any(not math.isfinite(row[key]) for key in (*LOSSES, 'training/grad_norm'))):
                raise ValueError('Candidate input, learning rate, rank or finite values differ from the control')
        for key in forward_errors:
            forward_errors[key] = max(forward_errors[key], abs(rows[1][key] - native_rows[1][key]))
        coverage.append(verify_coverage(directory / f'dense-wgrad-step-coverage-rank{rank}.jsonl',
                                        rank, mode['modules'], expected_steps=END))
    if any(forward_errors.values()):
        raise ValueError('The candidate changed the first-step forward losses')
    return {'status': 'VERIFIED_INTERRUPTED_PREFIX', 'case': CANDIDATE, 'steps_per_rank': END,
            'verified_collection_files': files, 'verified_native_control_files': native_files,
            'runtime_helper_sha256': hashes, 'coverage': coverage, 'first_forward_max_abs': forward_errors,
            'foreign_container_evidence': foreign,
            'receipt': {key: receipt[key] for key in ('exit_code', 'released', 'duration_seconds', 'reason')},
            'acceptance_200_steps': False}


def analyze() -> dict:
    """Preserve strict FAIL and assess only a verified common prefix for early rejection."""
    manifests = {case: ROOT / 'evidence' / (case.removeprefix('r129-') + '-evidence-manifest.json')
                 for case in (REFERENCE, CANDIDATE)}
    reference_audit = audit_complete(ROOT, REFERENCE, manifests[REFERENCE])
    if reference_audit['status'] != 'PASS_EXECUTION_AND_IDENTITY_ONLY':
        raise ValueError('The complete reference audit failed')
    candidate_audit = audit_candidate(manifests[CANDIDATE], reference_audit)
    comparison = strict_audit(ROOT / 'runs' / REFERENCE, ROOT / 'runs' / CANDIDATE, 1, END,
                              reference_manifest=manifests[REFERENCE], candidate_manifest=manifests[CANDIDATE])
    if (set(comparison['findings']) != expected_findings() or len(comparison['findings']) != len(expected_findings())
            or comparison['input_mismatch_count'] or comparison['initial_mismatches']):
        raise ValueError('Unexpected paired mismatch beyond the known incomplete candidate')
    native = strict_audit(ROOT / 'runs/r129-f637-fp32-fsdp8-0-200-r4',
                          ROOT / 'runs/r129-f637-fp32-cp2-200', 1, 200,
                          reference_manifest=ROOT / 'evidence/f637-fsdp8-r4-complete-evidence-manifest.json',
                          candidate_manifest=ROOT / 'evidence/f637-cp2-200-complete-evidence-manifest.json')
    if native['findings'] or native['input_mismatch_count'] or native['initial_mismatches']:
        raise ValueError('Original recipe control failed its identity audit')
    errors = comparison['step_errors']['training/foundation_loss/lm']
    first = next((index + 1 for index, value in enumerate(errors) if value > 0.005), None)
    resources = {case: summarize([read_rows(ROOT / 'runs' / case, rank) for rank in range(8)], 1, END, 20)
                 for case in (REFERENCE, CANDIDATE)}
    return {'status': 'REJECT_MAIN_LM_CANDIDATE_IN_OBSERVED_PREFIX' if first else 'INCONCLUSIVE_PREFIX',
            'comparison_steps': [1, END], 'ranks': 8, 'main_lm_max_absolute_error': max(errors),
            'main_lm_first_step_above_0_005': first, 'acceptance_200_steps': False,
            'native_main_lm_max_over_matching_prefix': max(
                native['step_errors']['training/foundation_loss/lm'][:END]),
            'reference_audit': reference_audit, 'candidate_audit': candidate_audit,
            'strict_all_component_comparison': comparison, 'native_recipe_comparison': native,
            'resources_common_prefix': resources, 'production_performance_claim': False,
            'scope': ('Only observed steps1-159. FSDP completed200, CP2 was interrupted after159. '
                      'No concatenation, no200-step acceptance, no TP/EP/resume/recompute acceptance. '
                      'MTP/aux remain observations; main LM independently rejects this candidate.'),
            'review': 'Primary-agent self-audit; no independent reviewer claimed',
            'strict_comparator_sha256': hashlib.sha256(
                (ROOT / 'tools/strict_compare_runs.py').read_bytes()).hexdigest()}


def main() -> None:
    """Write one new derived result while leaving every raw run unchanged."""
    result = analyze()
    target = ROOT / 'evidence/f637-dense-wgrad-cp2-v1-prefix-analysis.json'
    with target.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print(json.dumps({key: result[key] for key in ('status', 'comparison_steps', 'main_lm_max_absolute_error',
                                                  'main_lm_first_step_above_0_005', 'acceptance_200_steps')}))


if __name__ == '__main__':
    main()
