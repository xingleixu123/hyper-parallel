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

"""CPU counterexamples for the offline acceptance gate; never training evidence."""

import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile

from strict_compare_runs import _validate_output, audit as audit_runs


ROOT = Path(__file__).resolve().parents[1]
SOURCE = 'a' * 40
IMAGE = 'sha256:' + 'b' * 64


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding='utf-8')


def _fixture(directory: Path, config: dict, identity: str) -> list[dict]:
    directory.mkdir(parents=True)
    _write(directory / 'config.resolved.json', config)
    (directory / 'source-commit.txt').write_text(SOURCE)
    (directory / 'source.patch').write_bytes(b'')
    (directory / 'EXIT_CODE').write_text('0')
    rows = []
    for rank in range(8):
        _write(directory / f'initial-state-rank{rank}.json', {'synthetic_initial_state': True})
        _write(directory / f'start-rank{rank}.json', {'global_step': 0})
        _write(directory / f'complete-rank{rank}.json', {'global_step': 200})
        rank_rows = [dict(step=step, rank=rank, **{
            'training/total_loss': 4.0001, 'training/foundation_loss/lm': 3.0,
            'training/foundation_loss/mtp': 1.0, 'training/foundation_loss/aux': 0.0001,
            'training/grad_norm': 1.0, 'training/lr': 0.0001,
            'memory/device_max_allocated_gb': 2.0, 'memory/device_max_reserved_gb': 3.0,
            'performance/step_time': 1.0, 'data/step_tokens': 32768, 'data/step_samples': 8,
            'data/consumed_tokens': 32768 * step, 'data/consumed_samples': 8 * step,
            'input_identity': {'global_sha256': 'c' * 64, 'rank_count': 8},
        }) for step in range(1, 201)]
        _rows(directory, rank, rank_rows)
        if rank == 7:
            rows = rank_rows
    scheduling = directory.parent.parent / 'scheduling'
    scheduling.mkdir(exist_ok=True)
    _write(scheduling / (directory.name + '.receipt.json'), {
        'case': directory.name, 'source': SOURCE, 'patch_sha256': hashlib.sha256(b'').hexdigest(),
        'container_id': identity * 64, 'image': IMAGE, 'attempted': True, 'released': True,
        'exit_code': 0, 'devices': list(range(8)),
    })
    return rows


def _rows(directory: Path, rank: int, rows: list[dict]) -> None:
    (directory / f'metrics-rank{rank}.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))


def _seal(directory: Path) -> Path:
    root = directory.parent.parent
    receipt = root / 'scheduling' / (directory.name + '.receipt.json')
    paths = sorted(directory.iterdir()) + [receipt]
    manifest = root / (directory.name + '.manifest.json')
    entries = [{'path': path.relative_to(root).as_posix(), 'bytes': path.stat().st_size,
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest()} for path in paths]
    _write(manifest, {'files': entries})
    return manifest


def main() -> None:
    """Exercise complete and deliberately invalid synthetic captures, then clean them."""
    config = json.loads((ROOT / 'runs/r129-f637-fp32-fsdp8-0-200-r3/config.resolved.json').read_text())
    checks = []

    def audit(reference: Path, candidate: Path, start: int, end: int) -> dict:
        return audit_runs(reference, candidate, start, end,
                          reference_manifest=_seal(reference), candidate_manifest=_seal(candidate))

    def check(name: str, condition: bool) -> None:
        if not condition:
            raise RuntimeError(f'Synthetic gate check failed: {name}')
        checks.append(name)

    def rejects(name: str, function) -> None:
        try:
            function()
        except ValueError:
            checks.append(name)
        else:
            raise RuntimeError(f'Synthetic invalid input was accepted: {name}')

    with tempfile.TemporaryDirectory(prefix='jt-loss-gate-') as temporary:
        directory = Path(temporary)
        reference, candidate = directory / 'runs/reference', directory / 'runs/candidate'
        _fixture(reference, config, '1')
        baseline = _fixture(candidate, config, '2')
        check('complete_200_steps_8_ranks_pass', audit(reference, candidate, 1, 200)['status'] == 'PASS')
        altered = copy.deepcopy(baseline)
        altered[-1]['training/foundation_loss/aux'] += 0.000001
        _rows(candidate, 7, altered)
        result = audit(reference, candidate, 1, 200)
        check('rank7_step200_raw_aux_failure_not_hidden_by_weighted_pass',
              result['weighted_status'] == 'PASS' and result['unweighted_status'] == 'FAIL'
              and result['status'] == 'FAIL')
        _rows(candidate, 7, baseline[:-1])
        check('missing_rank_step_fails', audit(reference, candidate, 1, 200)['status'] == 'FAIL')
        altered[-1]['training/foundation_loss/aux'] = float('nan')
        _rows(candidate, 7, altered)
        result = audit(reference, candidate, 1, 200)
        check('nan_loss_is_unverified_not_pass', result['status'] == 'FAIL'
              and result['unweighted_status'] == 'UNVERIFIED')
        altered = copy.deepcopy(baseline)
        altered[-1]['data/consumed_samples'] += 8
        _rows(candidate, 7, altered)
        check('resumed_counters_must_match', audit(reference, candidate, 1, 200)['status'] == 'FAIL')
        altered[-1]['data/consumed_samples'] -= 8
        altered[-1]['rank'] = 0
        _rows(candidate, 7, altered)
        check('rank_labels_cannot_be_copied', audit(reference, candidate, 1, 200)['status'] == 'FAIL')
        _rows(candidate, 7, baseline)
        reference_manifest, candidate_manifest = _seal(reference), _seal(candidate)
        altered = copy.deepcopy(baseline)
        altered[-1]['training/foundation_loss/lm'] += 1
        _rows(candidate, 7, altered)
        candidate_manifest = _seal(candidate)
        _rows(candidate, 7, baseline)
        result = audit_runs(reference, candidate, 1, 200, reference_manifest=reference_manifest,
                            candidate_manifest=candidate_manifest)
        check('replacing_failed_metrics_with_reference_does_not_pass', result['status'] == 'FAIL')
        altered = copy.deepcopy(baseline)
        altered[-1]['input_identity']['global_sha256'] = 'd' * 64
        _rows(reference, 7, altered)
        _rows(candidate, 7, altered)
        check('same_cross_rank_input_error_in_both_runs_fails',
              audit(reference, candidate, 1, 200)['status'] == 'FAIL')
        _rows(reference, 7, baseline)
        _rows(candidate, 7, baseline)
        (reference / 'start-rank7.json').unlink()
        check('reference_start_marker_required', audit(reference, candidate, 1, 200)['status'] == 'FAIL')
        _write(reference / 'start-rank7.json', {'global_step': 0})
        rejects('self_comparison_rejected', lambda: audit(reference, reference, 1, 200))
        check('one_step_cannot_pass_200_step_acceptance', audit(reference, candidate, 1, 1)['status'] == 'FAIL')
        receipt_path = directory / 'scheduling/candidate.receipt.json'
        receipt = json.loads(receipt_path.read_text())
        receipt['container_id'] = '1' * 64
        _write(receipt_path, receipt)
        check('copied_execution_identity_fails', audit(reference, candidate, 1, 200)['status'] == 'FAIL')
        receipt['container_id'] = '2' * 64
        _write(receipt_path, receipt)
        (candidate / 'source-commit.txt').write_text('')
        rejects('missing_git_sha_rejected', lambda: audit(reference, candidate, 1, 200))
        (candidate / 'source-commit.txt').write_text('f' * 40)
        check('mixed_source_fails', audit(reference, candidate, 1, 200)['status'] == 'FAIL')
        (candidate / 'source-commit.txt').write_text(SOURCE)
        changed = copy.deepcopy(config)
        changed['optimizer']['adamw_config']['adamw_betas'][1] = 0.99
        _write(candidate / 'config.resolved.json', changed)
        check('optimizer_recipe_change_fails', audit(reference, candidate, 1, 200)['status'] == 'FAIL')
        changed = copy.deepcopy(config)
        changed['accelerator']['tp_size'] = 2
        changed['fsdp_config']['dp_shard_size'] = 4
        changed['activation_checkpoint']['mode'] = 'full'
        changed['checkpoint']['checkpoint_dir'] = '/synthetic/elsewhere'
        changed['training']['train_iters'] = 400
        changed['plan_overrides'].append({'match': 'synthetic.weight', 'params': {'weight': {'tp': 'shard(0)'}},
                                         'when': None, 'exact_type': False, 'module_type': None})
        _write(candidate / 'config.resolved.json', changed)
        check('declared_parallel_recompute_window_axes_allowed',
              audit(reference, candidate, 1, 200)['status'] == 'PASS')
        changed = copy.deepcopy(config)
        changed['accelerator']['cp_size'] = 2
        cp_rule = {
            'match': '*.self_attn', 'when': 'cp', 'inner_target': 'self',
            'inner_wrapper': {
                '_target_': ('hyper_parallel.models.jt_deepseek_v3.adapter.distributed.'
                            'context_parallel.mla_cp_wrapper'),
                'strategy': 'expanded_ulysses',
            },
        }
        changed['plan_overrides'].append(cp_rule)
        _write(candidate / 'config.resolved.json', changed)
        check('explicit_default_jt_cp_communication_wrapper_allowed',
              audit(reference, candidate, 1, 200)['status'] == 'PASS')
        cp_rule['inner_wrapper']['overlap'] = False
        _write(candidate / 'config.resolved.json', changed)
        check('resolved_default_cp_overlap_false_allowed',
              audit(reference, candidate, 1, 200)['status'] == 'PASS')
        for key, value in (('when', None), ('match', '*.mlp'), ('inner_target', 'other')):
            invalid = copy.deepcopy(changed)
            invalid['plan_overrides'][-1][key] = value
            _write(candidate / 'config.resolved.json', invalid)
            check(f'cp_wrapper_changed_{key}_fails', audit(reference, candidate, 1, 200)['status'] == 'FAIL')
        for key, value in (('_target_', 'different.wrapper'), ('loss_scale', 0.5),
                           ('strategy', 'unknown'), ('overlap', True)):
            invalid = copy.deepcopy(changed)
            invalid['plan_overrides'][-1]['inner_wrapper'][key] = value
            _write(candidate / 'config.resolved.json', invalid)
            check(f'cp_wrapper_changed_option_{key}_fails', audit(reference, candidate, 1, 200)['status'] == 'FAIL')
        changed['model']['config']['mtp_loss_factor'] = 0.5
        _write(candidate / 'config.resolved.json', changed)
        rejects('loss_coefficient_change_rejected', lambda: audit(reference, candidate, 1, 200))
        rejects('input_evidence_cannot_be_overwritten',
                lambda: _validate_output(candidate / 'metrics-rank0.jsonl', reference, candidate))
        rejects('execution_receipt_cannot_be_overwritten',
                lambda: _validate_output(receipt_path, reference, candidate))
        rejects('collection_manifest_cannot_be_overwritten',
                lambda: _validate_output(reference_manifest, reference, candidate, (reference_manifest,)))
    hashes = {name: hashlib.sha256((ROOT / 'tools' / name).read_bytes()).hexdigest()
              for name in ('strict_compare_runs.py', 'compare_runs.py', Path(__file__).name)}
    report = {'checked_at': datetime.now(timezone.utc).isoformat(), 'status': 'PASS', 'checks': checks,
              'script_sha256': hashes, 'synthetic_cpu_evidence': True, 'training_pass_claimed': False,
              'synthetic_files_cleaned': not directory.exists()}
    _write(ROOT / 'evidence/strict-loss-gate-validation.json', report)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
