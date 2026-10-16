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

"""Audit complete dense-wgrad execution and identity, without claiming paired precision."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from audit_long_dgrad_run import verify_collection
from compare_runs import LOSSES, read_rows
from strict_compare_runs import _execution, _trace_integrity, _verify_manifest


HELPERS = ('dense_wgrad_experiment.py', 'dense_wgrad_long_observer.py',
           'capture_dense_wgrad_200_training.py')
CONTROLS = {'fsdp8': ('r129-f637-fp32-fsdp8-0-200-r4', 'f637-fsdp8-r4-complete'),
            'cp2': ('r129-f637-fp32-cp2-200', 'f637-cp2-200-complete')}


def integer(value: object, minimum: int = 0) -> bool:
    """Reject booleans and rounded substitutes for integer evidence."""
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def verify_coverage(path: Path, rank: int, modules: dict, *, expected_steps: int = 200) -> dict:
    """Require actual accumulation for every selected module at every optimizer step."""
    lines = path.read_bytes().splitlines()
    rows = [json.loads(line) for line in lines]
    if (not integer(expected_steps, 1) or expected_steps > 200 or len(rows) != expected_steps
            or any(len(line) + 1 > 64 * 1024 for line in lines)):
        raise ValueError('Dense coverage is incomplete or exceeds the per-step bound')
    totals = Counter()
    native = Counter()
    for step, row in enumerate(rows, start=1):
        if (not integer(row['rank']) or row['rank'] != rank or not integer(row['step'], 1)
                or row['step'] != step or not integer(row['checked_owner_buffers'], 1)):
            raise ValueError('Invalid dense coverage rank, order or owner count')
        records = row['records']
        if not 1 <= len(records) <= 512:
            raise ValueError('Invalid dense coverage inventory size')
        counts, seen = {}, set()
        for entry in records:
            name, event = entry['module'], entry['event']
            if name not in modules or event not in ('forward', 'fp32_wgrad') or not integer(entry['calls'], 1):
                raise ValueError('Unknown module/event or invalid actual call count')
            inputs, weights = entry['input_shape'], entry['weight_shape']
            if (len(inputs) < 2 or len(weights) != 2 or not all(integer(v, 1) for v in (*inputs, *weights))
                    or inputs[-1] != weights[-1] or not integer(entry['row_start'])):
                raise ValueError('Invalid Linear shape or row offset')
            signature = (name, tuple(inputs), tuple(weights), entry['row_start'])
            key = (signature, event)
            if key in seen:
                raise ValueError('Duplicated dense coverage entry')
            seen.add(key)
            counts[key] = entry['calls']
            totals[event] += entry['calls']
        actual_modules = {key[0][0] for key in counts if key[1] == 'fp32_wgrad'}
        if actual_modules != set(modules) or 'lm_head' not in actual_modules:
            raise ValueError('Not every selected Linear had an actual FP32 weight-gradient calculation')
        for (signature, event), count in counts.items():
            if event == 'fp32_wgrad' and count > counts.get((signature, 'forward'), 0):
                raise ValueError('Actual weight-gradient calls exceed observed forwards')
        if any(not integer(count) for count in row['unscoped_native_forward'].values()):
            raise ValueError('Invalid native unscoped call count')
        native.update(row['unscoped_native_forward'])
    return {'rank': rank, 'steps': expected_steps, 'selected_modules': len(modules),
            'calls': dict(totals), 'unscoped_native_forward': dict(native)}


def audit(root: Path, case: str, manifest: Path) -> dict:
    """Validate raw inputs, changed-recipe helpers, counters and real source/execution identity."""
    valid = {f'r129-f637-dense-wgrad-{layout}-200-v1': layout for layout in CONTROLS}
    if case not in valid:
        raise ValueError('Expected a prepared dense-weight-gradient case')
    layout = valid[case]
    directory = root / 'runs' / case
    control_case, control_tag = CONTROLS[layout]
    control = root / 'runs' / control_case
    control_manifest = root / 'evidence' / (control_tag + '-evidence-manifest.json')
    counts = [verify_collection(root, path) for path in (manifest, control_manifest)]
    request = json.loads((root / 'scheduling' / (case + '.request.json')).read_bytes())
    config = json.loads((directory / 'config.resolved.json').read_bytes())
    expected = json.loads((control / 'config.resolved.json').read_bytes())
    expected['checkpoint'].update(save_ckpt=False, checkpoint_dir='/work/checkpoints/' + case)
    if config != expected or config['training']['train_iters'] != 200:
        raise ValueError('Resolved config changed beyond the disabled checkpoint destination')
    source = {'commit': (directory / 'source-commit.txt').read_text().strip(),
              'patch_sha256': hashlib.sha256((directory / 'source.patch').read_bytes()).hexdigest()}
    if (source['commit'] != request['source'] or source['patch_sha256'] != request['patch_sha256']
            or (directory / 'source-status.txt').read_text().strip()):
        raise ValueError('Actual source differs from the pinned clean checkout')
    execution, findings = _execution(directory, source)
    findings.extend(_verify_manifest(directory, manifest))
    findings.extend(_trace_integrity(control, directory, 1, 200))
    if execution.get('image') != request['image'] or (directory / 'EXIT_CODE').read_text().strip() != '0':
        findings.append('Runtime image or exit differs from the prepared successful experiment')
    hashes = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in HELPERS}
    if any(value != request['files']['tools/' + name] for name, value in hashes.items()):
        raise ValueError('Actual numerical helper differs from the pinned request')
    config_sha = hashlib.sha256((directory / 'config.yaml').read_bytes()).hexdigest()
    if config_sha != request['files']['configs/' + case + '.yaml']:
        raise ValueError('Runtime config bytes differ from the pinned parser input')
    expected_initial = json.loads((control / 'initial-state-rank0.json').read_bytes())
    control_rows = read_rows(control, 0)
    first_forward_errors = {name: 0.0 for name in LOSSES}
    coverage, inventories = [], []
    for rank in range(8):
        rows = read_rows(directory, rank)
        if list(rows) != list(range(1, 201)):
            raise ValueError('Expected all 200 ordered steps on each rank')
        if json.loads((directory / f'complete-rank{rank}.json').read_bytes())['global_step'] != 200:
            raise ValueError('Missing final completion')
        if json.loads((directory / f'start-rank{rank}.json').read_bytes())['global_step'] != 0:
            raise ValueError('Unexpected resumed start')
        if json.loads((directory / f'initial-state-rank{rank}.json').read_bytes()) != expected_initial:
            raise ValueError('Initial parameters differ from the native control')
        for step, row in rows.items():
            if (row['rank'] != rank or row['input_identity']['rank_count'] != 8
                    or row['input_identity']['global_sha256'] != control_rows[step]['input_identity']['global_sha256']):
                raise ValueError('Global input identity differs from the native control')
            if any(not math.isfinite(row[name]) for name in (*LOSSES, 'training/grad_norm')):
                raise ValueError('Nonfinite training value')
        for name in first_forward_errors:
            first_forward_errors[name] = max(first_forward_errors[name], abs(rows[1][name] - control_rows[1][name]))
        mode = json.loads((directory / f'dense-wgrad-long-experiment-rank{rank}.json').read_bytes())
        if (mode['numeric_recipe_changed'] is not True or mode['steps'] != 200
                or mode['cp_size'] != (1 if layout == 'fsdp8' else 2)
                or mode['forward_and_dgrad'] != 'native BF16'):
            raise ValueError('Actual numerical mode does not match the candidate')
        inventories.append(mode['modules'])
        coverage.append(verify_coverage(directory / f'dense-wgrad-step-coverage-rank{rank}.jsonl', rank,
                                        mode['modules']))
    if any(first_forward_errors.values()) or any(item != inventories[0] for item in inventories):
        raise ValueError('First forward changed or selected module inventory differs across ranks')
    return {'status': 'PASS_EXECUTION_AND_IDENTITY_ONLY' if not findings else 'FAIL', 'case': case,
            'source': source, 'execution': execution, 'verified_file_counts': counts,
            'runtime_helper_sha256': hashes, 'native_identity_control': control_case,
            'first_step_forward_loss_errors': first_forward_errors, 'coverage': coverage,
            'modules': inventories[0], 'findings': findings, 'cross_layout_precision_acceptance': 'NOT_EVALUATED',
            'scope': ('Complete run and identity audit only; '
                      'matching changed-recipe FSDP/CP2 comparison remains required')}


def main() -> None:
    """Write a new report outside immutable inputs without replacing previous results."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--case', required=True)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.root, args.case, args.manifest)
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False))
    if result['status'] == 'FAIL':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
