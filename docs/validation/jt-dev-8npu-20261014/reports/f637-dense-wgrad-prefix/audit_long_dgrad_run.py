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

"""Audit a complete 200-step diagnostic and its matching three-step numerical prefix."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from compare_runs import LOSSES, read_rows
# Reuse the fixed evidence/source checks, without relaxing the original acceptance gate.
from strict_compare_runs import _execution, _recipe, _verify_manifest


def _nonnegative_integer(value: object) -> bool:
    """Exclude booleans and lossy numeric substitutes from evidence counters."""
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def verify_collection(root: Path, manifest_path: Path) -> int:
    """Verify every collected file, including the actual derivative and helper evidence."""
    manifest = json.loads(manifest_path.read_bytes())
    names = set()
    for row in manifest['files']:
        path = (root / row['path']).resolve()
        if row['path'] in names or root.resolve() not in path.parents or path.is_symlink():
            raise ValueError('Invalid or duplicate collection path')
        data = path.read_bytes()
        if len(data) != row['bytes'] or hashlib.sha256(data).hexdigest() != row['sha256']:
            raise ValueError('Collected file changed: ' + row['path'])
        names.add(row['path'])
    return len(names)


def coverage_rows(path: Path, rank: int) -> dict:
    """Require exactly 200 ordered, bounded and nonempty actual derivative observations."""
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if [row.get('step') for row in rows] != list(range(1, 201)):
        raise ValueError('Derivative coverage must contain every step exactly once in order')
    totals = Counter()
    for row in rows:
        counts = row['counts']
        if (not _nonnegative_integer(row.get('rank')) or row['rank'] != rank
                or not _nonnegative_integer(row.get('step')) or not 1 <= len(row['shapes']) <= 128):
            raise ValueError('Invalid derivative coverage rank or shape inventory')
        if any(not _nonnegative_integer(value) for value in counts.values()):
            raise ValueError('Derivative branch counts must be nonnegative integers')
        if not 0 < counts.get('fp32_dgrad', 0) <= counts.get('actual_backward', 0) <= counts['eligible_forward']:
            raise ValueError('Actual derivative coverage is missing or inconsistent')
        if any(counts.get(key, 0) > counts['actual_backward'] for key in ('native_wgrad', 'native_bias_grad')):
            raise ValueError('Native derivative coverage exceeds actual backward calls')
        if any(not _nonnegative_integer(shape['calls']) or shape['calls'] == 0 for shape in row['shapes']):
            raise ValueError('Invalid shape call counts')
        if sum(shape['calls'] for shape in row['shapes']) != counts['eligible_forward']:
            raise ValueError('Shape coverage does not match actual forward calls')
        totals.update(counts)
    return {'steps': len(rows), 'rank': rank, 'totals': dict(totals)}


def audit(root: Path, case: str, manifest_path: Path) -> dict:
    """Check execution, frozen helper hashes, input integrity and all-rank prefix equality."""
    if case not in {f'r129-f637-dgrad-{layout}-fp32-200-v{version}'
                    for layout in ('fsdp8', 'cp2') for version in (1, 2)}:
        raise ValueError('Expected one of the two prepared 200-step diagnostic runs')
    directory = root / 'runs' / case
    config = json.loads((directory / 'config.resolved.json').read_bytes())
    layout = {1: 'fsdp8', 2: 'cp2'}[config['accelerator']['cp_size']]
    short_case = f'r129-f637-dgrad-{layout}-fp32-3'
    short = root / 'runs' / short_case
    short_manifest = root / 'evidence' / f'f637-dgrad-{layout}-fp32-3-evidence-manifest.json'
    file_counts = [verify_collection(root, path) for path in (manifest_path, short_manifest)]
    source = {'commit': (directory / 'source-commit.txt').read_text().strip(),
              'patch_sha256': hashlib.sha256((directory / 'source.patch').read_bytes()).hexdigest()}
    execution, findings = _execution(directory, source)
    if (source['commit'] != (short / 'source-commit.txt').read_text().strip()
            or source['patch_sha256'] != hashlib.sha256((short / 'source.patch').read_bytes()).hexdigest()):
        findings.append('Source differs from the corresponding short arm')
    findings.extend(_verify_manifest(directory, manifest_path))
    if config['training']['train_iters'] != 200 or config['checkpoint']['save_ckpt']:
        findings.append('Expected 200 fresh steps with checkpoint output disabled')
    if _recipe(config) != _recipe(json.loads((short / 'config.resolved.json').read_bytes())):
        findings.append('Numerical config differs from the corresponding short arm')
    request = json.loads((root / 'scheduling' / f'{case}.request.json').read_bytes())
    helper_hashes = {}
    for name in ('linear_dgrad_experiment.py', 'linear_dgrad_trace_experiment.py',
                 'linear_dgrad_long_observer.py', 'capture_dgrad_200_training.py'):
        digest = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        if digest != request['files']['tools/' + name]:
            findings.append('Runtime helper differs from pinned request: ' + name)
        helper_hashes[name] = digest
    prefix_errors = {name: 0.0 for name in (*LOSSES, 'training/grad_norm')}
    input_hashes, coverage = {}, []
    for rank in range(8):
        rows, expected = read_rows(directory, rank), read_rows(short, rank)
        if sorted(rows) != list(range(1, 201)):
            raise ValueError('Expected exactly 200 complete training steps on each rank')
        if json.loads((directory / f'complete-rank{rank}.json').read_bytes())['global_step'] != 200:
            findings.append(f'Missing final completion at rank {rank}')
        if json.loads((directory / f'start-rank{rank}.json').read_bytes())['global_step'] != 0:
            findings.append(f'Unexpected resumed start at rank {rank}')
        if json.loads((directory / f'initial-state-rank{rank}.json').read_bytes()) != json.loads(
                (short / 'initial-state-rank0.json').read_bytes()):
            findings.append(f'Initial state differs at rank {rank}')
        for step, row in rows.items():
            identity = row['input_identity']
            if row['rank'] != rank or identity['rank_count'] != 8:
                findings.append(f'Invalid rank label at rank {rank}, step {step}')
            if rank == 0:
                input_hashes[step] = identity['global_sha256']
            elif identity['global_sha256'] != input_hashes[step]:
                findings.append(f'Global input identity differs across ranks at step {step}')
            if any(not math.isfinite(row[name]) for name in prefix_errors):
                findings.append(f'Nonfinite training value at rank {rank}, step {step}')
            if step <= 3:
                if identity['global_sha256'] != expected[step]['input_identity']['global_sha256']:
                    findings.append(f'Short-prefix input differs at rank {rank}, step {step}')
                for name in prefix_errors:
                    prefix_errors[name] = max(prefix_errors[name], abs(row[name] - expected[step][name]))
        coverage.append(coverage_rows(directory / f'linear-dgrad-step-coverage-rank{rank}.jsonl', rank))
        mode = json.loads((directory / f'linear-dgrad-long-experiment-rank{rank}.json').read_bytes())
        if (mode['steps'] != 200 or mode['numeric_recipe_changed'] is not True
                or mode['cp_size'] != (1 if layout == 'fsdp8' else 2)):
            findings.append(f'Numerical mode identity differs at rank {rank}')
    if any(prefix_errors.values()):
        findings.append('Observed numerical prefix differs from the corresponding reviewed short arm')
    return {'status': 'PASS_RUN_AND_PREFIX_ONLY' if not findings else 'FAIL', 'case': case,
            'short_reference': short_case, 'verified_file_counts': file_counts, 'source': source,
            'execution': execution, 'runtime_helper_sha256': helper_hashes, 'findings': findings,
            'prefix_max_absolute_errors': prefix_errors, 'actual_derivative_coverage': coverage,
            'cross_layout_200_step_acceptance': 'NOT_DECIDED_BY_THIS_AUDIT',
            'production_performance_claim': False,
            'scope': ('Complete execution and three-step numerical prefix only; '
                      'a paired 200-step LM comparison remains required')}


def main() -> None:
    """Keep the audit outside immutable captures and never overwrite earlier results."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--case', required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve().parent != args.root.resolve() / 'evidence':
        raise ValueError('Audit output must stay in the task evidence directory')
    result = audit(args.root, args.case, args.manifest)
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps({key: result[key] for key in ('status', 'case', 'findings', 'prefix_max_absolute_errors')}))
    raise SystemExit(0 if not result['findings'] else 1)


if __name__ == '__main__':
    main()
