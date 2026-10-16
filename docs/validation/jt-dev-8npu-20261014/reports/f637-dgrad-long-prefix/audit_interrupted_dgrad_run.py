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

"""Preserve a conflict-interrupted diagnostic as partial evidence, never acceptance."""

from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from audit_long_dgrad_run import verify_collection
from compare_runs import LOSSES, read_rows
# Reuse numerical recipe identity without changing the complete-run acceptance gate.
from strict_compare_runs import _recipe


ROOT = Path(__file__).resolve().parents[1]
CASE = 'r129-f637-dgrad-fsdp8-fp32-200-v1'
TAG = 'f637-dgrad-fsdp8-fp32-200-v1'


def audit() -> dict:
    """Verify raw bytes, the observed prefix and the distinct container ownership."""
    directory = ROOT / 'runs' / CASE
    short = ROOT / 'runs/r129-f637-dgrad-fsdp8-fp32-3'
    native = ROOT / 'runs/r129-f637-fp32-fsdp8-0-200-r4'
    manifests = [TAG, 'f637-dgrad-fsdp8-fp32-3', 'f637-fsdp8-r4-complete']
    counts = {tag: verify_collection(ROOT, ROOT / 'evidence' / f'{tag}-evidence-manifest.json')
              for tag in manifests}
    receipt = json.loads((ROOT / 'scheduling' / f'{CASE}.receipt.json').read_bytes())
    event = json.loads((ROOT / receipt['ownership_event']).read_bytes())
    if (receipt['reason'] != 'foreign-job-appeared-on-selected-devices' or receipt['exit_code'] == 0
            or receipt['released'] is not True or receipt['remaining_owned_pids']):
        raise ValueError('Expected a verified release after a foreign-job interruption')
    if (directory / 'EXIT_CODE').exists() or list(directory.glob('complete-rank*.json')):
        raise ValueError('Unexpected completed execution in this interrupted capture')
    owned = set(event['owned_pids_seen'])
    foreign = {}
    for pid in receipt['foreign_pids']:
        identity = event['identities_after_refresh'][str(pid)]
        cgroups = {line.split(':', 2)[-1] for line in identity['cgroup'].splitlines()}
        if pid in owned or any(receipt['container_id'] in value for value in cgroups):
            raise ValueError('Reported foreign process matches our container')
        if not cgroups or any(not value.startswith('/docker/') for value in cgroups):
            raise ValueError('Foreign container identity is not established')
        foreign[pid] = {'cgroups': sorted(cgroups), 'pid_namespace': identity['pid_namespace']}
    config = json.loads((directory / 'config.resolved.json').read_bytes())
    if _recipe(config) != _recipe(json.loads((short / 'config.resolved.json').read_bytes())):
        raise ValueError('Changed numerical recipe versus the reviewed short arm')
    for name in ('source-commit.txt', 'source.patch', 'source-status.txt'):
        if (directory / name).read_bytes() != (short / name).read_bytes():
            raise ValueError('Source differs from the reviewed short arm: ' + name)
    request = json.loads((ROOT / 'scheduling' / f'{CASE}.request.json').read_bytes())
    for path, digest in request['files'].items():
        runtime = directory / Path(path).name
        if runtime.exists() and hashlib.sha256(runtime.read_bytes()).hexdigest() != digest:
            raise ValueError('Captured input differs from the request: ' + path)
    errors = {name: 0.0 for name in (*LOSSES, 'training/grad_norm')}
    coverage = Counter()
    steps = None
    for rank in range(8):
        rows = read_rows(directory, rank)
        raw = [json.loads(line) for line in (directory / f'metrics-rank{rank}.jsonl').read_text().splitlines()]
        ordered = [row['step'] for row in raw]
        if steps is None:
            steps = len(ordered)
        if not 3 <= steps < 200 or ordered != list(range(1, steps + 1)):
            raise ValueError('Partial metrics are missing, duplicated or not ordered')
        original, short_rows = read_rows(native, rank), read_rows(short, rank)
        for name in ('initial-state', 'start'):
            left = json.loads((directory / f'{name}-rank{rank}.json').read_bytes())
            right = json.loads((short / f'{name}-rank{rank}.json').read_bytes())
            if name == 'start':
                if left.pop('train_iters') != 200 or right.pop('train_iters') != 3:
                    raise ValueError('Unexpected configured run lengths')
            if left != right:
                raise ValueError('Initial state or starting counters changed')
        for step, row in rows.items():
            if (row['rank'] != rank or row['input_identity']['rank_count'] != 8
                    or row['input_identity'] != original[step]['input_identity']):
                raise ValueError('Input identity does not match the original reference')
            if any(not math.isfinite(row[name]) for name in errors):
                raise ValueError('Nonfinite captured training metric')
            if step <= 3:
                for name in errors:
                    errors[name] = max(errors[name], abs(row[name] - short_rows[step][name]))
        path = directory / f'linear-dgrad-step-coverage-rank{rank}.jsonl'
        observed = [json.loads(line) for line in path.read_text().splitlines()]
        if [row['step'] for row in observed] != ordered:
            raise ValueError('Actual derivative coverage differs from the metric prefix')
        for row in observed:
            branches = row['counts']
            if row['rank'] != rank or any(not isinstance(value, int) or isinstance(value, bool) or value < 0
                                          for value in branches.values()):
                raise ValueError('Invalid rank or derivative counter')
            if not 0 < branches['fp32_dgrad'] <= branches['actual_backward'] <= branches['eligible_forward']:
                raise ValueError('Missing actual FP32 derivative coverage')
            if sum(shape['calls'] for shape in row['shapes']) != branches['eligible_forward']:
                raise ValueError('Shape call counts disagree with derivative coverage')
            coverage.update(branches)
    return {'status': 'VERIFIED_INTERRUPTED_PREFIX', 'case': CASE, 'observed_steps_per_rank': steps,
            'verified_collection_files': counts, 'actual_derivative_counts': dict(coverage),
            'three_step_prefix_max_absolute_errors': errors,
            'three_step_prefix_exact': not any(errors.values()), 'foreign_container_evidence': foreign,
            'receipt': {key: receipt[key] for key in ('exit_code', 'released', 'duration_seconds', 'reason')},
            'acceptance_200_steps': False, 'production_performance_claim': False,
            'scope': 'Self-audit of a partial diagnostic. No resumed or concatenated completion; '
                     'no paired cross-layout training verdict.'}


def main() -> None:
    """Write a separate immutable audit without modifying any captured input."""
    result = audit()
    path = ROOT / 'evidence' / f'{TAG}-interrupted-audit.json'
    with path.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
