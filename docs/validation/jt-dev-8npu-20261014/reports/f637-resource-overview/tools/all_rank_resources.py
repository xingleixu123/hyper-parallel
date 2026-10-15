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

"""Summarize all-rank resource observations without deciding precision acceptance."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

from strict_compare_runs import _validate_output


TIME = 'performance/step_time'
ALLOCATED = 'memory/device_max_allocated_gb'
RESERVED = 'memory/device_max_reserved_gb'
TOKENS = 'data/step_tokens'
SAMPLES = 'data/step_samples'


def summarize(rows: list[dict], start: int, end: int, warmup: int = 20) -> dict:
    """Aggregate each step across eight ranks, counting logical data only once.

    Args:
        rows: Eight rank-indexed dictionaries mapping step numbers to recorded metrics.
        start: First step in the observation window.
        end: Last step in the observation window.
        warmup: Initial steps excluded from time and throughput statistics.

    Returns:
        Resource observations and their measurement boundaries, without a loss verdict.

    Raises:
        ValueError: If ranks, steps, finite metrics or replicated data counts are invalid.
    """
    if len(rows) != 8 or start < 1 or end < start or not 0 <= warmup < end - start + 1:
        raise ValueError('Require eight ranks, a nonempty window and a smaller nonnegative warmup')
    observations = []
    for step in range(start, end + 1):
        if any(step not in values for values in rows):
            raise ValueError(f'Missing resource observation at step {step}')
        values = [rank_rows[step] for rank_rows in rows]
        for rank, row in enumerate(values):
            if (type(row.get('rank')) is not int or type(row.get('step')) is not int
                    or row['rank'] != rank or row['step'] != step):
                raise ValueError(f'Incorrect rank/step label: rank {rank}, step {step}')
            for metric in (TIME, ALLOCATED, RESERVED, TOKENS, SAMPLES):
                value = row.get(metric)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                    raise ValueError(f'Invalid {metric}: rank {rank}, step {step}')
                if metric in (TOKENS, SAMPLES) and not float(value).is_integer():
                    raise ValueError(f'Noninteger {metric}: rank {rank}, step {step}')
            if row[TIME] <= 0:
                raise ValueError(f'Nonpositive step time: rank {rank}, step {step}')
        for metric in (TOKENS, SAMPLES):
            if len({row[metric] for row in values}) != 1:
                raise ValueError(f'Logical data count differs across ranks: {metric}, step {step}')
        observations.append({
            'step': step, 'seconds': max(row[TIME] for row in values),
            'rank0_seconds': values[0][TIME], 'tokens': values[0][TOKENS], 'samples': values[0][SAMPLES],
            'allocated_gib': max(row[ALLOCATED] for row in values),
            'reserved_gib': max(row[RESERVED] for row in values),
        })
    measured = observations[warmup:]
    elapsed = sum(item['seconds'] for item in measured)
    tokens = sum(item['tokens'] for item in measured)
    return {
        'steps': [start, end], 'ranks': 8, 'warmup_steps_excluded': warmup,
        'median_step_seconds': statistics.median(item['seconds'] for item in measured),
        'summed_step_seconds': elapsed, 'supervised_tokens': tokens,
        'supervised_tokens_per_second': tokens / elapsed,
        'allocated_gib': max(item['allocated_gib'] for item in observations),
        'reserved_gib': max(item['reserved_gib'] for item in observations),
        'rank0_time_underreport_steps': sum(item['rank0_seconds'] < item['seconds'] for item in observations),
        'per_step': observations, 'precision_acceptance_claimed': False,
        'scope': ('Each step uses the maximum published callback time over all eight ranks. Logical token counts '
                  'are counted once. Device peaks remain cumulative from process start, including warmup; '
                  'timing excludes warmup and is not end-to-end wall time. This is not a precision verdict.'),
    }


def write_new_report(path: Path, result: dict) -> None:
    """Create an output exclusively, raising FileExistsError if anything exists there."""
    with path.open('x', encoding='utf-8', newline='\n') as handle:
        handle.write(json.dumps(result, ensure_ascii=False, indent=2) + '\n')


def _read_rank(path: Path) -> tuple[dict, str]:
    """Hash exactly the byte snapshot used for parsing, even if a source later changes."""
    raw = path.read_bytes()
    records = [json.loads(line) for line in raw.splitlines()]
    rows = {row['step']: row for row in records}
    if len(rows) != len(records):
        raise ValueError(f'Duplicate steps in {path}')
    return rows, hashlib.sha256(raw).hexdigest()


def main() -> None:
    """Write a separate derived report while preserving raw captures and loss audits."""
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('start', type=int)
    parser.add_argument('end', type=int)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    directory = args.directory.resolve()
    _validate_output(args.output, directory, directory)
    snapshots = [_read_rank(directory / f'metrics-rank{rank}.jsonl') for rank in range(8)]
    result = summarize([rows for rows, _ in snapshots], args.start, args.end, args.warmup)
    result['case'] = directory.name
    result['input_sha256'] = {f'metrics-rank{rank}.jsonl': digest for rank, (_, digest) in enumerate(snapshots)}
    result['summarizer_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    write_new_report(args.output, result)
    print(json.dumps({key: value for key, value in result.items() if key not in ('per_step', 'input_sha256')}))


if __name__ == '__main__':
    main()
