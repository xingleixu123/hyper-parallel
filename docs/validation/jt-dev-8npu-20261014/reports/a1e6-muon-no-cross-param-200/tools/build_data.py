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

"""Rebuild the portable main-LM report data from four completed, hash-bound local runs."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


REPORT = Path(__file__).resolve().parents[1]
CONTROL = 'evidence/a1e6-muon-no-cross-param-200-control-v1'
RUNS = {
    'old_f': ('r129-a1e6-fsdp8-0-200-v3', 'evidence/129-a1e6-fsdp8-200-v3-evidence-manifest.json',
              'ffc52868bb02fd0f788aafcf3fa762f2528ec7556a511859d8533fc100c607ca'),
    'old_ep': ('r129-a1e6-ep4-200-v2', 'evidence/129-a1e6-ep4200-v2-evidence-manifest.json',
               'a5a21c62ae30382f6a0854c85f28bf04d1e3e4147a33dca0f6d397cb9f88b41d'),
    'new_f': ('r129-a1e6-muon-no-cross-param-fsdp8-200-v2', CONTROL + '/collections/fsdp8-v2-c1/manifest.json',
              '727e8d08bf740dbbbfc65b629ca32a5ff8ab8ee227e748fab94d716024cae0da'),
    'new_ep': ('r129-a1e6-muon-no-cross-param-ep4-200-v1', CONTROL + '/collections/ep4-v1-c1/manifest.json',
               '71e7a6e8297c3328a54b948fa50582daf73dd1eb4e8aa45c9efd094656f7aec0'),
}
RELATIONS = {'old_pair': ('old_f', 'old_ep'), 'new_pair': ('new_f', 'new_ep'),
             'cross_source_f': ('old_f', 'new_f')}
MAIN = 'training/foundation_loss/lm'


def require(condition: bool, message: str) -> None:
    """Refuse incomplete or corrupted numerical evidence."""
    if not condition:
        raise ValueError(message)


def fingerprint(path: Path) -> dict:
    """Stream source hashes while checking a stable regular file."""
    require(path.resolve() == path and path.is_file() and not path.is_symlink(), 'Unsafe input path')
    before = path.stat()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b''):
            digest.update(block)
    after = path.stat()
    require((before.st_size, before.st_mtime_ns, before.st_ino) ==
            (after.st_size, after.st_mtime_ns, after.st_ino), 'Input changed')
    return {'bytes': before.st_size, 'sha256': digest.hexdigest()}


def build(root: Path, output: Path) -> None:
    """Export only main-LM scalars and source descriptors, never checkpoint tensors or private lifecycle text."""
    provenance, data, long_rows = [], {}, []
    for label, (case, manifest_name, expected) in RUNS.items():
        manifest_path = root / manifest_name
        require(fingerprint(manifest_path)['sha256'] == expected, 'Source manifest differs')
        manifest = json.loads(manifest_path.read_bytes())
        mapping = {item['remote_path']: item['local_path'] for item in manifest.get('remote_to_local', [])}
        entries = {item['path']: item for item in manifest['files']}
        require(len(entries) == len(manifest['files']), 'Duplicate original manifest entry')
        for name, descriptor in entries.items():
            path = root / mapping.get(name, name)
            require(fingerprint(path) == {key: descriptor[key] for key in ('bytes', 'sha256')}, 'Original member differs')
        source = (root / 'runs' / case / 'source-commit.txt').read_text().strip()
        require(source == ('544526f7db8342c607244fd41009b2a74adb84c8' if label.startswith('old_')
                           else '59b25cdd15505a4f6fb1990484d4e890e3d3724c'), 'Unexpected source')
        ranks, rank_sources = [], []
        for rank in range(8):
            directory = root / 'runs' / case
            path = directory / ('metrics-rank%d.jsonl' % rank)
            rows = [json.loads(line) for line in path.read_bytes().splitlines()]
            require(len(rows) == 200 and [row['step'] for row in rows] == list(range(1, 201)), '200step coverage')
            require(json.loads((directory / ('start-rank%d.json' % rank)).read_bytes())['global_step'] == 0
                    and json.loads((directory / ('complete-rank%d.json' % rank)).read_bytes())['global_step'] == 200,
                    'Start/completion differs')
            values = []
            for row in rows:
                require(type(row['rank']) is int and row['rank'] == rank and type(row[MAIN]) in (int, float)
                        and math.isfinite(row[MAIN]), 'Invalid finite main-LM row')
                values.append(row[MAIN])
                long_rows.append((label, case, rank, row['step'], row[MAIN]))
            ranks.append(values)
            rank_sources.append({'path': path.relative_to(root).as_posix(), **fingerprint(path)})
        require(all(values == ranks[0] for values in ranks), 'Expected global main-LM rank consensus differs')
        data[label] = ranks
        provenance.append({'label': label, 'case': case, 'source_commit': source,
                           'original_manifest': {'path': manifest_name, **fingerprint(manifest_path)},
                           'all_original_file_hashes_verified': len(entries), 'main_lm_sources': rank_sources})
    summaries, differences = {}, {}
    for label, (left, right) in RELATIONS.items():
        values = [max(abs(data[left][rank][step] - data[right][rank][step]) for rank in range(8))
                  for step in range(200)]
        differences[label] = values
        exceeded = [step + 1 for step, value in enumerate(values) if value > .005]
        summaries[label] = {'left': left, 'right': right, 'max_abs': max(values),
                            'max_step': values.index(max(values)) + 1,
                            'first_exceeded_step': exceeded[0] if exceeded else None,
                            'exceeded_steps': exceeded, 'exceeded_step_count': len(exceeded),
                            'nonzero_rank_steps': sum(data[left][rank][step] != data[right][rank][step]
                                                      for rank in range(8) for step in range(200)),
                            'same_source_pair': label != 'cross_source_f'}
    output.mkdir(parents=True, exist_ok=True)
    long_path, curves_path = output / 'main-lm-allranks.csv', output / 'curves.csv'
    with long_path.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(('label', 'case', 'rank', 'step', 'main_lm'))
        writer.writerows(long_rows)
    with curves_path.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(('step', 'old_f', 'old_ep', 'new_f', 'new_ep', 'old_pair', 'new_pair',
                         'cross_source_f', 'tolerance_absolute'))
        writer.writerows((step + 1, *(data[label][0][step] for label in RUNS),
                          *(differences[label][step] for label in RELATIONS), .005) for step in range(200))
    summary = {'schema': 'no-cross-param200-mainlm-report-data-v1', 'status': 'NOT_PUBLISHED',
               'runs': provenance, 'main_lm_units': 'unweighted logged scalar',
               'ranks': 8, 'steps': 200, 'long_rows': len(long_rows), 'within_each_run_rank_consensus': True,
               'relations': summaries, 'csv': {'main-lm-allranks.csv': fingerprint(long_path),
                                              'curves.csv': fingerprint(curves_path)},
               'performance_fields_read_for_analysis': False}
    with (output / 'summary.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'status': 'BUILT_MAIN_LM_ONLY', 'relations': summaries}, ensure_ascii=False))


def main() -> None:
    """Build from retained original evidence or independently reproduce into a fresh data directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=REPORT.parents[1])
    parser.add_argument('--output-dir', type=Path, default=REPORT / 'data')
    args = parser.parse_args()
    build(args.source_root.resolve(), args.output_dir)


if __name__ == '__main__':
    main()
