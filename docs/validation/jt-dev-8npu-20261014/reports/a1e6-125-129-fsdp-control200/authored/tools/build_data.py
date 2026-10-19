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

"""Build small portable report data from pinned existing scientific evidence."""

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path


REPORT = Path(__file__).resolve().parents[1]
REFERENCE = 'r129-a1e6-fsdp8-0-200-v3'
CANDIDATE = 'r125-a1e6-fsdp8-control200-v1'
SCIENTIFIC = 'evidence/a1e6-125-fsdp-control200-v1/collection-v2/scientific-comparison-v1/'
SOURCES = {
    SCIENTIFIC + 'comparison-v1.json': '0589cc0ec73b69afa985fde02c3b2cf0ae9bd073ada6ba750b4b76ee4d403379',
    'evidence/a1e6-expert-grad-replay-root-review-v1/125-129-direct-control200-independent-v1.json':
        '0b7bdef7ee2754b699ee2d165f79ba9937ac3782d77b35375fa574a4b6a01886',
    'evidence/129-a1e6-fsdp8-200-v3-evidence-manifest.json':
        'ffc52868bb02fd0f788aafcf3fa762f2528ec7556a511859d8533fc100c607ca',
    'evidence/125-a1e6-fsdp8-control200-v1-c2-evidence-manifest.json':
        'c94c435cd29a1387971b485c7e28f70edfd7737f38e31cca809c58a931f2541c',
    'evidence/a1e6-image-k-shift-cann-alias-readonly.json':
        '597b08e6f6b97f9d4aea68cc4fef355c984ef098de613667263142855823096e',
}
FIELDS = ('training/total_loss', 'training/foundation_loss/lm', 'training/foundation_loss/mtp',
          'training/foundation_loss/aux', 'training/grad_norm', 'training/lr',
          'data/consumed_tokens', 'data/consumed_samples', 'data/step_tokens', 'data/step_samples')
CSV_FIELDS = ('step', 'lm_129_rank0', 'lm_125_rank0', 'lm_8rank_max_abs', 'tolerance_absolute')


def require(condition: bool, message: str) -> None:
    """Reject changed or incomplete source evidence."""
    if not condition:
        raise ValueError(message)


def snapshot(root: Path, name: str, expected: str, bindings: list[dict]) -> bytes:
    """Read one bounded raw snapshot and bind its exact source SHA."""
    path = root / name
    require(path.resolve() == path and path.is_file() and not path.is_symlink(), 'Unsafe source path')
    require(path.stat().st_size < 2 * 1024 ** 2, 'Source metadata is too large')
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected, 'Changed source: ' + name)
    bindings.append({'path': name, 'bytes': len(raw), 'sha256': expected})
    return raw


def build(root: Path) -> tuple[bytes, dict, dict]:
    """Recompute figure values from all sixteen original metric files."""
    bindings = []
    source = {name: json.loads(snapshot(root, name, digest, bindings)) for name, digest in SOURCES.items()}
    comparison, independent, left_manifest, right_manifest, cann = source.values()
    require(comparison['status'] == 'PASS_OBSERVED_CROSS_HOST_200X8_LOGGED_PRECISION', 'Wrong comparison status')
    require(independent['status'] ==
            'PASS_DIRECT_FULL_200X8_SCALARS_INPUT_INITIAL_RUNTIME_AND_MAPPED_FILE_CLOSURE',
            'Missing independent scientific review')
    records = []
    all_rank = []
    for case, manifest in ((REFERENCE, left_manifest), (CANDIDATE, right_manifest)):
        entries = {item['path']: item for item in manifest['files']}
        require(len(entries) == len(manifest['files']), 'Duplicate manifest path')
        ranks = []
        for rank in range(8):
            name = f'runs/{case}/metrics-rank{rank}.jsonl'
            item = entries[name]
            raw = snapshot(root, name, item['sha256'], bindings)
            require(len(raw) == item['bytes'], 'Metric byte count differs')
            rows = [json.loads(line) for line in raw.splitlines()]
            require([row['step'] for row in rows] == list(range(1, 201)), 'Incomplete 200-step trace')
            require(all(type(row['rank']) is int and row['rank'] == rank for row in rows), 'Wrong rank')
            require(all(type(row[field]) in (int, float) and math.isfinite(row[field])
                        for row in rows for field in FIELDS), 'Invalid numeric source')
            ranks.append(rows)
        all_rank.append(ranks)
    for field in FIELDS:
        measured = max(abs(all_rank[1][rank][step][field] - all_rank[0][rank][step][field])
                       for rank in range(8) for step in range(200))
        require(measured == comparison['metrics'][field]['max_abs'] ==
                independent['comparisons'][field]['max_abs_diff'] == 0, 'Scientific scalar disagreement')
        require(all(run[rank][step][field] == run[0][step][field]
                    for run in all_rank for rank in range(8) for step in range(200)),
                'Expected reduced global scalar/counter rank consensus is absent')
    for step in range(200):
        lm_key = 'training/foundation_loss/lm'
        error = max(abs(all_rank[1][rank][step][lm_key] - all_rank[0][rank][step][lm_key])
                    for rank in range(8))
        require(error == comparison['metrics'][lm_key]['per_step_max_abs'][step], 'Figure error differs')
        records.append({'step': step + 1, 'lm_129_rank0': all_rank[0][0][step][lm_key],
                        'lm_125_rank0': all_rank[1][0][step][lm_key],
                        'lm_8rank_max_abs': error, 'tolerance_absolute': .005})
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, lineterminator='\n')
    writer.writeheader()
    writer.writerows(records)
    data = stream.getvalue().encode('utf-8')
    require(cann['image'] == comparison['execution'][0]['image'] == comparison['execution'][1]['image'],
            'CANN image does not match both executions')
    aliases = [entry['symlink'] for row in cann['records'] if row['image_path'] == '/workspace/Ascend/cann'
               for entry in row['entries'] if 'symlink' in entry]
    require(aliases == ['cann-9.2.0-beta.2'], 'Unexpected fixed CANN alias')
    closure = {}
    for role, value in right_manifest['remote_scheduling'].items():
        canonical = snapshot(root, value['remote_path'], value['canonical_sha256'], bindings)
        raw = snapshot(root, value['local_path'], value['sha256'], bindings)
        require(json.loads(canonical) == json.loads(raw) and canonical != raw, 'Raw/canonical mapping differs')
        closure[role] = {'canonical_sha256': value['canonical_sha256'], 'remote_raw_sha256': value['sha256'],
                         'json_equal': True, 'raw_is_collected_member': True,
                         'canonical_is_direct_collected_member': False}
    summary = {
        'schema': '125-129-fsdp-control200-portable-summary-v1',
        'status': comparison['status'], 'source_commit': comparison['source'][0]['commit'],
        'image_sha256': cann['image'], 'cann': {'display': '9.2b2', 'image_alias': aliases[0]},
        'reference_case': REFERENCE, 'candidate_case': CANDIDATE, 'hosts': ['129', '125'],
        'steps': 200, 'ranks': 8, 'runs_per_host': 1, 'csv_rows': len(records),
        'csv_sha256': hashlib.sha256(data).hexdigest(),
        'metrics': {field: {key: value for key, value in record.items() if key != 'per_step_max_abs'}
                    for field, record in comparison['metrics'].items()},
        'derived_unweighted_max_abs': comparison['derived_unweighted_max_abs'],
        'loss_coefficients': comparison['loss_coefficients'],
        'four_loss_pairs': 6400, 'six_scalar_pairs': 9600,
        'input_identity_object_mismatch_count': comparison['input_identity_full_object_mismatch_count'],
        'cpu_initial': comparison['cpu_initial'], 'runtime_all16_equal': comparison['runtime_all16_equal'],
        'runtime_record': comparison['runtime_record'],
        'recipe_differences': [
            {'field': 'checkpoint.save_ckpt', '129': True, '125': False},
            {'field': 'checkpoint.checkpoint_dir', 'meaning': 'Changed only to the new case output'}],
        'recorded_data_file_sha256': comparison['data_requests'][0]['recorded_data_file_sha256'],
        'raw_canonical_mapping': closure, 'original_strict': comparison['original_strict_result'],
        'independent_input_review': {
            'source_sha256': SOURCES[list(SOURCES)[1]], 'status': independent['status'],
            'method': independent['method']},
        'performance_included': False,
        'limits': ['One ordinary FSDP8 run per host, not eight independent repetitions.',
                   'global_sha256 is an unordered global component fingerprint, not token UID or DP-slot proof.',
                   'CPU initial evidence is 48 recorded tensor fingerprints per rank, not optimizer internal state.',
                   'No claim about parallel-matrix acceptance, kernel root cause, or performance benefit.'],
    }
    provenance = {'schema': '125-129-control200-report-provenance-v1', 'source_path_base': 'task root',
                  'original_sources_read_and_sha_verified': bindings,
                  'transformation': (
                      'Whitelisted scientific fields only; metric16 raw snapshots produce200 CSV rows. '
                      'Private lifecycle paths, full container IDs, checkpoint and donor files are not copied.'),
                  'raw_retained_at_source': True, 'raw_training_text_or_weights_in_report': False}
    return data, summary, provenance


def main() -> None:
    """Write only new data files, with no remote or training operations."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--task-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=REPORT)
    args = parser.parse_args()
    root, output = args.task_root.resolve(), args.output_dir.resolve()
    require(output != root and root / 'runs' not in output.parents, 'Unsafe output path')
    outputs = (output / 'data/control200.csv', output / 'data/summary.json', output / 'evidence/source-provenance.json')
    require(all(not path.exists() for path in outputs), 'Existing data files are preserved')
    csv_raw, summary, provenance = build(root)
    values = (csv_raw, (json.dumps(summary, ensure_ascii=False, indent=2) + '\n').encode('utf-8'),
              (json.dumps(provenance, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
    for path, raw in zip(outputs, values):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(raw)
    print(json.dumps({'status': 'BUILT_FROM_PINNED_REAL_SOURCE', 'csv_rows': 200,
                      'source_snapshots': len(provenance['original_sources_read_and_sha_verified'])}))


if __name__ == '__main__':
    main()
