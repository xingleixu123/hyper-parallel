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

"""Extract small five-arm report data without opening the large donor tensors."""

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import struct


REPORT = Path(__file__).resolve().parents[1]
CORE = 'evidence/a1e6-expert-grad-replay-core-v2'
CONTROL = 'evidence/a1e6-expert-grad-replay-control-v2'
SOURCES = {
    CORE + '/intervention-review-tools/actual-comparison-v1.json':
        '300152f2995e1af14327470706be70a774908e8e23c5dbf344f2edfd383e0310',
    CORE + '/intervention-review-tools/actual-bindings-v1.json':
        'e66b4f58ffff9a96d2d120a04ad7ec931ea7a1fb875be23fec568550e860633f',
    'evidence/a1e6-expert-grad-replay-root-review-v1/five-role-reader-actual-execution-v1.json':
        'ffbd981444182e101d4846c788d872cd14062fbf3311d96efe238a4d995b55d6',
}
ROLES = ('capture', 'self', 'norm', 'grad', 'both')
LABELS = {'capture': 'F', 'self': 'S', 'norm': 'N', 'grad': 'G', 'both': 'B'}
SCALARS = ('training/total_loss', 'training/foundation_loss/lm', 'training/foundation_loss/mtp',
           'training/foundation_loss/aux', 'training/grad_norm', 'training/lr')
STAGES = ('gradient_before', 'gradient_after', 'parameter_before', 'after_clip',
          'optimizer_prepared', 'optimizer_actual_read', 'parameter_after')


def require(value: bool, message: str) -> None:
    """Reject changed, incomplete or nonfinite original records."""
    if not value:
        raise ValueError(message)


def read(root: Path, name: str, expected: str, bindings: list[dict]) -> bytes:
    """Read exactly one SHA-verified bounded metadata snapshot."""
    path = root / name
    require(path.resolve() == path and path.is_file() and not path.is_symlink(), 'Unsafe source path')
    require(path.stat().st_size < 2 * 1024 ** 2, 'This builder never opens a large payload')
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected, 'Changed source: ' + name)
    bindings.append({'path': name, 'bytes': len(raw), 'sha256': expected})
    return raw


def fp32(hexadecimal: str) -> tuple[float, int]:
    """Decode the original four little-endian bytes without rounding their display."""
    raw = bytes.fromhex(hexadecimal)
    require(len(raw) == 4, 'A native norm must contain exactly four bytes')
    value, bits = struct.unpack('<f', raw)[0], struct.unpack('<I', raw)[0]
    require(math.isfinite(value) and value > 0, 'Invalid original norm')
    return value, bits


def csv_bytes(rows: list[dict]) -> bytes:
    """Serialize exact Python float representations into a portable CSV."""
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode('utf-8')


def build(root: Path) -> dict[str, bytes]:
    """Recheck small metric/identity/event snapshots and derive both figure inputs."""
    bindings = []
    source = {name: json.loads(read(root, name, digest, bindings)) for name, digest in SOURCES.items()}
    comparison, external, execution = source.values()
    require(comparison['status'] == 'COMPLETE_OBSERVED_INTERVENTIONS_NOT_200_STEP_ACCEPTANCE_OR_REPAIR'
            and comparison['blockers'] == [] and execution['exit_code'] == 0, 'Actual reader did not complete')
    require(execution['output_sha256'] == next(iter(SOURCES.values()))
            and execution['bindings_sha256'] == comparison['external_binding_file']['sha256'],
            'Actual output/execution binding differs')
    require(comparison['steps'] == [1, 3] and comparison['ranks'] == 8
            and comparison['target_count'] == 32 and tuple(comparison['roles']) == ROLES, 'Wrong report scope')
    targets = comparison['native']['capture']['targets']
    require(len(targets) == 32, 'Need all32 whole-expert target records')
    loss_rows, norm_rows, parameter_rows = [], [], []
    base_norm, base_bits = fp32(comparison['native']['capture']['norm_hex'])
    source_closure = {}
    raw_rows = {}
    for role in ROLES:
        case = comparison['roles'][role]
        proof = comparison['input_bindings'][role]
        require(proof['collection_sha256'] == external['runs'][role]['collection_sha256'], 'Collection SHA differs')
        manifest_name = CONTROL + '/collections/' + role + '-manifest-v1.json'
        manifest = json.loads(read(root, manifest_name, proof['collection_sha256'], bindings))
        require(manifest['case'] == case and manifest['released'] is True and manifest['allow_incomplete'] is False,
                'Missing complete collected execution')
        entries = {item['path']: item for item in manifest['files']}
        require(len(entries) == len(manifest['files']), 'Duplicate collected member')
        config_name = f'runs/{case}/config.resolved.json'
        config = json.loads(read(root, config_name, entries[config_name]['sha256'], bindings))
        require(config['training']['train_iters'] == 3 and config['checkpoint']['save_ckpt'] is False
                and config['model']['config']['mtp_loss_factor'] == .3
                and config['model']['config']['moe_aux_loss_coeff'] == .0001, 'Unexpected recipe/units')
        norms, clip_records, ranks = [], [], []
        identity_records = comparison['native'][role]['identity_bindings']
        for rank in range(8):
            prefix = f'runs/{case}/'
            identity_name = prefix + f'expert-replay-rank{rank}.json'
            identity_raw = read(root, identity_name, entries[identity_name]['sha256'], bindings)
            require(hashlib.sha256(identity_raw).hexdigest() == identity_records[rank]['identity_sha256'],
                    'Identity does not match actual reader')
            identity = json.loads(identity_raw)
            require(identity['rank'] == rank and identity['mode'] == role
                    and identity['own_wrappers_removed'] is True, 'Wrong producer identity')
            snapshots = {}
            for key in ('metrics', 'metadata'):
                descriptor = identity[key]
                name = prefix + descriptor['file']
                require(entries[name]['sha256'] == descriptor['sha256'], 'Descriptor is not collected')
                raw = read(root, name, descriptor['sha256'], bindings)
                require(len(raw) == entries[name]['bytes'] == descriptor['bytes'], 'Descriptor size differs')
                snapshots[key] = [json.loads(line) for line in raw.splitlines()]
            metrics = snapshots['metrics']
            require([item['step'] for item in metrics] == [1, 2, 3]
                    and all(item['rank'] == rank for item in metrics), 'Missing three-step metrics')
            require(all(type(row[key]) in (float, int) and math.isfinite(row[key])
                        for row in metrics for key in SCALARS), 'Invalid scalar')
            for step, row in enumerate(metrics, 1):
                require(all(row[key] == comparison['scalar_values_all_rank_step'][role][str(rank)][str(step)][key]
                            for key in SCALARS), 'Metric value differs from actual reader')
            ranks.append(metrics)
            measured = [item for item in snapshots['metadata'] if item['kind'] == 'measured_norm']
            clipped = [item for item in snapshots['metadata'] if item['kind'] == 'clip_used']
            require(len(measured) == len(clipped) == 1, 'Ambiguous norm/clip observation')
            require(measured[0]['native_hex'] == identity['norm_hex'] == clipped[0]['measured_hex'],
                    'Measured native bytes disagree')
            norms.append(measured[0]['native_hex'])
            clip_records.append({key: clipped[0][key] for key in
                                 ('measured_hex', 'used_hex', 'override', 'base_predicted_postclip_norm_valid')})
        require(len(set(norms)) == 1 and all(value == clip_records[0] for value in clip_records),
                'Norm/clip bytes differ across ranks')
        measured_value, measured_bits = fp32(norms[0])
        used_hex = clip_records[0]['used_hex']
        used_value, used_bits = fp32(used_hex)
        require(norms[0] == comparison['native'][role]['norm_hex'], 'Wrong actual measured norm')
        require(used_hex == comparison['native'][role].get('used_norm_hex', norms[0]), 'Wrong actual used norm')
        norm_rows.append({'role': role, 'arm': LABELS[role], 'measured_fp32': measured_value,
                          'measured_native_hex_le': norms[0], 'used_fp32': used_value,
                          'used_native_hex_le': used_hex, 'measured_minus_F_ULP': measured_bits - base_bits,
                          'used_minus_F_ULP': used_bits - base_bits, 'clip_override': clip_records[0]['override'],
                          'base_predicted_postclip_norm_valid': clip_records[0]['base_predicted_postclip_norm_valid']})
        current = comparison['native'][role]['targets']
        require(set(current) == set(targets), 'Whole-expert key coverage differs')
        counts = {stage: sum(current[key][stage] == targets[key][stage] for key in targets) for stage in STAGES}
        declared = comparison['parameter_after_full_native_hash_equality'][role]['capture']
        require(counts['parameter_after'] == declared['equal_targets'] and declared['max_abs'] is None,
                'Parameter hash counts are not the declared whole-expert comparison')
        for key in sorted(targets):
            parameter_rows.append({
                'role': role, 'arm': LABELS[role], 'target_key': key,
                'parameter_after_native_sha256': current[key]['parameter_after'],
                'matches_F_hash': current[key]['parameter_after'] == targets[key]['parameter_after']})
        raw_rows[role] = ranks
        source_closure[role] = {'case': case, 'collection_sha256': proof['collection_sha256'],
                                'inventory_sha256': proof['inventory_sha256'],
                                'request_sha256': proof['request_sha256'], 'receipt_sha256': proof['receipt_sha256'],
                                'donor_manifest_sha256': proof['donor_manifest_sha256'],
                                'stage_recorded_native_hash_matches_F': counts,
                                'parameter_after_hash_unit': declared['unit'], 'parameter_numeric_error': None}
    for role in ROLES:
        ranks = raw_rows[role]
        for step in range(3):
            require(all(all(rank[step][field] == ranks[0][step][field] for field in SCALARS)
                        for rank in ranks), 'Expected global scalar rank consensus is absent')
            values = {field.rsplit('/', 1)[-1]: ranks[0][step][field] for field in SCALARS}
            error = max(abs(ranks[rank][step]['training/foundation_loss/lm'] -
                            raw_rows['capture'][rank][step]['training/foundation_loss/lm']) for rank in range(8))
            if step == 0:
                require(all(ranks[0][step][field] == raw_rows['capture'][0][step][field]
                            for field in SCALARS[:4]), 'First-forward loss changed before intervention')
            loss_rows.append({'role': role, 'arm': LABELS[role], 'step': step + 1, **values,
                              'main_lm_absdiff_vs_F_8rankmax': error})
    summary = {
        'schema': 'expert-grad-replay-first-update-portable-summary-v1', 'source': comparison['source'],
        'status': comparison['status'], 'roles': {key: {'case': case, 'arm': LABELS[key]}
                                                for key, case in comparison['roles'].items()},
        'steps': 3, 'ranks': 8, 'runs_per_arm': 1, 'target_count': 32,
        'source_closure': source_closure, 'control_own_prefix_checks': comparison['control_own_prefix_checks'],
        'intervention_semantics': comparison['intervention_semantics'], 'runtime': comparison['runtime'],
        'image_sha256': comparison['executions']['capture']['image'], 'norm_F_fp32': base_norm,
        'norm_ULP_absolute_difference': fp32('9fc9b341')[0] - fp32('9ec9b341')[0],
        'donor_payload_bytes': 1509949440, 'donor_payload_gib': 1.40625,
        'log_units': {'lm': 'unweighted', 'mtp': 'logged weighted by0.3', 'aux': 'logged weighted by0.0001'},
        'full_donor_reproduction_package': False, 'parameter_error_not_measured': True,
        'scope': comparison['scope'], 'gate_created': False, 'qualification_created': False,
    }
    values = {'data/losses.csv': csv_bytes(loss_rows), 'data/norms.csv': csv_bytes(norm_rows),
              'data/parameter-hashes.csv': csv_bytes(parameter_rows)}
    summary['csv_sha256'] = {name: hashlib.sha256(raw).hexdigest() for name, raw in values.items()}
    values['data/summary.json'] = (json.dumps(summary, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
    provenance = {'schema': 'expert-replay-first-update-report-source-provenance-v1',
                  'source_path_base': 'task root', 'source_snapshots': bindings,
                  'method': 'Only small source-pinned JSON/JSONL records; no donor payload is opened or copied. '
                            'LM is recomputed from40 metrics; norm4B from40 native journals; '
                            'whole-expert recorded SHA counts are recomputed from the frozen actual comparison.',
                  'report_does_not_reproduce_complete_donor': True}
    values['evidence/source-provenance.json'] = (json.dumps(provenance, ensure_ascii=False, indent=2) + '\n').encode()
    return values


def main() -> None:
    """Write exclusive new portable data; no training, qualification or remote work."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--task-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=REPORT)
    args = parser.parse_args()
    values = build(args.task_root.resolve())
    output = args.output_dir.resolve()
    require(all(not (output / name).exists() for name in values), 'Existing output data is preserved')
    for name, raw in values.items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(raw)
    print(json.dumps({'status': 'BUILT_FROM_FROZEN_ACTUAL_SMALL_EVIDENCE', 'arms': 5, 'steps': 3,
                      'donor_payload_opened': False, 'output_files': len(values)}))


if __name__ == '__main__':
    main()
