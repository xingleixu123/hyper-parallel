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

"""Require independent runs and both weighted and derived unweighted loss gates."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re

from compare_runs import audit as audit_weighted
from compare_runs import read_rows


def _recipe(config: dict) -> dict:
    """Remove only the explicitly allowed parallel, recompute and run-window axes."""
    required = ('model', 'optimizer', 'lr_scheduler', 'loss_fn', 'training', 'dataset',
                'dataloader', 'fsdp_config', 'accelerator', 'activation_checkpoint')
    if any(key not in config for key in required):
        raise ValueError('Resolved training recipe is incomplete')
    result = json.loads(json.dumps(config))
    result.pop('accelerator')
    result.pop('activation_checkpoint')
    result['training'].pop('train_iters', None)
    for key in ('dp_shard_size', 'edp_shard_size', 'replicate_params'):
        result['fsdp_config'].pop(key, None)
    for key in ('checkpoint_dir', 'restore_from', 'save_ckpt', 'save_steps', 'save_epochs',
                'save_consolidated', 'staging_dir'):
        result.get('checkpoint', {}).pop(key, None)
    placement_fields = {'params', 'in_src', 'in_dst', 'out_src', 'out_dst', 'region_dispatch', 'tp_divide_attrs'}
    overrides = []
    for rule in result.get('plan_overrides', []):
        retained = {key: value for key, value in rule.items() if key not in placement_fields and value is not None}
        if not retained.get('module_type') and retained.get('exact_type') is False:
            retained.pop('exact_type')
        if set(retained) - {'match', 'when'}:
            overrides.append(retained)
    if 'plan_overrides' in result:
        result['plan_overrides'] = overrides
    return result


def _execution(directory: Path, source: dict) -> tuple[dict, list[str]]:
    """Read the resource guard's separately recorded execution identity."""
    path = directory.parent.parent / 'scheduling' / (directory.name + '.receipt.json')
    if not path.is_file():
        return {}, [f'Missing execution receipt: {directory.name}']
    receipt = json.loads(path.read_text())
    problems = []
    if receipt.get('case') != directory.name or receipt.get('source') != source['commit']:
        problems.append(f'Execution receipt does not match the run: {directory.name}')
    if receipt.get('patch_sha256') != source['patch_sha256']:
        problems.append(f'Execution receipt patch does not match: {directory.name}')
    if not re.fullmatch(r'[0-9a-f]{64}', receipt.get('container_id', '')):
        problems.append(f'Missing complete container identity: {directory.name}')
    if not re.fullmatch(r'sha256:[0-9a-f]{64}', receipt.get('image', '')):
        problems.append(f'Missing immutable image identity: {directory.name}')
    if (receipt.get('attempted') is not True or receipt.get('released') is not True
            or receipt.get('exit_code') != 0 or receipt.get('reason')):
        problems.append(f'Execution did not finish cleanly: {directory.name}')
    devices = receipt.get('devices', [])
    if len(devices) != 8 or len(set(devices)) != 8:
        problems.append(f'Execution was not on eight distinct devices: {directory.name}')
    identity = {key: receipt.get(key) for key in ('case', 'container_id', 'image', 'exit_code', 'reason')}
    return identity, problems


def _validate_output(output: Path, reference: Path, candidate: Path, manifests: tuple = ()) -> None:
    """Keep reports outside both immutable input evidence directories."""
    target = output.resolve()
    for directory in (reference, candidate):
        resolved = directory.resolve()
        if target == resolved or resolved in target.parents:
            raise ValueError('The output report must be outside both input run directories')
        receipt = directory.parent.parent / 'scheduling' / (directory.name + '.receipt.json')
        if target == receipt.resolve():
            raise ValueError('The output report must not overwrite an execution receipt')
    if any(target == path.resolve() for path in manifests if path is not None):
        raise ValueError('The output report must not overwrite a collection manifest')


def _verify_manifest(directory: Path, manifest_path: Path | None) -> list[str]:
    """Bind the exact audit inputs and execution receipt to the collected file hashes."""
    if manifest_path is None:
        return [f'Missing collection manifest: {directory.name}']
    manifest = json.loads(manifest_path.read_text())
    entries = {entry['path']: entry for entry in manifest['files']}
    if len(entries) != len(manifest['files']):
        return [f'Duplicate collection manifest entries: {directory.name}']
    names = ['config.resolved.json', 'source-commit.txt', 'source.patch', 'EXIT_CODE']
    for rank in range(8):
        names.extend(f'{prefix}-rank{rank}.{extension}' for prefix, extension in
                     (('initial-state', 'json'), ('start', 'json'), ('complete', 'json'), ('metrics', 'jsonl')))
    paths = [directory / name for name in names]
    root = directory.parent.parent
    paths.append(root / 'scheduling' / (directory.name + '.receipt.json'))
    problems = []
    for path in paths:
        entry = entries.get(path.relative_to(root).as_posix())
        if entry is None or not path.is_file():
            problems.append(f'Missing collected input: {directory.name}/{path.name}')
            continue
        contents = path.read_bytes()
        if entry['bytes'] != len(contents) or entry['sha256'] != hashlib.sha256(contents).hexdigest():
            problems.append(f'Collected input SHA or size differs: {directory.name}/{path.name}')
    return problems


def _trace_integrity(reference: Path, candidate: Path, start: int, end: int) -> list[str]:
    """Check rank labels, global input identities and exact data counters."""
    problems = []
    counter_fields = ('data/consumed_tokens', 'data/consumed_samples', 'data/step_tokens', 'data/step_samples')
    global_hashes = [{}, {}]
    for rank in range(8):
        expected, actual = read_rows(reference, rank), read_rows(candidate, rank)
        reference_start = reference / f'start-rank{rank}.json'
        if not reference_start.is_file() or json.loads(reference_start.read_text()).get('global_step') != 0:
            problems.append(f'Reference must have a recorded start at zero: rank {rank}')
        if sorted(expected) != list(range(1, max(expected, default=0) + 1)):
            problems.append(f'Reference trace is not contiguous from step one: rank {rank}')
        for step in range(start, end + 1):
            if step not in expected or step not in actual:
                continue
            for index, row in enumerate((expected[step], actual[step])):
                identity = row.get('input_identity', {})
                if (row.get('rank') != rank or identity.get('rank_count') != 8
                        or not re.fullmatch(r'[0-9a-f]{64}', identity.get('global_sha256', ''))):
                    problems.append(f'Invalid rank or global input identity: rank {rank}, step {step}')
                digest = identity.get('global_sha256')
                if rank == 0:
                    global_hashes[index][step] = digest
                elif digest != global_hashes[index].get(step):
                    problems.append(
                        f'Global input identity differs across ranks: run {index}, rank {rank}, step {step}')
            for field in counter_fields:
                left, right = expected[step].get(field), actual[step].get(field)
                if (not isinstance(left, (float, int)) or not isinstance(right, (float, int))
                        or not math.isfinite(left) or not math.isfinite(right) or left < 0 or left != right):
                    problems.append(f'Data counter differs or is invalid: {field}, rank {rank}, step {step}')
    return problems


def audit(reference: Path, candidate: Path, start: int, end: int, tolerance: float = 0.005,
          *, reference_manifest: Path | None = None, candidate_manifest: Path | None = None) -> dict:
    """Audit matching-source runs with the full all-rank precision requirements.

    Unweighted errors are derived from logged scalars and their recorded model
    coefficients. This does not claim to capture the pre-multiplication tensors.
    """
    if reference.resolve() == candidate.resolve():
        raise ValueError("Independent reference and candidate directories are required")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Tolerance must be a finite nonnegative number")
    if start < 1 or end < start:
        raise ValueError("A nonempty positive step window is required")
    coefficients = []
    sources = []
    recipes = []
    for directory in (reference, candidate):
        config = json.loads((directory / 'config.resolved.json').read_text())
        recipes.append(_recipe(config))
        model = config['model']['config']
        factors = {key: float(model[key]) for key in ('mtp_loss_factor', 'moe_aux_loss_coeff')}
        if any(not math.isfinite(value) or value <= 0 for value in factors.values()):
            raise ValueError("Positive finite loss coefficients are required to recover unweighted errors")
        coefficients.append(factors)
        commit = (directory / 'source-commit.txt').read_text().strip()
        if not re.fullmatch(r'[0-9a-f]{40}', commit):
            raise ValueError('A complete source Git SHA is required')
        sources.append({
            'commit': commit,
            'patch_sha256': hashlib.sha256((directory / 'source.patch').read_bytes()).hexdigest(),
        })
    if coefficients[0] != coefficients[1]:
        raise ValueError("Reference and candidate loss coefficients differ")
    result = audit_weighted(reference, candidate, start, end, tolerance)
    result['findings'].extend(_trace_integrity(reference, candidate, start, end))
    for directory, manifest in ((reference, reference_manifest), (candidate, candidate_manifest)):
        result['findings'].extend(_verify_manifest(directory, manifest))
    result['collection_manifest_sha256'] = [hashlib.sha256(path.read_bytes()).hexdigest() if path else None
                                          for path in (reference_manifest, candidate_manifest)]
    result['weighted_status'] = result['status']
    result['sources'] = {'reference': sources[0], 'candidate': sources[1]}
    result['recipe_sha256'] = [hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
                               for value in recipes]
    if end - start + 1 != 200:
        result['findings'].append('Final acceptance requires exactly 200 consecutive steps')
    if sources[0] != sources[1]:
        result['findings'].append('Final acceptance requires matching source commit and patch')
    if recipes[0] != recipes[1]:
        result['findings'].append('Numerical training recipes differ beyond the allowed experiment axes')
    executions = []
    for directory, source in zip((reference, candidate), sources):
        identity, problems = _execution(directory, source)
        executions.append(identity)
        result['findings'].extend(problems)
    if executions[0].get('container_id') == executions[1].get('container_id'):
        result['findings'].append('Distinct container execution identities are required')
    if executions[0].get('image') != executions[1].get('image'):
        result['findings'].append('Reference and candidate runtime images differ')
    result['execution_identities'] = executions
    errors = result['errors']
    raw = {
        'lm': errors['training/foundation_loss/lm']['max_abs'],
        'mtp': errors['training/foundation_loss/mtp']['max_abs'] / coefficients[0]['mtp_loss_factor'],
        'aux': errors['training/foundation_loss/aux']['max_abs'] / coefficients[0]['moe_aux_loss_coeff'],
    }
    result['loss_coefficients'] = coefficients[0]
    result['loss_units'] = {'lm': 'unweighted',
                            'mtp': f"weighted by {coefficients[0]['mtp_loss_factor']}",
                            'aux': f"weighted by {coefficients[0]['moe_aux_loss_coeff']}"}
    result['unweighted_error_diagnostics'] = raw
    result['unweighted_derivation'] = (
        'Absolute difference of logged weighted scalars divided by the recorded coefficient')
    result['unweighted_tolerance_absolute'] = tolerance
    if result['findings']:
        result['unweighted_status'] = 'UNVERIFIED'
        result['measurement_context']['eligible_for_matching_build_mode_performance_comparison'] = False
        result['measurement_context']['excluded_reasons'].append('The complete fixed-recipe execution audit failed')
    else:
        result['unweighted_status'] = ('PASS' if all(math.isfinite(v) and v <= tolerance for v in raw.values())
                                       else 'FAIL')
    passed = result['weighted_status'] == 'PASS' and result['unweighted_status'] == 'PASS' and not result['findings']
    result['status'] = 'PASS' if passed else 'FAIL'
    if not passed:
        result['measurement_context']['eligible_for_matching_build_mode_performance_comparison'] = False
    return result


def main() -> None:
    """Write a precision report without changing any captured training evidence."""
    parser = argparse.ArgumentParser()
    parser.add_argument('reference', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('start', type=int)
    parser.add_argument('end', type=int)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference-manifest', type=Path, required=True)
    parser.add_argument('--candidate-manifest', type=Path, required=True)
    args = parser.parse_args()
    _validate_output(args.output, args.reference, args.candidate, (args.reference_manifest, args.candidate_manifest))
    result = audit(args.reference, args.candidate, args.start, args.end,
                   reference_manifest=args.reference_manifest, candidate_manifest=args.candidate_manifest)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: value for key, value in result.items() if key != 'step_errors'}, indent=2))
    raise SystemExit(0 if result['status'] == 'PASS' else 1)


if __name__ == '__main__':
    main()
