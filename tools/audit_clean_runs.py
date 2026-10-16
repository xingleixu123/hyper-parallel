"""Compare independently recorded complete or partial continuation trajectories."""
import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np


def scalar(value):
    while isinstance(value, list):
        if len(value) != 1:
            raise ValueError('Expected a recorded scalar')
        value = value[0]
    return float(value)


def metrics(path, family):
    if not path.exists():
        return {}
    result = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        components = row['components']
        if row['step'] in result:
            raise ValueError('Duplicate metric step')
        if family == 'hp':
            total = row['loss']
            lm = components['foundation_loss/lm']
            mtp = components['foundation_loss/mtp'] / .3
            aux = components['foundation_loss/aux'] / 1e-4
        else:
            total = scalar(row['outputs'][0])
            lm = scalar(components['lm_loss'])
            mtp = scalar(components['mtp_loss']) / .3
            aux = scalar(components['aux_loss'])
        if not all(math.isfinite(value) for value in (total, lm, mtp, aux)):
            raise ValueError('Nonfinite training metric')
        if abs(total - (lm + mtp * .3 + aux * 1e-4)) > 2e-5:
            raise ValueError('Component total recomposition failed')
        result[row['step']] = {'total': total, 'mtp_raw': mtp, 'aux_raw': aux}
    return result


def compare(actual, reference):
    steps = sorted(actual.keys() & reference.keys())
    report = {'count': len(steps), 'first_step': steps[0] if steps else None,
              'last_step': steps[-1] if steps else None, 'metrics': {}}
    for key, tolerance in [('total', .0005), ('mtp_raw', .001), ('aux_raw', .001)]:
        errors = [abs(actual[step][key] - reference[step][key]) for step in steps]
        if not errors:
            continue
        report['metrics'][key] = {
            'first': errors[0], 'max': max(errors), 'max_step': steps[errors.index(max(errors))],
            'last': errors[-1], 'bad_steps': [step for step, error in zip(steps, errors) if error > .005],
            'first_tolerance': tolerance, 'passed': errors[0] <= tolerance and max(errors) <= .005,
        }
    report['complete'] = sorted(actual) == sorted(reference) == list(range(201, 401))
    report['passed'] = report['complete'] and all(item['passed'] for item in report['metrics'].values())
    return report


def position_policy(root, mf_run):
    """Verify the immutable MF runtime/config evidence before interpreting position metadata."""
    receipt = root / 'evidence' / 'mf-position-policy.json'
    if not receipt.exists():
        return {'mode': 'compare_recorded_position_ids', 'verified': False}
    policy = json.loads(receipt.read_text())
    log = root / 'runs' / (mf_run + '.log')
    if policy['run'] != mf_run or hashlib.sha256(log.read_bytes()).hexdigest() != policy['runtime_log_sha256']:
        raise ValueError('MF position-policy evidence does not belong to this run')
    flags = re.findall(r"'use_eod_reset': (True|False)", log.read_text(errors='replace'))
    if not flags or set(flags) != {'False'} or policy['runtime_use_eod_reset'] is not False:
        raise ValueError('MF runtime does not establish absolute positions')
    for entry in policy['sources'].values():
        source = root / 'evidence' / entry['snapshot']
        if hashlib.sha256(source.read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError('MF source evidence changed')
    return {'mode': 'absolute_arange_because_runtime_use_eod_reset_false', 'verified': True,
            'evidence': policy}


def inputs_match(hp, mf, policy):
    """Compare effective tokens, supervision and packed boundaries, preserving position differences."""
    expected = {f'{step:06d}.npz' for step in range(200, 400)}
    files = {family: {item.name for item in path.glob('*.npz')}
             for family, path in [('hp', hp), ('mf', mf)]}
    result = {'complete': files['hp'] == files['mf'] == expected,
              'compared': 0, 'position_metadata_difference_steps': [],
              'effective_position_mismatch_steps': [], 'mismatches': [], 'position_policy': policy}
    for name in sorted(files['hp'] & files['mf']):
        with np.load(hp / name, allow_pickle=False) as actual, np.load(mf / name, allow_pickle=False) as reference:
            for a, b in [('input_ids', 'input_ids'), ('shift_labels', 'labels'), ('loss_mask', 'loss_mask')]:
                if not np.array_equal(actual[a], reference[b]):
                    result['mismatches'].append({'file': name, 'field': a})
            # MF pads cumulative lengths with leading zero and repeated last end.
            ends = np.unique(reference['actual_seq_len'])
            ends = ends[ends > 0]
            if not np.array_equal(actual['actual_seq_len'].reshape(-1), ends):
                result['mismatches'].append({'file': name, 'field': 'actual_seq_len'})
            if not np.array_equal(actual['position_ids'], reference['position_ids']):
                result['position_metadata_difference_steps'].append(int(name[:6]) + 1)
            expected_positions = reference['position_ids']
            if policy['verified']:
                expected_positions = np.arange(reference['input_ids'].shape[-1])[None, :]
            if not np.array_equal(actual['position_ids'], expected_positions):
                result['effective_position_mismatch_steps'].append(int(name[:6]) + 1)
        result['compared'] += 1
    result['passed'] = (result['complete'] and not result['mismatches']
                        and not result['effective_position_mismatch_steps'])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('/alignment/phase13'))
    parser.add_argument('--hp', default='hp-product200')
    parser.add_argument('--mf', default='mf-reference200')
    parser.add_argument('--diagnostic', action='store_true', help='Report failed or partial runs without exit 1')
    args = parser.parse_args()
    paths = {family: args.root / 'runs' / name / 'metrics.jsonl'
             for family, name in [('hp', args.hp), ('mf', args.mf)]}
    values = {family: metrics(path, family) for family, path in paths.items()}
    report = compare(values['hp'], values['mf'])
    policy = position_policy(args.root, args.mf)
    report['effective_inputs'] = inputs_match(paths['hp'].parent / 'batches', paths['mf'].parent / 'batches', policy)
    report['passed'] = report['passed'] and report['effective_inputs']['passed']
    report['record_counts'] = {family: len(value) for family, value in values.items()}
    report['sha256'] = {family: hashlib.sha256(path.read_bytes()).hexdigest()
                        for family, path in paths.items() if path.exists()}
    reference_path = Path('/alignment/phase12/runs/mf-resume/metrics.jsonl')
    if reference_path.exists():
        report['mf_repeat_vs_historical'] = compare(values['mf'], metrics(reference_path, 'mf'))
    destination = args.root / 'evidence' / (args.hp + '-numerical-audit.json')
    destination.write_text(json.dumps(report, indent=2))
    compact = json.loads(json.dumps(report))
    for group in [compact, compact.get('mf_repeat_vs_historical', {})]:
        for item in group.get('metrics', {}).values():
            item['bad_count'] = len(item.pop('bad_steps'))
    print(json.dumps(compact))
    if not args.diagnostic and not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
