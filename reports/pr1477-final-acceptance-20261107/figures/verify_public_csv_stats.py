"""Verify the seven public scalar windows using only csv/math/hashlib.

No framework, checkpoint, private YAML, browser, network, or device access.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

CASES = ('fsdp_full', 'fsdp_backbone', 'ep2', 'cp2dp2', 'tp2cp2', 'tp2ep2sp')
STEPS = list(range(1501, 2501))
METRICS = {'loss': 'LOSS', 'mtp': 'MTPLOSS_raw', 'aux': 'AUX_raw_observation'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise AssertionError(message)


def raw_csv(path):
    with Path(path).open('r', encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def scalar_window(rows, path):
    require([int(row['step']) for row in rows] == STEPS, f'{path}: full step1501–2500 required')
    output = {key: [float(row[key]) for row in rows] for key in rows[0] if key != 'step'}
    require(all(math.isfinite(value) for values in output.values() for value in values), f'{path}: nonfinite')
    return output


def independent_stats(delta):
    n = len(delta)
    maximum = max(abs(v) for v in delta)
    return {'signed_mean': math.fsum(delta) / n,
            'absolute_signed_mean': abs(math.fsum(delta) / n),
            'MAE': math.fsum(abs(v) for v in delta) / n,
            'positive': sum(v > 0 for v in delta), 'negative': sum(v < 0 for v in delta),
            'zero': sum(v == 0 for v in delta), 'max_abs': maximum,
            'max_step': STEPS[next(i for i, value in enumerate(delta) if abs(value) == maximum)]}


def cumulative_signed_mean(delta):
    # Independent fsum at every actual prefix; no smoothing or sampling.
    return [math.fsum(delta[:i]) / i for i in range(1, len(delta) + 1)]


def load_and_verify(archive):
    archive = Path(archive)
    manifest_path = archive / 'manifest.json'
    manifest = json.loads(manifest_path.read_bytes())
    inputs = {}
    cases = {}
    for case in ('single', *CASES):
        folder = archive / 'cases' / case
        wanted = ['primary-pairs.csv', 'statistics.csv']
        if case != 'single':
            wanted += ['native-metrics.csv', 'AUX-pairs-vs-new-FSDP.csv']
        tables = {}
        for name in wanted:
            relative = f'cases/{case}/{name}'
            path = folder / name
            before = path.stat()
            actual = sha(path)
            expected = manifest['files'][relative]
            require(actual == expected['sha256'] and before.st_size == expected['bytes'], relative + ': SHA/size differs')
            tables[name] = raw_csv(path)
            after = path.stat()
            require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), relative + ': changed')
            inputs[relative] = {'sha256': actual, 'bytes': after.st_size, 'rows': len(tables[name])}
        primary = scalar_window(tables['primary-pairs.csv'], case + '/primary-pairs.csv')
        native = scalar_window(tables['native-metrics.csv'], case + '/native-metrics.csv') if case != 'single' else None
        auxpairs = scalar_window(tables['AUX-pairs-vs-new-FSDP.csv'], case + '/AUX-pairs.csv') if case != 'single' else None
        declared = {row['metric']: row for row in tables['statistics.csv']}
        require(set(declared) == set(METRICS.values()), case + ': statistic coverage differs')
        output = {'steps': STEPS[:], 'metrics': {}}
        for metric, stat_name in METRICS.items():
            if case == 'single':
                stem = {'loss': 'loss', 'mtp': 'mtp_raw', 'aux': 'aux_raw'}[metric]
                actual, reference, delta = (primary[stem + suffix] for suffix in ('_hp', '_mf', '_delta'))
                reference_label = 'MF52f continuous'
            elif metric == 'aux':
                actual, reference, delta = (auxpairs[key] for key in ('aux_raw_actual', 'aux_raw_new_FSDP', 'delta'))
                require(actual == native['aux_raw'], case + ': AUX actual/native differs')
                reference_label = 'new FSDP source496'
            else:
                stem = {'loss': 'loss', 'mtp': 'mtp_raw'}[metric]
                actual, reference, delta = (primary[stem + suffix] for suffix in ('_actual', '_reference', '_delta'))
                require(actual == native[stem], case + ': actual/native ' + metric + ' differs')
                reference_label = 'old continuous a852'
            require(delta == [a - b for a, b in zip(actual, reference)], case + ': signed delta differs ' + metric)
            calculated = independent_stats(delta)
            given = declared[stat_name]
            for name, value in calculated.items():
                converted = int(given[name]) if name in ('positive', 'negative', 'zero', 'max_step') else float(given[name])
                require(converted == value, f'{case}/{metric}: fsum statistic {name} differs: {value} versus {converted}')
            if metric == 'aux':
                require(given['mean_threshold'] == given['mean_pass'] == '', case + ': AUX threshold/pass must be absent')
                threshold, passed = None, None
            else:
                threshold = float(given['mean_threshold'])
                passed = given['mean_pass'] == 'True'
                require(threshold == .005 and passed == (abs(calculated['signed_mean']) <= threshold), case + ': required mean rule differs')
            cumulative = cumulative_signed_mean(delta)
            require(cumulative[-1] == calculated['signed_mean'], case + ': cumulative end differs')
            output['metrics'][metric] = {'actual': actual, 'reference': reference, 'delta': delta,
                                        'cumulative': cumulative, 'stats': calculated,
                                        'mean_threshold': threshold, 'mean_pass': passed,
                                        'reference_kind': reference_label}
        cases[case] = output
    for metric in ('loss', 'mtp'):
        baseline = cases[CASES[0]]['metrics'][metric]['reference']
        require(all(cases[case]['metrics'][metric]['reference'] == baseline for case in CASES), 'four-card old reference arrays differ')
    for case in CASES:
        require(cases[case]['metrics']['aux']['reference'] == cases['fsdp_full']['metrics']['aux']['actual'], 'AUX reference is not new FSDP')
    metadata_path = archive / 'source-version-parallelism.json'
    metadata = json.loads(metadata_path.read_bytes())
    relative = metadata_path.name
    require(sha(metadata_path) == manifest['files'][relative]['sha256'], 'version metadata differs')
    require(metadata['MF_baseline']['MF_git_content_origin'].startswith('52f614ee'), 'actual MF reference must remain52f')
    require(metadata['sources']['single']['manifest_sha256'].startswith('49bffc'), 'single source differs')
    require(metadata['sources']['four_accepted']['manifest_sha256'].startswith('496788'), 'TP1 source differs')
    require(metadata['sources']['two_TP']['manifest_sha256'].startswith('3292a5'), 'TP source differs')
    report = {'status': 'all_public_CSV_SHA_and_independent_fsum_statistics_verified',
              'scope': 'Public scalar CSV only. No checkpoint/private config/framework/device/browser/network access.',
              'source_archive_manifest_sha256': sha(manifest_path),
              'source_version_metadata_sha256': sha(metadata_path), 'CSV_files': inputs,
              'cases': {case: {m: {**data['stats'], 'mean_threshold': data['mean_threshold'], 'mean_pass': data['mean_pass'],
                                  'reference_kind': data['reference_kind']} for m, data in value['metrics'].items()}
                        for case, value in cases.items()},
              'window': [1501, 2500], 'points_per_numeric_series': 1000,
              'source_CSV_count': len(inputs), 'required_mean_rules': 14,
              'all_required_mean_rules_pass': all(cases[c]['metrics'][m]['mean_pass'] for c in cases for m in ('loss', 'mtp')),
              'AUX_threshold': None, 'AUX_pass_fail_assigned': False,
              'all_six_old_reference_arrays_identical': True, 'all_six_AUX_references_are_new_FSDP': True,
              'cumulative_formula': 'math.fsum(delta[:i])/i for each i=1..1000; final equals full signed mean',
              'raw_MTP_rule': 'Use archived raw-MTP columns exactly, including prior weighted-to-raw rounding; no new division by0.1.'}
    return cases, report, metadata


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf8', newline='\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path, default=Path(__file__).resolve().parent.parent / 'numerical')
    p.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'CSV-verification.json')
    args = p.parse_args()
    _, report, _ = load_and_verify(args.archive)
    write_json(args.output, report)
    print(json.dumps({'status': report['status'], 'CSV_files': report['source_CSV_count'], 'required_rules': 14,
                      'all_required_mean_rules_pass': report['all_required_mean_rules_pass'], 'AUX_threshold': None}))


if __name__ == '__main__':
    main()
