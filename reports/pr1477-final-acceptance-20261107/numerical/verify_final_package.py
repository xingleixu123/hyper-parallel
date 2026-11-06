"""Offline stdlib verifier of the scalar-only final public evidence package."""
import csv
import hashlib
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
CASES = ('single', 'fsdp_full', 'fsdp_backbone', 'ep2', 'cp2dp2', 'tp2cp2', 'tp2ep2sp')


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def table(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        rows = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(stream)]
    require([r['step'] for r in rows] == list(range(1501, 2501))
            and all(math.isfinite(v) for r in rows for v in r.values()), 'scalar full window invalid')
    return rows


def metric(rows, column):
    values = [row[column] for row in rows]
    return {'signed_mean': math.fsum(values) / len(values), 'MAE': math.fsum(abs(v) for v in values) / len(values),
            'positive': sum(v > 0 for v in values), 'negative': sum(v < 0 for v in values),
            'zero': sum(v == 0 for v in values), 'max_abs': max(abs(v) for v in values)}


def main():
    manifest = load(ROOT / 'manifest.json')
    actual = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file()}
    require(actual == set(manifest['files']) | {'manifest.json', 'SHA256SUMS'}, 'unlisted or missing public file')
    for name, seal in manifest['files'].items():
        path = ROOT / name
        require(not path.is_symlink() and path.suffix in ('.py', '.json', '.csv', '.md')
                and 'private' not in path.relative_to(ROOT).parts
                and path.stat().st_size == seal['bytes']
                and hashlib.sha256(path.read_bytes()).hexdigest() == seal['sha256'], 'public file seal differs')
        text = path.read_text(encoding='utf-8-sig')
        pattern = 'github' + r'_pat_|gh[pousr]_[A-Za-z0-9]{25,}|BEGIN [A-Z ]*PRIVATE KEY|Authorization:\s*Bearer'
        require(re.search(pattern, text) is None,
                'credential-like public text')
    sums = dict(line.split('  ', 1)[::-1] for line in (ROOT / 'SHA256SUMS').read_text(encoding='utf8').splitlines())
    expected_sums = {name: row['sha256'] for name, row in manifest['files'].items()}
    expected_sums['manifest.json'] = hashlib.sha256((ROOT / 'manifest.json').read_bytes()).hexdigest()
    require(sums == expected_sums, 'SHA256SUMS differs')
    matrix = load(ROOT / 'matrix-summary.json')
    require(tuple(matrix['case_order']) == CASES and matrix['all_seven_numerical_cases_complete'] is True
            and matrix['all_fourcard_layouts_complete'] is True and matrix['full_goal_completion_claimed'] is False
            and matrix['website_figures_generated'] is False and matrix['performance_claim'] is False,
            'scope/matrix differs')
    for name in CASES:
        folder = ROOT / 'cases' / name
        rows = table(folder / 'primary-pairs.csv')
        summary = load(folder / 'statistics-and-state-summary.json')
        require(summary['actual_updates'] == 1000 and summary['native_train_complete'] is True
                and summary['CPU_clean_exit_and_holder_removed'] is True
                and (summary['global_model_tensors'], summary['global_optimizer_slots'], summary['optimizer_native_option_bytes'])
                == (65, 91, 643) and summary['runtime_aliases_reconstructed'] is False,
                'native state scope/counts differ')
        for label, column in (('LOSS', 'loss_delta'), ('MTPLOSS_raw', 'mtp_raw_delta')):
            recomputed = metric(rows, column)
            for field, value in recomputed.items():
                require(math.isclose(value, summary['required_metrics'][label][field], rel_tol=1e-12, abs_tol=1e-12),
                        'public metric recomputation differs')
            require(abs(recomputed['signed_mean']) <= .005, 'required mean threshold failed')
        require(summary['AUX_threshold'] is None and summary['AUX_observation']['threshold'] is None,
                'invented AUX threshold')
        if not summary['original_wrapper_clean_exit']:
            require(summary['bounded_audit_sha256'] and summary['bounded_audit_status'], 'failed wrapper audit missing')
        if name != 'single':
            aux = table(folder / 'AUX-pairs-vs-new-FSDP.csv')
            require(all(math.isclose(r['delta'], r['aux_raw_actual'] - r['aux_raw_new_FSDP'], rel_tol=1e-12, abs_tol=1e-12)
                        for r in aux), 'AUX scalar delta differs')
            aux_stat = metric(aux, 'delta')
        else:
            aux_stat = metric(rows, 'aux_raw_delta')
        for field, value in aux_stat.items():
            require(math.isclose(value, summary['AUX_observation'][field], rel_tol=1e-12, abs_tol=1e-12),
                    'AUX observation recomputation differs')
        with (folder / 'statistics.csv').open(encoding='utf8', newline='') as stream:
            stat_rows = list(csv.DictReader(stream))
        require([r['metric'] for r in stat_rows] == ['LOSS', 'MTPLOSS_raw', 'AUX_raw_observation'],
                'statistics CSV metric coverage differs')
        for row in stat_rows:
            expected = summary['AUX_observation'] if row['metric'] == 'AUX_raw_observation' else summary['required_metrics'][row['metric']]
            for field in ('signed_mean', 'absolute_signed_mean', 'MAE', 'positive', 'negative', 'zero', 'max_abs', 'max_step'):
                require(math.isclose(float(row[field]), expected[field], rel_tol=1e-12, abs_tol=1e-12),
                        'statistics CSV differs from sealed summary')
    review = load(ROOT / 'review-disposition.json')
    require(len(review['threads']) == len({r['thread_id'] for r in review['threads']}) == 44
            and len(review['timeline']) == len({str(r['id']) for r in review['timeline']}) == 148,
            'historical review matrix differs')
    print(json.dumps({'status': 'all_public_scalar_seals_statistics_and_scope_verified', 'cases': 7,
                      'full_window_rows_per_case': 1000, 'threads': 44, 'timeline': 148,
                      'network_used': False, 'framework_imported': False, 'payload_read': False}))


if __name__ == '__main__':
    main()
