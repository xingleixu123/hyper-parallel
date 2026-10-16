"""Verify completion, restored state, frozen source and independent provenance."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path('/alignment/phase13')
RUN = 'hp-archive200-r2'
SOURCE = ROOT / 'archive-r2-source'


def read(path):
    return json.loads(path.read_text())


def main():
    audit = read(ROOT / 'evidence' / f'{RUN}-numerical-audit.json')
    assert audit['passed'] and audit['count'] == 200 and audit['complete']
    assert (ROOT / 'runs' / f'{RUN}.exit').read_text().strip() == '0'
    run = ROOT / 'runs' / RUN
    assert read(run / 'SUCCESS.json') == {'steps': 200, 'first': 201, 'last': 400}
    restore = read(run / 'RESTORE.json')
    assert restore['all_exact'] and restore['step'] == 200
    assert len(restore['model']) == 48 and len(restore['optimizer']) == 64
    assert restore['consumed_samples'] == [200]
    scheduler = ast.literal_eval(restore['lr_scheduler'])
    assert all(value['last_epoch'] == 200 for value in scheduler.values())
    assert all(all(lr == 1e-4 for lr in value['_last_lr']) for value in scheduler.values())
    manifest = read(ROOT / 'evidence/source-archive-r2-manifest.json')
    mismatches = [name for name, digest in manifest['files'].items()
                  if hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() != digest]
    assert not mismatches, mismatches
    modules = read(run / 'IMPORTED_MODULES.json')
    assert all(Path(path).is_relative_to(SOURCE) for path in modules.values())
    assert all(Path(path).relative_to(SOURCE).as_posix() in manifest['files'] for path in modules.values())
    runtime = read(run / 'RUNTIME.json')
    assert 'Version=9.2.' in runtime['cann']
    assert runtime['environment']['CLOSE_MATMUL_K_SHIFT'] == '0'
    assert runtime['deterministic']
    old = ROOT.parent / 'phase12/runs/hp-jte213-shareddown396400/metrics.jsonl'
    current = [json.loads(line) for line in (run / 'metrics.jsonl').read_text().splitlines()]
    reference = [json.loads(line) for line in old.read_text().splitlines()]
    assert len(reference) == 200
    trajectory_fields = ('loss', 'components', 'grad_norm')
    history_differences = [a['step'] for a, b in zip(current, reference)
                           if a['step'] != b['step'] or any(a[key] != b[key] for key in trajectory_fields)]
    report = {
        'run': RUN, 'passed': True, 'integration_commit': manifest['commit'],
        'base': manifest['base'], 'frozen_files_verified': len(manifest['files']),
        'imported_hyperparallel_modules_verified': len(modules),
        'restored_model_arrays': len(restore['model']),
        'restored_optimizer_arrays': len(restore['optimizer']),
        'restored_step': restore['step'], 'restored_cursor': restore['consumed_samples'],
        'scheduler': scheduler, 'source_mismatches': mismatches,
        'numerical_audit': audit,
        'historical_trajectory_comparison': {
            'reference': str(old), 'compared': 200, 'fields': trajectory_fields,
            'different_steps': history_differences,
            'note': 'Only an additional equivalence check; acceptance uses the fresh MF run.'},
        'observer_sha256': hashlib.sha256((ROOT / 'tools/observe_archived_hp.py').read_bytes()).hexdigest(),
        'config_sha256': hashlib.sha256((ROOT / 'configs' / f'{RUN}.yaml').read_bytes()).hexdigest(),
    }
    (ROOT / 'evidence/clean-final-validation.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ('numerical_audit', 'scheduler')}, indent=2))


if __name__ == '__main__':
    main()
