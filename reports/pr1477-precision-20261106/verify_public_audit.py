"""Validate this prepared public evidence package without training or network calls."""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(name):
    return json.loads((ROOT / name).read_bytes())


def text_values(value):
    if isinstance(value, dict):
        for child in value.values():
            yield from text_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from text_values(child)
    elif isinstance(value, str):
        yield value


def main():
    manifest = read('manifest.json')
    expected = set(manifest['files'])
    actual = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file()}
    assert actual == expected | {'manifest.json', 'SHA256SUMS'}, 'unexpected or missing public artifact'
    for name, proof in manifest['files'].items():
        path = ROOT / name
        assert path.is_file() and not path.is_symlink()
        assert path.stat().st_size == proof['bytes'] and sha(path) == proof['sha256'], name
        assert 'private' not in path.relative_to(ROOT).parts and path.suffix.lower() not in ('.yaml', '.yml')
        if path.suffix == '.py':
            compile(path.read_bytes(), str(path), 'exec')
        if path.suffix in ('.json', '.md', '.py', '.patch'):
            raw = path.read_text(encoding='utf8')
            values = list(text_values(json.loads(raw))) if path.suffix == '.json' else [raw]
            for value in values:
                assert not re.search(r'(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[A-Z0-9]{16}|https?://[^\s/@]+:[^\s/@]+@)', value), name
                assert not re.search(r'(?im)^\s*```(?:yaml|yml)\s*$', value), name
                headers = re.findall(r'(?m)^\s*(model|optimizer|training|dataset|dataloader|fsdp_config|plan_overrides|lr_scheduler)\s*:', value)
                assert not headers, (name, 'unexpected configuration block')
    checksums = {}
    for line in (ROOT / 'SHA256SUMS').read_text(encoding='utf8').splitlines():
        digest, name = line.split('  ', 1)
        assert name not in checksums
        checksums[name] = digest
        assert sha(ROOT / name) == digest, name
    assert set(checksums) == expected | {'manifest.json'}

    review = read('review-disposition.json')
    assert len(review['threads']) == 44 and len(review['timeline']) == 148
    assert len({t['thread_id'] for t in review['threads']}) == 44
    inline_ids = [comment for thread in review['threads'] for comment in thread['comment_ids']]
    assert len(inline_ids) == len(set(inline_ids)) == 82
    kinds = Counter('inline' if '#discussion_r' in row['url'] else
                    'review' if '#pullrequestreview-' in row['url'] else
                    'conversation' if '#issuecomment-' in row['url'] else 'other'
                    for row in review['timeline'])
    assert kinds == {'inline': 82, 'review': 50, 'conversation': 16}
    timeline_inline_ids = {int(row['url'].split('#discussion_r')[1])
                           for row in review['timeline'] if '#discussion_r' in row['url']}
    assert timeline_inline_ids == set(inline_ids)

    with (ROOT / 'mf-baseline-2500-three-losses.csv').open(encoding='utf8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert [int(row['step']) for row in rows] == list(range(1, 2501))
    assert all(math.isfinite(float(row[key])) for row in rows for key in ('loss', 'mtp_raw', 'aux_raw'))
    assert sha(ROOT / 'mf-baseline-2500-three-losses.csv') == '24339acb29204fc866d61fa4215115d2bd0e2b6dd28a390789483b3f80d17588'
    activation = read('activation-policy/probe-result.json')
    assert activation['frozen_config_equal'] and activation['frozen_field_count'] == 112
    patch = (ROOT / 'activation-policy/activation-policy.patch').read_text(encoding='utf8')
    added = [line for line in patch.splitlines() if line.startswith('+') and not line.startswith('+++')]
    removed = [line for line in patch.splitlines() if line.startswith('-') and not line.startswith('---')]
    assert len(added) == 2 and not removed
    pr = read('activation-policy/pr.json')
    assert pr['draft'] and pr['head'] == '3475d4944d18f823d301c893330077a87252a9eb'
    status = read('npu-status.json')['historical_native_gate_a3']
    assert (status['model_tensors_verified'], status['optimizer_slots_verified'], status['fp32_alias_verified']) == (65, 91, 61)
    assert status['global_step'] == 1500 and status['consumed_samples'] == 3000 and status['completed_training_updates'] == 0
    latest = read('npu-status.json')
    single = latest['native_single_full']
    accepted = read('single-acceptance.json')
    assert accepted['status'] == 'single_full1000_acceptance_verified' and accepted['closed_producer_rc'] == 0
    assert accepted['cpu_readback_sha256'] == '9d0fdd0434377fe0d73053d6c4f94b3b6eb4ebce05b4be773a9b4f14cf1d3ecb'
    assert single['steps'] == 1000 and single['window'] == [1501, 2500] and single['source_files'] == 1542
    assert latest['full_single_card_precision_passed'] and not latest['full_four_card_precision_passed']
    with (ROOT / 'single-1501-2500-pairs.csv').open(encoding='utf8', newline='') as stream:
        single_rows = list(csv.DictReader(stream))
    assert [int(row['step']) for row in single_rows] == list(range(1501, 2501))
    for name, column in [('LOSS', 'loss_delta'), ('MTPLOSS', 'mtp_raw_delta'), ('AUXLOSS', 'aux_raw_delta')]:
        values = [float(row[column]) for row in single_rows]
        assert all(math.isfinite(value) for value in values)
        metric = single['metrics'][name]
        assert math.isclose(math.fsum(values) / 1000, metric['signed_mean'], rel_tol=0, abs_tol=1e-14)
        assert math.isclose(math.fsum(map(abs, values)) / 1000, metric['MAE'], rel_tol=0, abs_tol=1e-14)
        if name != 'AUXLOSS':
            assert abs(metric['signed_mean']) <= .005
    router = read('router-boundary-status.json')
    assert router['status'] == 'paired_diagnostic_passed' and router['paired_records'] == 12
    assert router['candidate_probe_returncode'] == -15 and not router['controller_exit_is_clean']
    assert not router['full_model_or_FSDP_acceptance'] and not router['fourcard_full_training_acceptance']
    with (ROOT / 'router-boundary-pairs.csv').open(encoding='utf8', newline='') as stream:
        router_rows = list(csv.DictReader(stream))
    assert {(int(row['rank']), row['case']) for row in router_rows} == {
        (rank, case) for rank in range(4) for case in ('tp2_ep1_sp', 'tp2_ep2_sp', 'tp2_ep2_nosp')}
    for row in router_rows:
        assert abs(float(row['candidate_aux']) - float(row['global_reference_aux'])) < 2e-5
        assert float(row['candidate_grad_relative_l2']) < 5e-4
        assert row['bias_counts_exact_and_unchanged'] == 'True'
        if row['case'] != 'tp2_ep2_sp':
            assert row['unpartitioned_aux_exactly_unchanged'] == row['unpartitioned_gradient_exactly_unchanged'] == 'True'
    assert read('router-statistics/pr.json')['head'] == '3b76915e6c87f4d1eb7d3383b3558b0439146a90'
    print(json.dumps({'status': 'passed_public_package_validation', 'publication_state': 'published_report_content',
                      'files_hashed': len(expected), 'manifest_sha256': sha(ROOT / 'manifest.json'),
                      'review_threads': 44, 'review_timeline_items': 148, 'mf_metric_rows': len(rows),
                      'yaml_files': 0, 'single_full1000_accepted': True, 'router_pairs': 12,
                      'router_candidate_controller_rc': -15, 'fourcard_full_training': 'pending'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
