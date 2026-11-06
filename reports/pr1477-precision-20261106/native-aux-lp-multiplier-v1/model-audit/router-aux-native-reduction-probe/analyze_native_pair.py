"""Independently read both closed native AUX cases and compare raw gradients."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import torch
from safetensors.torch import load_file

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(actual, reference):
    actual, reference = actual.double(), reference.double()
    delta = actual - reference
    return {'max_abs': float(delta.abs().max()),
            'relative_l2': float(delta.norm() / reference.norm()),
            'least_squares_multiplier': float((actual * reference).sum() / (reference * reference).sum())}


def read_case(path, lp, reference, target):
    rows, tensors, sources = [], {}, {}
    for rank in range(4):
        folder = path / ('rank-' + str(rank))
        complete = json.loads((folder / 'complete.json').read_text())
        result_path = folder / 'result.json'
        result = json.loads(result_path.read_text())
        assert complete['rank'] == result['rank'] == rank
        assert complete['status'] == result['status'] == 'native_aux_probe_complete'
        assert complete['loss_parallel'] is result['loss_parallel'] is lp
        assert complete['run_id'] == result['run_id']
        assert complete['result_sha256'] == sha(result_path)
        assert result['updates'] == 0 and result['manual_gradient_collectives_or_compensation'] is False
        assert result['fixture_sha256'] == reference['assets_sha256']
        assert result['reference_sha256'] == sha(HERE / 'aux-native-reference.json')
        assert result['source_manifest_sha256'] == reference['source_manifest_sha256']
        gradient_path = folder / 'native-gradients.safetensors'
        assert result['gradient_file_sha256'] == sha(gradient_path)
        native = load_file(str(gradient_path))
        before, actual = native['native_before_step_token_normalize'], native['native_after_step_token_normalize']
        assert before.dtype == actual.dtype == torch.float32
        assert torch.isfinite(before).all() and torch.isfinite(actual).all()
        chunk = result['native_chunk']
        assert list(actual.shape) == chunk['sizes']
        index = tuple(slice(offset, offset + size) for offset, size in zip(chunk['offsets'], chunk['sizes']))
        expected = target[index]
        one, two = compare(actual, expected), compare(actual, 2 * expected)
        for key in one:
            assert math.isclose(one[key], result['vs_independent_1x'][key], rel_tol=1e-9, abs_tol=1e-12)
            assert math.isclose(two[key], result['vs_independent_2x'][key], rel_tol=1e-9, abs_tol=1e-12)
        assert result['native_source_mesh'] == result['native_storage_source_mesh'] == ['tp']
        assert result['native_source_placements'] == result['native_storage_source_placements'] == ['Replicate()']
        assert result['actual_aux_groups'] == [result['actual_CP_group']]
        assert len(result['actual_CP_group']) == len(result['actual_TP_group']) == 2
        assert set(result['actual_CP_group']) & set(result['actual_TP_group']) == {rank}
        assert len(result['micros']) == 4
        for micro, observed in enumerate(result['micros']):
            cp_index = result['actual_CP_group'].index(rank)
            count = reference['foundation_counts_CP'][micro][cp_index]
            assert observed['local_foundation_count'] == count
            assert observed['bound_foundation_multiplier'] == observed['actual_aux_context_scale'] == 2 * count
            assert math.isclose(observed['native_CP_loss_sum_multiplier'], 2., rel_tol=1e-6, abs_tol=1e-6)
            assert observed['actual_local_S'] == 2048 and observed['router_padded_S'] == 2112
            assert math.isclose(observed['raw_aux'], reference['raw_aux_fp64_by_fixture_sample'][micro % 2], abs_tol=2e-5)
        normalization_error = float((before / reference['step_foundation_denominator'] - actual).abs().max())
        assert normalization_error < 1e-12
        row = {'loss_parallel': lp, 'rank': rank,
               'native_factor': result['native_gradient_scaling_factor'],
               'classification': result['classification'],
               'ratio_to_reference': one['least_squares_multiplier'],
               'relL2_to_1x': one['relative_l2'], 'max_abs_to_1x': one['max_abs'],
               'relL2_to_2x': two['relative_l2'], 'max_abs_to_2x': two['max_abs'],
               'native_chunk': chunk,
               'actual_grad_mesh': result['actual_grad_mesh'],
               'actual_grad_placements': result['actual_grad_placements'],
               'CP_group': result['actual_CP_group'], 'TP_group': result['actual_TP_group'],
               'normalize_max_abs': normalization_error}
        rows.append(row)
        tensors[rank] = actual
        sources[str(rank)] = {'result_sha256': sha(result_path), 'complete_sha256': sha(folder / 'complete.json'),
                              'gradient_sha256': sha(gradient_path), 'run_id': result['run_id']}
    for row in rows:
        for peer in row['TP_group']:
            assert rows[peer]['native_chunk'] == row['native_chunk']
            assert torch.equal(tensors[peer], tensors[row['rank']])
    assert len({source['run_id'] for source in sources.values()}) == 1
    return rows, tensors, sources


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lp-false', required=True, type=Path)
    parser.add_argument('--lp-true', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    reference = json.loads((HERE / 'aux-native-reference.json').read_text())
    assert sha(HERE / 'aux-native-fixture.safetensors') == reference['assets_sha256']
    assets = load_file(str(HERE / 'aux-native-fixture.safetensors'))
    target = assets['independent_fp64_global_gradient']
    assert target.dtype == torch.float64 and target.shape == (64, 3584) and target.norm() > 0
    false_rows, false_tensors, false_sources = read_case(args.lp_false, False, reference, target)
    true_rows, true_tensors, true_sources = read_case(args.lp_true, True, reference, target)
    assert false_sources['0']['run_id'] != true_sources['0']['run_id']
    paired = []
    for rank in range(4):
        assert false_rows[rank]['native_chunk'] == true_rows[rank]['native_chunk']
        paired.append({'rank': rank, 'LP_true_vs_LP_false': compare(true_tensors[rank], false_tensors[rank]),
                       'LP_true_vs_twice_LP_false': compare(true_tensors[rank], 2 * false_tensors[rank]),
                       'twice_exact': torch.equal(true_tensors[rank], 2 * false_tensors[rank])})
    summary = {'status': 'CPU_independent_native_AUX_pair_readback_complete',
        'rows': [*false_rows, *true_rows], 'paired': paired,
        'source_manifest_sha256': reference['source_manifest_sha256'],
        'fixture_sha256': reference['assets_sha256'], 'reference_sha256': sha(HERE / 'aux-native-reference.json'),
        'LP_false_proof': false_sources, 'LP_true_proof': true_sources,
        'scope': 'AUX-only native autograd/FSDP scheduler diagnostic. Controller closure is external; no complete MoE/CE/optimizer/training acceptance claimed.'}
    args.output.mkdir(exist_ok=False, parents=True)
    (args.output / 'native-aux-pair.json').write_text(json.dumps(summary, indent=2) + '\n')
    with (args.output / 'native-aux-pair.csv').open('w', newline='') as stream:
        fields = ['loss_parallel', 'rank', 'native_factor', 'classification', 'ratio_to_reference',
                  'relL2_to_1x', 'max_abs_to_1x', 'relL2_to_2x', 'max_abs_to_2x', 'normalize_max_abs']
        writer = csv.DictWriter(stream, fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(summary['rows'])
    print(json.dumps({'status': summary['status'], 'rows': summary['rows'], 'paired': paired}, indent=2))


if __name__ == '__main__':
    main()
