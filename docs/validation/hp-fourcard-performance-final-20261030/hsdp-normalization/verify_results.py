"""Offline verification of recorded scalars/hashes; not a second device test."""
import ast
import hashlib
import json
import math
from pathlib import Path

P = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
row = json.loads((P / 'results.json').read_bytes())
pins = json.loads((P / 'source-pins.json').read_bytes())
assert row['actual_source_manifest_sha256'] == pins['actual_source_manifest_sha256']
assert row['actual_source_manifest_sha256'] == 'b18fef99d3d55649c439d865718db90cfc1d0b1f6742bad1f7c87fd7d9a9a8cb'
assert row['source_has_profile_instrumentation'] is False
assert row['worker_sha256'] == sha(P / 'worker.py') == pins['worker_sha256']
assert sha(P / 'pinned_test.py') == pins['source_files_sha256']['tests/torch/fully_shard/_test_hsdp_avg_grad_scale_parity.py']
for name in ('worker.py', 'pinned_test.py'):
    ast.parse((P / name).read_text(encoding='utf8'))
assert len(row['ranks']) == 4
cases = {'no_fusion_avg': 'mean', 'fusion_avg': 'mean', 'no_fusion_sum': 'four_mean'}
maximum = {case: 0.0 for case in cases}
weights, inputs = [], []
comparisons = 0
for rank, value in enumerate(row['ranks']):
    assert value['rank'] == rank and value['status'] == 'complete' and value['pass'] is True
    assert value['torch_version'] == '2.12.0+cpu' and value['torch_npu_version'] == '2.12.0.post3'
    direct = value['direct']
    base = list(range(1 + 4 * (rank % 2), 5 + 4 * (rank % 2)))
    rs = [v * (1.5 if rank < 2 else 3.5) for v in base]
    mean = [v * 2.5 for v in base]
    assert direct['rs_actual'] == direct['rs_expected'] == rs and direct['rs_equal']
    assert direct['final_actual'] == direct['final_expected'] == mean and direct['final_equal']
    assert direct['shard_ranks'] == ([0, 1] if rank < 2 else [2, 3])
    assert direct['replica_ranks'] == ([0, 2] if rank % 2 == 0 else [1, 3])
    assert len(value['weight_hashes']) == 8
    assert value['actual_pre_wrap_weight_hashes'] == [value['weight_hashes']] * 3
    assert len(value['input_hashes']) == 4 and value['actual_input_hash'] == value['input_hashes'][rank]
    weights.append(value['weight_hashes'])
    inputs.append(value['input_hashes'])
    assert set(value['cases']) == set(cases)
    for case, target in cases.items():
        values = value['cases'][case]
        assert set(values) == {'fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias'}
        # The deliberately wrong-scale alternative must be distinguishable.
        wrong = 'close_mean' if target == 'four_mean' else 'close_four_mean'
        assert any(not v[wrong] for v in values.values())
        for name, result in values.items():
            assert result['shard_ranks'] == direct['shard_ranks']
            assert result['replica_ranks'] == direct['replica_ranks']
            assert result['global_shape'] == ([64, 64] if name.endswith('weight') else [64])
            assert result['local_shape'] == ([32, 64] if name.endswith('weight') else [32])
            assert result['offset0'] == 32 * (rank % 2)
            assert result['close_' + target]
            error = result['max_abs_diff_' + target]
            assert math.isfinite(error) and error >= 0
            maximum[case] = max(maximum[case], error)
            comparisons += 1
assert all(v == weights[0] for v in weights) and all(v == inputs[0] for v in inputs)
assert comparisons == 48 and maximum == row['max_absolute_gradient_errors']
print(json.dumps({'status': 'recorded_data_verified', 'direct_scalar_oracle_exact': True,
                  'module_comparisons': comparisons, 'max_recorded_errors': maximum,
                  'scope': 'Does not reconstruct unexported full gradient arrays.'}))
