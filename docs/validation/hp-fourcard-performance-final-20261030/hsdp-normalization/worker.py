"""Four-rank FP32 HSDP diagnostic; independent candidate, not a training test."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback
import time

EXPECTED_TEST = '0db619eaaf50dfce96a493461f4056011268d31ebf85fd130e9018714821dd57'
EXPECTED_PARAM_GROUP = 'a939929814a787eac4b639c66ec03bc61761a4a3d16d4f122065f942c13955f0'

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main(result):
    source = Path(os.environ['HP_SOURCE']).resolve(strict=True)
    test_file = source / 'tests/torch/fully_shard/_test_hsdp_avg_grad_scale_parity.py'
    pg_file = source / 'hyper_parallel/core/fully_shard/param_group.py'
    result['source_hashes'] = {'test': digest(test_file), 'param_group': digest(pg_file)}
    assert result['source_hashes'] == {'test': EXPECTED_TEST, 'param_group': EXPECTED_PARAM_GROUP}
    sys.path.insert(0, str(source))
    os.environ.setdefault('HYPER_PARALLEL_PLATFORM', 'torch')
    import numpy as np
    import torch
    import torch_npu
    import torch.distributed as dist
    import hyper_parallel
    assert Path(hyper_parallel.__file__).resolve().is_relative_to(source)
    from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
    spec = importlib.util.spec_from_file_location('pinned_hsdp_test', test_file)
    t = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(t)
    rank = int(os.environ['RANK'])
    assert int(os.environ['WORLD_SIZE']) == 4
    local_rank = int(os.environ['LOCAL_RANK'])
    if local_rank != rank or Path(os.environ['OUTPUT_DIR']) != Path('/work/run'):
        raise RuntimeError('device gate requires local rank == rank and OUTPUT_DIR=/work/run')
    torch.npu.set_device(local_rank)
    torch.empty(1, dtype=torch.float32, device='npu')
    torch.npu.synchronize()
    worker_sha = digest(__file__)
    proc_stat = Path('/proc/self/stat').read_text()
    starttick = int(proc_stat[proc_stat.rfind(')') + 2:].split()[19])
    ready = {'rank': rank, 'local_rank': local_rank, 'pid': os.getpid(),
             'starttick': starttick, 'pid_namespace': os.readlink('/proc/self/ns/pid'),
             'current_device': int(torch.npu.current_device()),
             'device_name': str(torch.npu.get_device_name()), 'worker_sha256': worker_sha}
    if ready['current_device'] != local_rank:
        raise RuntimeError('current NPU device differs from local rank')
    out = Path(os.environ['OUTPUT_DIR'])
    ready_path = out / f'device-ready-rank-{rank}.json'
    temp_path = out / f'.device-ready-rank-{rank}.{os.getpid()}.tmp'
    with temp_path.open('x', encoding='utf-8') as f:
        json.dump(ready, f, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    # A hard link is an atomic no-overwrite publication in the same directory.
    os.link(temp_path, ready_path)
    temp_path.unlink()
    result['device_ready'] = ready
    release = out / 'device-release.json'
    deadline = time.monotonic() + 180
    while not release.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError('root device-map release not received within 180 seconds')
        time.sleep(0.2)
    raw = release.read_bytes()
    proof = json.loads(raw)
    mapping = proof.get('mapping')
    if (proof.get('kind') != 'root-verified-physical-device-map'
            or proof.get('worker_sha256') != worker_sha
            or proof.get('run_id') != 'hp-hsdp-normalization-129-v1'
            or proof.get('cid') != os.environ['HP_DIAG_CID']
            or not isinstance(mapping, list) or len(mapping) != 4):
        raise RuntimeError('invalid root device-map release identity')
    if (any(type(item.get('rank')) is not int or type(item.get('physical_card')) is not int for item in mapping)
            or {item['rank'] for item in mapping} != {0, 1, 2, 3}
            or any(item['physical_card'] != 4 + item['rank'] for item in mapping)):
        raise RuntimeError('invalid root physical-card mapping')
    result['device_release_sha256'] = hashlib.sha256(raw).hexdigest()
    dist.init_process_group('hccl')
    assert dist.get_rank() == rank and dist.get_world_size() == 4
    result['rank'] = rank
    result['torch_version'] = torch.__version__
    result['torch_npu_version'] = getattr(torch_npu, '__version__', 'unknown')
    shard_groups = [dist.new_group(ranks=g) for g in ([0,1], [2,3])]
    replica_groups = [dist.new_group(ranks=g) for g in ([0,2], [1,3])]
    sg, rg = shard_groups[rank//2], replica_groups[rank%2]
    assert dist.get_process_group_ranks(sg) == ([0,1] if rank < 2 else [2,3])
    assert dist.get_process_group_ranks(rg) == ([0,2] if rank%2 == 0 else [1,3])
    x = torch.arange(1, 9, dtype=torch.float32, device='npu') * (rank+1)
    y = torch.empty(4, dtype=torch.float32, device='npu')
    dist.reduce_scatter_tensor(y, x, op=dist.ReduceOp.AVG, group=sg)
    actual_rs = y.cpu()
    first = (rank%2)*4 + 1
    rs_expected = torch.arange(first, first+4, dtype=torch.float32) * (1.5 if rank < 2 else 3.5)
    dist.all_reduce(y, op=dist.ReduceOp.SUM, group=rg)
    y.div_(2)
    expected = torch.arange(first, first+4, dtype=torch.float32) * 2.5
    result['direct'] = {'shard_ranks': dist.get_process_group_ranks(sg), 'replica_ranks': dist.get_process_group_ranks(rg), 'rs_actual': actual_rs.tolist(), 'rs_expected': rs_expected.tolist(), 'rs_equal': torch.equal(actual_rs, rs_expected), 'final_actual': y.cpu().tolist(), 'final_expected': expected.tolist(), 'final_equal': torch.equal(y.cpu(), expected)}
    def tensor_hash(x):
        return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    # CPU oracle is built from the same test initializer; all actual parameters
    # are captured before wrapping and compared byte-for-byte to this reference.
    cpu = t._StackedMLP(t._NUM_LAYERS, t._HIDDEN)
    t._init_deterministic_weights(cpu, 'cpu')
    weights = {name: tensor_hash(param) for name, param in cpu.named_parameters()}
    inputs = [t._per_rank_input('cpu', r) for r in range(4)]
    input_hashes = [tensor_hash(v) for v in inputs]
    for data in inputs:
        (cpu(data)/4).backward()
    refs = {name: p.grad.detach().clone() for name, p in cpu.layers[0].named_parameters()}
    assert set(refs) == {'fc1.weight','fc1.bias','fc2.weight','fc2.bias'}
    init = t._init_deterministic_weights
    captures = []
    def checked_init(model, device):
        init(model, device)
        current = {name: tensor_hash(param) for name, param in model.named_parameters()}
        captures.append(current)
        if current != weights:
            raise RuntimeError('actual pre-wrap weights differ from CPU oracle')
    t._init_deterministic_weights = checked_init
    data = t._per_rank_input('npu', rank)
    assert tensor_hash(data) == input_hashes[rank]
    mesh = t.init_device_mesh(device_type='npu', mesh_shape=(2,2), mesh_dim_names=('replicate','shard'))
    result['weight_hashes'] = weights
    result['input_hashes'] = input_hashes
    result['actual_input_hash'] = tensor_hash(data)
    result['cases'] = {}
    for label, fusion, op in [('no_fusion_avg',False,'avg'), ('fusion_avg',True,'avg'), ('no_fusion_sum',False,'sum')]:
        model = t._build_sharded_model(mesh, comm_fusion=fusion, reduce_op=op, device_type='npu')
        actual_grads = t._run_one_backward(model, data)
        case = {}
        for name, param in model.layers[0].named_parameters():
            assert name in refs
            grad = param.grad
            placements = grad.placements
            actual_sg = dist.get_process_group_ranks(grad.device_mesh.get_group('shard'))
            actual_rg = dist.get_process_group_ranks(grad.device_mesh.get_group('replicate'))
            assert actual_sg == dist.get_process_group_ranks(sg) and actual_rg == dist.get_process_group_ranks(rg), (actual_sg, actual_rg)
            assert len(placements) == 2 and isinstance(placements[0], Replicate) and isinstance(placements[1], Shard) and placements[1].dim == 0, (name, placements)
            shape = tuple(refs[name].shape)
            assert shape[0] % 2 == 0
            ref = refs[name].chunk(2, dim=0)[rank%2].numpy()
            actual = actual_grads[name]
            assert actual.shape == ref.shape and np.isfinite(actual).all() and np.isfinite(ref).all()
            case[name] = {'placement': [str(p) for p in placements], 'shard_ranks': actual_sg, 'replica_ranks': actual_rg, 'global_shape': list(shape), 'local_shape': list(actual.shape), 'offset0': (rank%2)*(shape[0]//2), 'grad_sha256': hashlib.sha256(actual.tobytes()).hexdigest(), 'max_abs_diff_mean': float(np.max(np.abs(actual-ref))), 'max_abs_diff_four_mean': float(np.max(np.abs(actual-4*ref))), 'close_mean': bool(np.allclose(actual, ref, rtol=1e-4, atol=1e-5)), 'close_four_mean': bool(np.allclose(actual, 4*ref, rtol=1e-4, atol=1e-5))}
        assert set(case) == set(refs)
        result['cases'][label] = case
        del model
    assert len(captures) == 3 and all(c == weights for c in captures)
    result['actual_pre_wrap_weight_hashes'] = captures
    result['pass'] = result['direct']['rs_equal'] and result['direct']['final_equal'] and all(v['close_mean'] for v in result['cases']['no_fusion_avg'].values()) and all(v['close_mean'] for v in result['cases']['fusion_avg'].values()) and all(v['close_four_mean'] for v in result['cases']['no_fusion_sum'].values())
    dist.destroy_process_group()
    if not result['pass']:
        raise AssertionError('diagnostic comparison failed; see recorded values')

if __name__ == '__main__':
    rank = int(os.environ.get('RANK', '-1'))
    out = Path(os.environ['OUTPUT_DIR'])
    result = {'rank': rank, 'status': 'error', 'pass': False, 'scope': 'toy FP32 preclip gradient; no optimizer'}
    try:
        assert rank in range(4) and out.is_dir()
        main(result)
        result['status'] = 'complete'
    except BaseException:
        result['error'] = traceback.format_exc()
        raise
    finally:
        if rank in range(4) and out.is_dir():
            with (out / f'rank-{rank}.json').open('x', encoding='utf-8') as f:
                json.dump(result, f, indent=2, sort_keys=True, allow_nan=False)




