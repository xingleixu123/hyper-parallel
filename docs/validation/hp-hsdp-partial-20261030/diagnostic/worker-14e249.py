"""CPU/Gloo 2-rank FSDP or 4-rank HSDP conditional-accumulation diagnostic.

Run through torch.distributed.run with --no-python and Python -B. This test
uses the existing 1a CPU scheduler test seam, not public fully_shard(cpu).
The 2-D SUM oracle is a sum over the full world and shards on the last axis.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

# Must be fixed before imports; the launcher supplies and the worker verifies it.
if (os.environ.get('HYPER_PARALLEL_PLATFORM') != 'torch'
        or os.environ.get('HYPER_PARALLEL_TEST_DEVICE_TYPE') != 'cpu'
        or os.environ.get('TORCH_DEVICE_BACKEND_AUTOLOAD') != '0'
        or os.environ.get('PYTHONDONTWRITEBYTECODE') != '1'):
    raise RuntimeError('CPU test environment is not fixed')

import torch
import torch.distributed as dist
from torch import nn
import hyper_parallel
from hyper_parallel import SkipDTensorDispatch, init_device_mesh
from hyper_parallel.core.fully_shard.api import HSDPModule, _extend_module_with_hsdp_interface
from hyper_parallel.core.fully_shard.hsdp_scheduler import HSDPSchedulerV2
from hyper_parallel.core.fully_shard.utils import MixedPrecisionPolicy, OffloadPolicy
from hyper_parallel.trainer.runtime.fsdp import configure_fsdp_gradient_sync
import tests.torch.utils as torch_test_utils

HERE = Path(__file__).resolve().parent
PINS = json.loads((HERE / 'source-and-config-pins.json').read_bytes())

def check_source():
    root = Path(os.environ['HP_TEST_SOURCE_ROOT']).resolve(strict=True)
    origin = Path(hyper_parallel.__file__).resolve(strict=True)
    if not origin.is_relative_to(root / 'hyper_parallel'):
        raise RuntimeError('import did not use the declared baseline source')
    if Path(torch_test_utils.__file__).resolve(strict=True) != (root / 'tests/torch/utils.py').resolve(strict=True):
        raise RuntimeError('test fixture import did not use the declared baseline source')
    for name, expected in PINS['source_file_sha256'].items():
        value = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if value != expected:
            raise RuntimeError('1a baseline source file differs: ' + name)
    return root, origin

class Block(nn.Module):
    def __init__(self, offset):
        super().__init__()
        self.weight = nn.Parameter((torch.arange(16, dtype=torch.float32).reshape(4, 4) + offset) / 16.0)
    def forward(self, data):
        return data @ self.weight

class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.common = Block(1)
        self.early = Block(3)
    def forward(self, data, use_early):
        result = self.common(data)
        if use_early:
            result = result + self.early(data)
        return result

def data_for(rank, micro):
    return torch.tensor([[rank + 1.0, micro + 1.0, rank + micro + 1.0, 2.0]],
                        dtype=torch.float32, requires_grad=True)

def oracle(branch_both, world_size):
    # Independent ordinary PyTorch graph. SUM over all ranks matches the
    # explicitly selected Gloo SUM reduction and does not invoke HP/FSDP.
    reference = Toy()
    weights = {name: param.detach().clone() for name, param in reference.named_parameters()}
    for rank in range(world_size):
        for micro in range(2):
            use_early = branch_both or micro == 0
            reference(data_for(rank, micro), use_early).sum().backward()
    return weights, {name: param.grad.detach().clone() for name, param in reference.named_parameters()}

def attach_cpu_scheduler(module, mesh, comm_fusion):
    # Mirrors 1a tests/torch/fully_shard/_test_fully_shard_hook_state_machine.py.
    # Public 1a fully_shard() rejects CPU and is intentionally not patched.
    _extend_module_with_hsdp_interface(module)
    policy = MixedPrecisionPolicy(param_dtype=torch.float32, reduce_dtype=torch.float32,
                                  output_dtype=torch.float32, cast_forward_inputs=False,
                                  apply_grad_on_fp32_main_grad=True)
    module.hsdp_scheduler = HSDPSchedulerV2(module, mesh, True, None, policy, OffloadPolicy(),
                                            set(), set(), torch.device('cpu'), comm_fusion)

def build(mesh, comm_fusion):
    model = Toy()
    # Separate nested FSDP units expose the conditional-participation case.
    for unit in (model.common, model.early, model):
        attach_cpu_scheduler(unit, mesh, comm_fusion)
    # Gloo SUM is explicit; oracle uses the same operation. Production has AVG.
    model.set_reduce_op_type('sum')
    return model

def local(tensor):
    if tensor is None:
        return None
    method = getattr(tensor, 'to_local', None)
    return (method() if callable(method) else tensor).detach().cpu()

def tensor_record(tensor):
    value = local(tensor)
    if value is None:
        return None
    return {'dtype': str(value.dtype), 'shape': list(value.shape), 'values': value.tolist()}

def observe(model):
    seen = []
    names = {id(param): name for name, param in model.named_parameters()}
    for name, unit in model.named_modules():
        if not isinstance(unit, HSDPModule):
            continue
        scheduler = unit.hsdp_scheduler
        group = scheduler.hsdp_state.param_group
        group_partials = []
        if group is not None:
            managed_fqns = []
            for member in group.hsdp_params:
                fqn = names.get(id(member.sharded_param))
                if fqn is None:
                    raise RuntimeError('fused group parameter lacks identity binding')
                managed_fqns.append(fqn)
            for key, partial in group.reduce_partial_outputs.items():
                if not isinstance(key, tuple) or len(key) != 2:
                    raise RuntimeError('unexpected fused partial key')
                value = local(partial).contiguous()
                group_partials.append({'managed_fqns': managed_fqns,
                                       'stored_parameter_layout_known': False,
                                       'key_reduce_dtype': str(key[1]),
                                       'shape': list(value.shape), 'dtype': str(value.dtype),
                                       'sha256': hashlib.sha256(value.numpy().tobytes()).hexdigest()})
        for hp in scheduler.hsdp_state.hsdp_params:
            param_name = names.get(id(hp.sharded_param))
            if param_name is None:
                raise RuntimeError('managed parameter lacks identity binding')
            seen.append({'module': name, 'fqn': param_name,
                         'scheduler_state': None if scheduler.scheduler_state is None else scheduler.scheduler_state.name,
                         'reduce_grads': scheduler.hsdp_state.reduce_grads,
                         'requires_all_reduce': scheduler.hsdp_state.requires_all_reduce,
                         'pending_group_buckets': group_partials,
                         'pending_unsharded': tensor_record(hp.unsharded_accumulated_grad),
                         'pending_reduce_partial': tensor_record(hp.reduce_partial_output),
                         'model_parameter': tensor_record(hp.sharded_param),
                         'model_grad': tensor_record(hp.sharded_param.grad),
                         'model_main_grad': tensor_record(getattr(hp.sharded_param, 'main_grad', None))})
    if sorted(row['fqn'] for row in seen) != ['common.weight', 'early.weight']:
        raise RuntimeError('managed FQN coverage differs')
    return sorted(seen, key=lambda row: row['fqn'])

def case(mesh, rank, mesh_info, comm_fusion, synchronize_each_micro, branch_both):
    label = ('both' if branch_both else 'micro0_only') + ('_true' if synchronize_each_micro else '_false')
    model = build(mesh, comm_fusion)
    weights, expected = oracle(branch_both, mesh_info['world_size'])
    before = observe(model)
    if any(row['pending_unsharded'] is not None or row['pending_reduce_partial'] is not None
           or row['model_grad'] is not None or row['model_main_grad'] is not None
           or row['pending_group_buckets'] for row in before):
        raise RuntimeError('new model unexpectedly has pending gradients')
    for row in before:
        want = weights[row['fqn']].chunk(mesh_info['dp_shard_size'], dim=0)[mesh_info['shard_rank']].contiguous()
        record = row['model_parameter']
        actual = torch.tensor(record['values'], dtype=torch.float32)
        if record['dtype'] != 'torch.float32' or not torch.equal(actual, want):
            raise RuntimeError('initial sharded weights differ from independent oracle: ' + row['fqn'])
    cfg = SimpleNamespace(dp_shard_size=mesh_info['dp_shard_size'], requires_grad_sync=synchronize_each_micro)
    phases = []
    with SkipDTensorDispatch():
        for micro in range(2):
            configure_fsdp_gradient_sync([model], cfg, mesh_info['dp_replicate_size'], micro, 2)
            inputs = data_for(rank, micro)
            use_early = branch_both or micro == 0
            loss = model(inputs, use_early).sum()
            loss_value = float(loss.detach().item())
            loss.backward()
            phases.append({'micro': micro, 'input': inputs.detach().tolist(), 'use_early': use_early,
                           'loss': loss_value, 'observation': observe(model)})
    by_fqn = {row['fqn']: row for row in phases[-1]['observation']}
    failures = []
    compared = {}
    for name, full in expected.items():
        want = full.chunk(mesh_info['dp_shard_size'], dim=0)[mesh_info['shard_rank']].contiguous()
        row = by_fqn[name]
        record = row['model_main_grad']
        actual = None if record is None else torch.tensor(record['values'], dtype=torch.float32)
        equal = (actual is not None and record['dtype'] == 'torch.float32'
                 and tuple(actual.shape) == tuple(want.shape) and torch.equal(actual, want))
        if not equal:
            failures.append(name + ': preclip main_grad differs from independent SUM oracle')
        if (row['pending_unsharded'] is not None or row['pending_reduce_partial'] is not None
                or row['pending_group_buckets']):
            failures.append(name + ': delayed gradient remains pending after last micro')
        compared[name] = {'expected_local_fp32': want.tolist(), 'actual': record, 'equal_exact': equal}
    # At first micro false must retain each observed unsharded FP32 buffer;
    # true must not retain a no-sync buffer. Both tiny branches participate.
    for row in phases[0]['observation']:
        pending = row['pending_unsharded']
        if synchronize_each_micro and pending is not None:
            failures.append(row['fqn'] + ': true unexpectedly retained no-sync buffer')
        if not synchronize_each_micro and (pending is None or pending['dtype'] != 'torch.float32'):
            failures.append(row['fqn'] + ': false lacks first-micro FP32 buffer')
        if synchronize_each_micro and mesh_info['dp_replicate_size'] > 1:
            partial = row['pending_group_buckets'] if comm_fusion else row['pending_reduce_partial']
            if not partial:
                failures.append(row['fqn'] + ': expected first-micro replica partial not observed')
            if row['requires_all_reduce'] is not False:
                failures.append(row['fqn'] + ': first-micro replica all-reduce was not disabled')
    return {'schema': 'cpu-gloo-mesh-fusion-two-micro-v2', 'case': label, 'rank': rank,
            **mesh_info, 'comm_fusion': comm_fusion, 'reduction': 'SUM', 'fp32_main_grad': True,
            'initial': before, 'phases': phases, 'oracle': compared,
            'status': 'passed' if not failures else 'failed', 'failures': failures}

def mesh_contract(shape_text, world_size, rank, coordinate):
    """Validate the bounded mesh and return the actual last-axis shard coordinate."""
    if shape_text not in ('2', '2,2'):
        raise ValueError('mesh shape must be 2 or 2,2')
    shape = (2,) if shape_text == '2' else (2, 2)
    expected_world = shape[0] if len(shape) == 1 else shape[0] * shape[1]
    if world_size != expected_world or not 0 <= rank < world_size:
        raise RuntimeError('world size or rank differs from selected mesh')
    expected_coordinate = (rank,) if len(shape) == 1 else (rank // shape[-1], rank % shape[-1])
    if coordinate is None or tuple(coordinate) != expected_coordinate:
        raise RuntimeError('actual mesh coordinate differs from contiguous rank map')
    return {'mesh_shape': list(shape), 'rank_coordinate': list(coordinate),
            'world_size': world_size, 'dp_shard_size': shape[-1],
            'dp_replicate_size': 1 if len(shape) == 1 else shape[0],
            'shard_rank': int(coordinate[-1])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--mesh-shape', choices=('2', '2,2'), required=True)
    parser.add_argument('--comm-fusion', choices=('true', 'false'), required=True)
    args = parser.parse_args()
    root, origin = check_source()
    if not args.output_dir.is_dir() or args.output_dir.is_symlink() or any(args.output_dir.iterdir()):
        raise RuntimeError('output-dir must be an empty, pre-created directory')
    torch_test_utils.init_dist_gloo()
    rank, world_size = dist.get_rank(), dist.get_world_size()
    shape = (2,) if args.mesh_shape == '2' else (2, 2)
    expected_world = shape[0] if len(shape) == 1 else shape[0] * shape[1]
    if world_size != expected_world:
        raise RuntimeError('world size differs from selected mesh')
    names = ('fsdp',) if len(shape) == 1 else ('replicate', 'fsdp')
    mesh = init_device_mesh(device_type='cpu', mesh_shape=shape, mesh_dim_names=names)
    mesh_info = mesh_contract(args.mesh_shape, world_size, rank, mesh.get_coordinate())
    comm_fusion = args.comm_fusion == 'true'
    dist.barrier()  # All ranks checked the empty output directory before any rank writes.
    failed = False
    for branch_both, sync in ((True, True), (True, False), (False, True), (False, False)):
        result = case(mesh, rank, mesh_info, comm_fusion, sync, branch_both)
        result.update({'torch_version': torch.__version__, 'source_root': str(root), 'source_origin': str(origin)})
        path = args.output_dir / ('rank-%d-%s.json' % (rank, result['case']))
        with path.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        print(json.dumps({'rank': rank, 'case': result['case'], 'status': result['status'],
                          'failures': result['failures']}), flush=True)
        failed = failed or result['status'] != 'passed'
        dist.barrier()
    all_failed = [None] * world_size
    dist.all_gather_object(all_failed, failed)
    dist.destroy_process_group()
    if any(all_failed):
        raise SystemExit(1)

if __name__ == '__main__':
    main()