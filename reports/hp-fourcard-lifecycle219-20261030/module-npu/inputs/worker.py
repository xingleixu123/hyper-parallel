"""Two-rank HCCL nested fully_shard no-sync diagnostic; requires independent root gates.

This is a candidate only. No launcher, deployment, or device action occurs on import.
"""
import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

HEAD = '388ec460a163ae06e18b9189c490cf37da460f8a'
MANIFEST_SHA = 'dd4c52de67f7617e416f67b5ea81de3d405580c1756a8d294a45aae9c5d35256'
SCHEDULER_SHA = '219ce9346d528948dbe0755b50a261db025c87d1a6f207b893c22b8e9de0524d'
SOURCE = Path('/work/sources/hyper-parallel')
MANIFEST = Path('/work/input/source-manifest.json')
WORKER = Path('/work/input/worker.py')
SCHEDULER_REL = 'hyper_parallel/core/fully_shard/hsdp_scheduler.py'
KEYS = ('hyper_parallel/__init__.py', 'hyper_parallel/core/fully_shard/api.py',
        SCHEDULER_REL, 'hyper_parallel/core/fully_shard/hsdp_state.py',
        'hyper_parallel/core/fully_shard/hsdp_param.py', 'hyper_parallel/core/fully_shard/param_group.py',
        'hyper_parallel/core/fully_shard/hsdp_utils.py', 'hyper_parallel/core/fully_shard/utils.py',
        'hyper_parallel/core/fully_shard/hook_function.py', 'hyper_parallel/core/dtensor/device_mesh.py',
        'hyper_parallel/core/optimizer/optimizer.py',
        'hyper_parallel/components/optim/mixed_precision_optimizer.py')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def publish(path, obj):
    data = (json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    temp = path.with_name('.' + path.name + '.' + str(os.getpid()) + '.tmp')
    with temp.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temp, path)
    finally:
        temp.unlink()


def source_check(arm):
    require(SOURCE.is_dir() and not SOURCE.is_symlink() and not MANIFEST.is_symlink(), 'unsafe source or manifest')
    raw = MANIFEST.read_bytes()
    require(sha(raw) == MANIFEST_SHA, 'source manifest SHA differs')
    doc = json.loads(raw)
    require(doc.get('schema') == 'pr1369-git-source-export-v1' and doc.get('head') == HEAD
            and len(doc['files']) == 2679, 'source manifest identity differs')
    expected = doc['files']
    require(arm == 'green' and expected[SCHEDULER_REL]['sha256'] == SCHEDULER_SHA, 'PR scheduler differs')
    actual = {}
    for rel in KEYS:
        path = SOURCE / rel
        require(path.is_file() and not path.is_symlink(), 'unsafe source file: ' + rel)
        data = path.read_bytes()
        require(sha(data) == expected[rel]['sha256'], 'source pin differs: ' + rel)
        require(len(data) == expected[rel]['size'], 'source size differs: ' + rel)
        actual[rel] = sha(data)
    return actual


def process_identity(rank, local, run_id, cid, arm, cards, worker_sha, scheduler_sha):
    line = Path('/proc/self/stat').read_text()
    starttick = int(line[line.rfind(')') + 2:].split()[19])
    return {'rank': rank, 'local_rank': local, 'namespace_pid': os.getpid(),
            'starttick': starttick, 'pid_namespace': os.readlink('/proc/self/ns/pid'),
            'run_id': run_id, 'cid': cid, 'arm': arm, 'physical_cards': cards,
            'worker_sha256': worker_sha, 'source_manifest_sha256': MANIFEST_SHA,
            'scheduler_sha256': scheduler_sha}


def validate_preimport(proof, me):
    require(proof.get('kind') == 'root-verified-preimport-v1'
            and proof.get('run_id') == me['run_id'] and proof.get('cid') == me['cid']
            and proof.get('arm') == me['arm'] and proof.get('physical_cards') == me['physical_cards'],
            'preimport release identity differs')
    workers = proof.get('workers')
    require(isinstance(workers, list) and len(workers) == 2
            and all(isinstance(entry, dict) and type(entry.get('rank')) is int for entry in workers)
            and me in workers and {entry['rank'] for entry in workers} == {0, 1}, 'preimport workers differ')


def validate_mapping(proof, me):
    require(proof.get('kind') == 'root-verified-physical-device-map'
            and proof.get('run_id') == me['run_id'] and proof.get('cid') == me['cid']
            and proof.get('arm') == me['arm'] and proof.get('worker_sha256') == me['worker_sha256']
            and proof.get('source_manifest_sha256') == MANIFEST_SHA
            and proof.get('scheduler_sha256') == me['scheduler_sha256']
            and proof.get('physical_cards') == me['physical_cards'], 'mapping release identity differs')
    rows = proof.get('rank_proof')
    require(isinstance(rows, list) and len(rows) == 2
            and all(isinstance(row, dict) and type(row.get('rank')) is int for row in rows)
            and {row['rank'] for row in rows} == {0, 1}, 'mapping ranks differ')
    for row in rows:
        rank = row['rank']
        require(type(row.get('host_pid')) is int and row['host_pid'] > 0
                and type(row.get('physical_card')) is int
                and row['physical_card'] == me['physical_cards'][rank], 'physical mapping differs')
    own = next(row for row in rows if row['rank'] == me['rank'])
    for key in ('namespace_pid', 'starttick', 'pid_namespace'):
        require(own.get(key) == me[key], 'mapping worker identity differs: ' + key)


def wait_release(path, validator, me):
    deadline = time.monotonic() + 600
    while not path.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError('root release missing: ' + str(path))
        time.sleep(0.2)
    raw = path.read_bytes()
    validator(json.loads(raw), me)
    return sha(raw)


def type_name(value):
    return type(value).__module__ + '.' + type(value).__qualname__


def model_types(model):
    units = {}
    for name, module in (('root', model), ('common', model.common), ('early', model.early)):
        state = module.hsdp_scheduler.hsdp_state
        units[name] = {'module_type': type_name(module), 'scheduler_type': type_name(module.hsdp_scheduler),
                       'state_type': type_name(state), 'device': str(module.hsdp_scheduler.device),
                       'comm_fusion': state.comm_fusion_policy.enable_comm_fusion,
                       'zero_copy': state.comm_fusion_policy.comm_fusion_zero_copy}
    return units



def tree_boundary(model):
    """Observe the real shared tree after backward; never modify scheduler state."""
    schedulers = [model.hsdp_scheduler, model.common.hsdp_scheduler, model.early.hsdp_scheduler]
    root = model.hsdp_scheduler
    ctx = root.scheduler_ctx
    require(len({id(scheduler) for scheduler in schedulers}) == 3, 'unexpected scheduler ownership')
    require({id(scheduler) for scheduler in ctx.all_hsdp_schedulers} ==
            {id(scheduler) for scheduler in schedulers}, 'shared scheduler tree differs')
    require(all(scheduler.scheduler_ctx is ctx for scheduler in schedulers), 'scheduler context differs')
    return {'root_bp_state': ctx.root_bp_state,
            'scheduler_states': {name: None if scheduler.scheduler_state is None else scheduler.scheduler_state.name
                                 for name, scheduler in (('root', root), ('common', model.common.hsdp_scheduler),
                                                         ('early', model.early.hsdp_scheduler))}}


def boundary_failures(boundary):
    errors = []
    if boundary['root_bp_state'] is not False:
        errors.append('root backward context remains active')
    for name, state in boundary['scheduler_states'].items():
        if state is not None:
            errors.append(name + ': scheduler state leaked after backward: ' + str(state))
    return errors


def run_cases(torch, dist, fully_shard, init_device_mesh, skip_dispatch, policy_type, hsdp_module_type, rank, local, run, me, src):
    nn = torch.nn
    device = torch.device('npu', local)
    mesh = init_device_mesh(device_type='npu', mesh_shape=(2,), mesh_dim_names=('fsdp',))
    require(tuple(mesh.get_coordinate()) == (rank,), 'mesh coordinate differs')

    class Block(nn.Module):
        def __init__(self, offset):
            super().__init__()
            self.weight = nn.Parameter((torch.arange(16, dtype=torch.float32).reshape(4, 4) + offset) / 16.0)
        def forward(self, data):
            return data @ self.weight

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.common, self.early = Block(1), Block(3)
        def forward(self, data, use_early):
            value = self.common(data)
            return value + self.early(data) if use_early else value

    def inputs(source_rank, micro, target):
        return torch.tensor([[source_rank + 1.0, micro + 1.0, source_rank + micro + 1.0, 2.0]],
                            dtype=torch.float32, device=target, requires_grad=True)

    def local_tensor(tensor):
        if tensor is None:
            return None
        method = getattr(tensor, 'to_local', None)
        return (method() if callable(method) else tensor).detach().cpu()

    def record(tensor):
        if tensor is None:
            return None
        value = local_tensor(tensor)
        return {'type': type_name(tensor), 'dtype': str(value.dtype), 'shape': list(value.shape),
                'values': value.tolist()}

    def observe(model):
        names = {id(param): name for name, param in model.named_parameters()}
        result = []
        for name, unit in model.named_modules():
            scheduler = getattr(unit, 'hsdp_scheduler', None)
            if scheduler is None:
                continue
            state = scheduler.hsdp_state
            group = state.param_group
            pending_groups = []
            if group is not None:
                for key, partial in group.reduce_partial_outputs.items():
                    require(isinstance(key, tuple) and len(key) == 2, 'unexpected partial bucket key')
                    data = local_tensor(partial).contiguous()
                    pending_groups.append({'shape': list(data.shape), 'dtype': str(data.dtype),
                                           'sha256': sha(data.numpy().tobytes()),
                                           'stored_parameter_layout_known': False})
            for hp in state.hsdp_params:
                fqn = names.get(id(hp.sharded_param))
                require(fqn is not None, 'managed FQN missing')
                result.append({'module': name, 'fqn': fqn,
                               'state': None if scheduler.scheduler_state is None else scheduler.scheduler_state.name,
                               'reduce_grads': state.reduce_grads, 'requires_all_reduce': state.requires_all_reduce,
                               'shard_world_size': hp.shard_world_size, 'shard_rank': hp.shard_rank,
                               'shard_group_size': dist.get_world_size(group=hp.mesh_info.shard_process_group),
                               'parameter': record(hp.sharded_param), 'main_grad': record(getattr(hp.sharded_param, 'main_grad', None)),
                               'grad': record(hp.sharded_param.grad), 'pending_unsharded': record(hp.unsharded_accumulated_grad),
                               'pending_reduce_partial': record(hp.reduce_partial_output), 'pending_group_buckets': pending_groups})
        require(sorted(x['fqn'] for x in result) == ['common.weight', 'early.weight'], 'managed FQN coverage differs')
        return sorted(result, key=lambda x: x['fqn'])

    failed = False
    for fusion in (False, True):
        for branch_both, sync_each in ((True, True), (True, False), (False, True), (False, False)):
            label = ('both' if branch_both else 'micro0_only') + ('_true' if sync_each else '_false')
            ref = Toy()
            weights = {name: param.detach().clone() for name, param in ref.named_parameters()}
            losses = {}
            for source_rank in range(2):
                for micro in range(2):
                    loss = ref(inputs(source_rank, micro, 'cpu'), branch_both or micro == 0).sum()
                    losses[(source_rank, micro)] = float(loss.detach().item())
                    loss.backward()
            oracle = {name: param.grad.detach().clone().chunk(2, dim=0)[rank].contiguous()
                      for name, param in ref.named_parameters()}
            model = Toy().to(device)
            policy = policy_type(param_dtype=torch.float32, reduce_dtype=torch.float32, output_dtype=torch.float32,
                                 cast_forward_inputs=False, apply_grad_on_fp32_main_grad=True)
            for unit in (model.common, model.early, model):
                wrapped = fully_shard(unit, mesh=mesh, reshard_after_forward=True, mp_policy=policy,
                                      comm_fusion=fusion, comm_fusion_zero_copy=False)
                require(wrapped is unit and isinstance(unit, hsdp_module_type), 'public fully_shard wrapper differs')
                state = unit.hsdp_scheduler.hsdp_state
                require(state.comm_fusion_policy.enable_comm_fusion is fusion
                        and state.comm_fusion_policy.comm_fusion_zero_copy is False, 'fusion policy differs')
            model.set_reduce_op_type('sum')
            initial = observe(model)
            failures = []
            for row in initial:
                want = weights[row['fqn']].chunk(2, dim=0)[rank].contiguous()
                p = row['parameter']
                got = torch.tensor(p['values'], dtype=torch.float32)
                if p['dtype'] != 'torch.float32' or not torch.equal(got, want):
                    failures.append(row['fqn'] + ': initial weight shard differs')
                if row['shard_world_size'] != 2 or row['shard_rank'] != rank or row['shard_group_size'] != 2:
                    failures.append(row['fqn'] + ': shard group differs')
                if row['main_grad'] is not None or row['pending_unsharded'] is not None:
                    failures.append(row['fqn'] + ': unexpected initial gradient')
            phases = []
            with skip_dispatch():
                for micro in range(2):
                    model.set_requires_gradient_sync(sync_each or micro == 1)
                    x = inputs(rank, micro, device)
                    loss = model(x, branch_both or micro == 0).sum()
                    value = float(loss.detach().item())
                    if value != losses[(rank, micro)]:
                        failures.append('micro%d loss differs from CPU reference' % micro)
                    loss.backward()
                    boundary = tree_boundary(model)
                    failures.extend(boundary_failures(boundary))
                    phases.append({'micro': micro, 'input': local_tensor(x).tolist(), 'loss': value,
                                   'use_early': branch_both or micro == 0, 'observation': observe(model),
                                   'tree_boundary': boundary})
            actual = {row['fqn']: row for row in phases[-1]['observation']}
            comparisons = {}
            for fqn, want in oracle.items():
                row = actual[fqn]
                p = row['main_grad']
                value = None if p is None else torch.tensor(p['values'], dtype=torch.float32)
                equal = p is not None and p['dtype'] == 'torch.float32' and torch.equal(value, want)
                if not equal:
                    failures.append(fqn + ': main_grad differs from CPU SUM oracle')
                if row['pending_unsharded'] is not None or row['pending_reduce_partial'] is not None or row['pending_group_buckets']:
                    failures.append(fqn + ': gradient remains pending')
                comparisons[fqn] = {'expected_local_fp32': want.tolist(), 'actual': p, 'equal_exact': equal}
            for row in phases[0]['observation']:
                pending = row['pending_unsharded']
                if sync_each and pending is not None:
                    failures.append(row['fqn'] + ': sync micro0 unexpectedly pending')
                if not sync_each and (pending is None or pending['dtype'] != 'torch.float32'):
                    failures.append(row['fqn'] + ': no-sync micro0 lacks FP32 accumulation')
            result = {'schema': 'npu-hccl-nosync-two-rank-v1', 'run_id': me['run_id'], 'arm': me['arm'],
                      'rank': rank, 'local_rank': local, 'fusion': fusion, 'fusion_zero_copy': False,
                      'case': label, 'mesh_shape': [2], 'reduction': 'SUM', 'device': str(device),
                      'source_pins': src, 'model_types': model_types(model), 'initial': initial,
                      'phases': phases, 'oracle': comparisons, 'status': 'passed' if not failures else 'failed',
                      'failures': failures}
            publish(run / ('rank-%d-fusion-%s-%s.json' % (rank, str(fusion).lower(), label)), result)
            print(json.dumps({'rank': rank, 'fusion': fusion, 'case': label, 'status': result['status'],
                              'failures': failures}), flush=True)
            failed = failed or bool(failures)
            dist.barrier()
    return failed


def run_child_only_cases(torch, dist, fully_shard, init_device_mesh, skip_dispatch,
                         policy_type, hsdp_module_type, chained_type, mixed_optimizer_type,
                         rank, local, run, me, src):
    """NPU analogue of upstream child-only lifecycle; no replacement hooks or collectives."""
    nn = torch.nn
    device = torch.device('npu', local)
    mesh = init_device_mesh(device_type='npu', mesh_shape=(2,), mesh_dim_names=('fsdp',))
    require(tuple(mesh.get_coordinate()) == (rank,), 'mesh coordinate differs')

    class Block(nn.Module):
        def __init__(self, offset):
            super().__init__()
            self.weight = nn.Parameter((torch.arange(16, dtype=torch.float32).reshape(4, 4) + offset) / 16.0)
        def forward(self, data):
            return data @ self.weight

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.common, self.early = Block(1), Block(3)
        def forward(self, data, use_early):
            value = self.common(data)
            return value + self.early(data) if use_early else value

    def inputs(source_rank, micro, target, input_grad):
        return torch.tensor([[source_rank + 1.0, micro + 1.0, source_rank + micro + 1.0, 2.0]],
                            dtype=torch.float32, device=target, requires_grad=input_grad)

    def local_tensor(tensor):
        if tensor is None:
            return None
        method = getattr(tensor, 'to_local', None)
        return (method() if callable(method) else tensor).detach().cpu()

    def optional_record(tensor):
        if tensor is None:
            return None
        value = local_tensor(tensor)
        return {'dtype': str(value.dtype), 'shape': list(value.shape), 'values': value.tolist()}

    def observe(model):
        names = {id(param): name for name, param in model.named_parameters()}
        rows = []
        for name, unit in model.named_modules():
            scheduler = getattr(unit, 'hsdp_scheduler', None)
            if scheduler is None:
                continue
            state = scheduler.hsdp_state
            group = state.param_group
            buckets = []
            if group is not None:
                for key, partial in group.reduce_partial_outputs.items():
                    require(isinstance(key, tuple) and len(key) == 2, 'unexpected partial bucket key')
                    data = local_tensor(partial).contiguous()
                    buckets.append({'shape': list(data.shape), 'dtype': str(data.dtype),
                                    'sha256': sha(data.numpy().tobytes()), 'stored_parameter_layout_known': False})
            for hp in state.hsdp_params:
                fqn = names.get(id(hp.sharded_param))
                require(fqn is not None, 'managed FQN missing')
                main = getattr(hp.sharded_param, 'main_grad', None)
                rows.append({'module': name, 'fqn': fqn,
                             'main_grad': None if main is None else {'dtype': str(local_tensor(main).dtype),
                                                                    'values': local_tensor(main).tolist()},
                             'pending_unsharded': optional_record(hp.unsharded_accumulated_grad),
                             'pending_reduce_partial': optional_record(hp.reduce_partial_output),
                             'pending_group_buckets': buckets})
        require(sorted(x['fqn'] for x in rows) == ['common.weight', 'early.weight'], 'managed FQN coverage differs')
        return {x['fqn']: x for x in rows}

    failed = False
    for fusion in (False, True):
        for input_grad in (False, True):
            ref = Toy()
            model = Toy().to(device)
            policy = policy_type(param_dtype=torch.float32, reduce_dtype=torch.float32, output_dtype=torch.float32,
                                 cast_forward_inputs=False, apply_grad_on_fp32_main_grad=True)
            for unit in (model.common, model.early, model):
                wrapped = fully_shard(unit, mesh=mesh, reshard_after_forward=True, mp_policy=policy,
                                      comm_fusion=fusion, comm_fusion_zero_copy=False)
                require(wrapped is unit and isinstance(unit, hsdp_module_type), 'public fully_shard wrapper differs')
                state = unit.hsdp_scheduler.hsdp_state
                require(state.comm_fusion_policy.enable_comm_fusion is fusion
                        and state.comm_fusion_policy.comm_fusion_zero_copy is False, 'fusion policy differs')
            model.set_reduce_op_type('sum')
            failures = []
            initial_weights = {}
            reference_params = dict(ref.named_parameters())
            for name, param in model.named_parameters():
                got = local_tensor(param)
                want_initial = reference_params[name].detach().chunk(2, dim=0)[rank].contiguous()
                equal_initial = got.dtype == torch.float32 and torch.equal(got, want_initial)
                if not equal_initial:
                    failures.append(name + ': initial weight shard differs')
                initial_weights[name] = {'actual': got.tolist(), 'expected_local_fp32': want_initial.tolist(),
                                         'equal_exact': equal_initial}
            require(sorted(initial_weights) == ['common.weight', 'early.weight'], 'initial weight coverage differs')
            phases = []
            with skip_dispatch():
                # Upstream CPU lifecycle fixture initializes tree ownership via a no-grad root forward,
                # then backpropagates only child outputs. No root output belongs to that graph.
                with torch.no_grad():
                    model(inputs(rank, 0, device, False), False)
                for step in range(2):
                    losses = {}
                    for source_rank in range(2):
                        loss_ref = ref.early(inputs(source_rank, step, 'cpu', input_grad)).sum()
                        losses[source_rank] = float(loss_ref.detach().item())
                        loss_ref.backward()
                    want = ref.early.weight.grad.detach().clone().chunk(2, dim=0)[rank].contiguous()
                    model.set_requires_gradient_sync(True)
                    data = inputs(rank, step, device, input_grad)
                    loss = model.early(data).sum()
                    if float(loss.detach().item()) != losses[rank]:
                        failures.append('child-only step%d loss differs from CPU reference' % step)
                    loss.backward()
                    boundary = tree_boundary(model)
                    failures.extend(boundary_failures(boundary))
                    observation = observe(model)
                    early = observation['early.weight']['main_grad']
                    actual = None if early is None else torch.tensor(early['values'], dtype=torch.float32)
                    equal = early is not None and early['dtype'] == 'torch.float32' and torch.equal(actual, want)
                    if not equal:
                        failures.append('child-only step%d accumulated main_grad differs from CPU SUM' % step)
                    if observation['common.weight']['main_grad'] is not None:
                        failures.append('child-only step%d unused common unexpectedly has main_grad' % step)
                    if any(row['pending_unsharded'] or row['pending_reduce_partial'] or row['pending_group_buckets']
                           for row in observation.values()):
                        failures.append('child-only step%d gradient remains pending' % step)
                    phases.append({'step': step, 'input': local_tensor(data).tolist(),
                                   'loss': float(loss.detach().item()), 'tree_boundary': boundary,
                                   'observation': observation, 'expected_local_fp32': want.tolist(),
                                   'equal_exact': equal})
            result = {'schema': 'npu-hccl-child-only-lifecycle-v1', 'run_id': me['run_id'], 'arm': me['arm'],
                      'rank': rank, 'local_rank': local, 'fusion': fusion, 'fusion_zero_copy': False,
                      'input_requires_grad': input_grad, 'mesh_shape': [2], 'reduction': 'SUM',
                      'device': str(device), 'source_pins': src, 'initial_weights': initial_weights,
                      'phases': phases, 'callback_count_observed': False,
                      'status': 'passed' if not failures else 'failed', 'failures': failures}
            publish(run / ('rank-%d-child-fusion-%s-input-grad-%s.json' %
                           (rank, str(fusion).lower(), str(input_grad).lower())), result)
            print(json.dumps({'rank': rank, 'fusion': fusion, 'input_requires_grad': input_grad,
                              'case': 'child_only', 'status': result['status'], 'failures': failures}), flush=True)
            failed = failed or bool(failures)
            dist.barrier()

    # Two accumulation windows on the same nested modules, aligned with the new
    # lifecycle CPU fixture. A real mixed-precision wrapper zero_grad is called
    # only after the first window's boundary, pending and exact SUM are recorded.
    for fusion in (False, True):
        ref = Toy()
        model = Toy().to(device)
        policy = policy_type(param_dtype=torch.float32, reduce_dtype=torch.float32, output_dtype=torch.float32,
                             cast_forward_inputs=False, apply_grad_on_fp32_main_grad=True)
        for unit in (model.common, model.early, model):
            wrapped = fully_shard(unit, mesh=mesh, reshard_after_forward=True, mp_policy=policy,
                                  comm_fusion=fusion, comm_fusion_zero_copy=False)
            require(wrapped is unit and isinstance(unit, hsdp_module_type), 'public fully_shard wrapper differs')
            state = unit.hsdp_scheduler.hsdp_state
            require(state.comm_fusion_policy.enable_comm_fusion is fusion
                    and state.comm_fusion_policy.comm_fusion_zero_copy is False, 'fusion policy differs')
        model.set_reduce_op_type('sum')
        failures = []
        initial_weights = {}
        reference_params = dict(ref.named_parameters())
        for name, param in model.named_parameters():
            got = local_tensor(param)
            want_initial = reference_params[name].detach().chunk(2, dim=0)[rank].contiguous()
            equal_initial = got.dtype == torch.float32 and torch.equal(got, want_initial)
            if not equal_initial:
                failures.append(name + ': initial weight shard differs')
            initial_weights[name] = {'actual': got.tolist(), 'expected_local_fp32': want_initial.tolist(),
                                     'equal_exact': equal_initial}
        require(sorted(initial_weights) == ['common.weight', 'early.weight'], 'initial weight coverage differs')
        # SGD is never stepped; the real wrapper zero_grad clears ordinary grad/main_grad.
        # No scheduler, pending-buffer, or communication state is manually cleared.
        optimizer = mixed_optimizer_type(chained_type(model, {'sgd': torch.optim.SGD(model.parameters(), lr=0.0)}), model)
        phases = []
        zero_grad_observation = None
        with skip_dispatch():
            for window in range(2):
                losses = {}
                for source_rank in range(2):
                    for micro in range(3):
                        loss_ref = ref(inputs(source_rank, micro + window, 'cpu', True), micro == 0).sum()
                        losses[(source_rank, micro)] = float(loss_ref.detach().item())
                        loss_ref.backward()
                oracle = {name: param.grad.detach().clone().chunk(2, dim=0)[rank].contiguous()
                          for name, param in ref.named_parameters()}
                for micro in range(3):
                    model.set_requires_gradient_sync(micro == 2)
                    data = inputs(rank, micro + window, device, True)
                    loss = model(data, micro == 0).sum()
                    value = float(loss.detach().item())
                    if value != losses[(rank, micro)]:
                        failures.append('window%d micro%d loss differs from CPU reference' % (window, micro))
                    loss.backward()
                    boundary = tree_boundary(model)
                    failures.extend(boundary_failures(boundary))
                    observation = observe(model)
                    expected_pending = {}
                    if micro < 2:
                        for name in ('common.weight', 'early.weight'):
                            used = range(micro + 1) if name == 'common.weight' else (0,)
                            vectors = [inputs(rank, k + window, 'cpu', False).reshape(4, 1).expand(4, 4)
                                       for k in used]
                            want = vectors[0].clone()
                            for vec in vectors[1:]:
                                want = want + vec
                            pending = observation[name]['pending_unsharded']
                            got = None if pending is None else torch.tensor(pending['values'], dtype=torch.float32)
                            equal = pending is not None and pending['dtype'] == 'torch.float32' and torch.equal(got, want)
                            if not equal:
                                failures.append('window%d micro%d %s pending differs' % (window, micro, name))
                            expected_pending[name] = {'expected_local_fp32': want.tolist(), 'equal_exact': equal}
                    else:
                        for name, want in oracle.items():
                            row = observation[name]
                            grad = row['main_grad']
                            got = None if grad is None else torch.tensor(grad['values'], dtype=torch.float32)
                            equal = grad is not None and grad['dtype'] == 'torch.float32' and torch.equal(got, want)
                            if not equal:
                                failures.append('window%d %s main_grad differs from CPU SUM' % (window, name))
                            if row['pending_unsharded'] is not None or row['pending_reduce_partial'] is not None or row['pending_group_buckets']:
                                failures.append('window%d %s gradient remains pending' % (window, name))
                            expected_pending[name] = {'expected_local_fp32': want.tolist(), 'equal_exact': equal}
                    phases.append({'window': window, 'micro': micro, 'input': local_tensor(data).tolist(),
                                   'loss': value, 'use_early': micro == 0, 'tree_boundary': boundary,
                                   'observation': observation, 'expected': expected_pending})
                if window == 0:
                    # Persist the last micro's state before the normal zero_grad, even if a later phase fails.
                    publish(run / ('rank-%d-window-fusion-%s-boundary0.json' % (rank, str(fusion).lower())),
                            {'schema': 'npu-hccl-window-boundary-v1', 'run_id': me['run_id'], 'arm': me['arm'],
                             'rank': rank, 'fusion': fusion, 'source_pins': src, 'last_micro': phases[-1],
                             'status': 'passed' if not failures else 'failed', 'failures': list(failures)})
                    optimizer.zero_grad(set_to_none=True)
                    ref.zero_grad(set_to_none=True)
                    zero_boundary = tree_boundary(model)
                    zero_rows = observe(model)
                    failures.extend(boundary_failures(zero_boundary))
                    if any(row['main_grad'] is not None or row['pending_unsharded'] is not None or
                           row['pending_reduce_partial'] is not None or row['pending_group_buckets']
                           for row in zero_rows.values()):
                        failures.append('normal zero_grad did not clear main grads or exposed pending')
                    zero_grad_observation = {'tree_boundary': zero_boundary, 'observation': zero_rows}
        result = {'schema': 'npu-hccl-two-window-lifecycle-v1', 'run_id': me['run_id'], 'arm': me['arm'],
                  'rank': rank, 'local_rank': local, 'fusion': fusion, 'fusion_zero_copy': False,
                  'case': 'two_windows_three_micro', 'mesh_shape': [2], 'reduction': 'SUM',
                  'device': str(device), 'source_pins': src, 'initial_weights': initial_weights,
                  'phases': phases, 'after_window0_normal_zero_grad': zero_grad_observation,
                  'optimizer_step_performed': False, 'callback_count_observed': False,
                  'status': 'passed' if not failures else 'failed', 'failures': failures}
        publish(run / ('rank-%d-window-fusion-%s.json' % (rank, str(fusion).lower())), result)
        print(json.dumps({'rank': rank, 'fusion': fusion, 'case': 'two_windows_three_micro',
                          'status': result['status'], 'failures': failures}), flush=True)
        failed = failed or bool(failures)
        dist.barrier()
    return failed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=('green',), required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    require(re.fullmatch(r'[a-z0-9][a-z0-9-]{0,63}', args.run_id) is not None, 'invalid run ID')
    require(sys.dont_write_bytecode and os.environ.get('PYTHONDONTWRITEBYTECODE') == '1'
            and os.environ.get('HYPER_PARALLEL_PLATFORM') == 'torch'
            and os.environ.get('HYPER_PARALLEL_TEST_DEVICE_TYPE') == 'npu'
            and os.environ.get('TORCH_DEVICE_BACKEND_AUTOLOAD') == '0', 'fixed environment differs')
    require(Path(__file__).resolve() == WORKER and args.output_dir.is_absolute()
            and args.output_dir.is_dir() and not args.output_dir.is_symlink(), 'worker or output path differs')
    rank, local = int(os.environ['RANK']), int(os.environ['LOCAL_RANK'])
    require(int(os.environ['WORLD_SIZE']) == 2 and rank in (0, 1) and local == rank, 'requires two local ranks')
    cid = os.environ['HP_DIAG_CID']
    require(re.fullmatch(r'[a-f0-9]{64}', cid) is not None, 'invalid CID')
    cards = [int(x) for x in os.environ['HP_PHYSICAL_CARDS'].split(',')]
    require(len(cards) == 2 and len(set(cards)) == 2 and all(0 <= x <= 7 for x in cards), 'invalid physical cards')
    require(os.environ.get('ASCEND_RT_VISIBLE_DEVICES') == '0,1', 'unexpected logical visibility')
    require(not any(name == 'torch' or name.startswith('torch.') or name == 'torch_npu'
                    or name.startswith('torch_npu.') or name == 'hyper_parallel'
                    or name.startswith('hyper_parallel.') for name in sys.modules),
            'framework imported before preimport gate')
    pins = source_check(args.arm)
    scheduler_sha = pins[SCHEDULER_REL]
    worker_sha = sha(WORKER.read_bytes())
    me = process_identity(rank, local, args.run_id, cid, args.arm, cards, worker_sha, scheduler_sha)
    publish(args.output_dir / ('preimport-rank-%d.json' % rank), me)
    pre_sha = wait_release(args.output_dir / 'preimport-release.json', validate_preimport, me)

    import torch  # pylint: disable=import-outside-toplevel
    import torch_npu  # pylint: disable=import-outside-toplevel,unused-import
    import torch.distributed as dist  # pylint: disable=import-outside-toplevel
    import hyper_parallel  # pylint: disable=import-outside-toplevel
    from hyper_parallel import HSDPModule, SkipDTensorDispatch, fully_shard, init_device_mesh  # pylint: disable=import-outside-toplevel
    from hyper_parallel.core.fully_shard import hsdp_scheduler  # pylint: disable=import-outside-toplevel
    from hyper_parallel.core.fully_shard.utils import MixedPrecisionPolicy  # pylint: disable=import-outside-toplevel
    from hyper_parallel.core.optimizer.optimizer import ChainedOptimizer  # pylint: disable=import-outside-toplevel
    from hyper_parallel.components.optim.mixed_precision_optimizer import Float16OptimizerWithFloat16Params  # pylint: disable=import-outside-toplevel
    require(Path(hyper_parallel.__file__).resolve() == SOURCE / 'hyper_parallel/__init__.py'
            and Path(hsdp_scheduler.__file__).resolve() == SOURCE / SCHEDULER_REL, 'import origin differs')
    require(sha(Path(hsdp_scheduler.__file__).read_bytes()) == scheduler_sha, 'imported scheduler differs')
    require(torch.npu.device_count() == 2, 'container did not expose exactly two logical devices')
    torch.npu.set_device(local)
    torch.empty(1, dtype=torch.float32, device='npu')
    torch.npu.synchronize()
    require(torch.npu.current_device() == local, 'logical device differs')
    ready = {'me': me, 'logical_device': int(torch.npu.current_device()),
             'device_name': str(torch.npu.get_device_name()), 'preimport_release_sha256': pre_sha,
             'torch': str(torch.__version__), 'torch_npu': str(torch_npu.__version__)}
    publish(args.output_dir / ('device-ready-rank-%d.json' % rank), ready)
    mapping_sha = wait_release(args.output_dir / 'physical-mapping-release.json', validate_mapping, me)
    dist.init_process_group(backend='hccl', timeout=timedelta(seconds=120))
    require(dist.get_world_size() == 2 and dist.get_rank() == rank and dist.get_backend() == 'hccl', 'process group differs')
    try:
        failed_main = run_cases(torch, dist, fully_shard, init_device_mesh, SkipDTensorDispatch, MixedPrecisionPolicy,
                                HSDPModule, rank, local, args.output_dir, me, pins)
        failed_child = run_child_only_cases(torch, dist, fully_shard, init_device_mesh, SkipDTensorDispatch,
                                           MixedPrecisionPolicy, HSDPModule, ChainedOptimizer,
                                           Float16OptimizerWithFloat16Params, rank, local, args.output_dir, me, pins)
        failed = failed_main or failed_child
    finally:
        dist.destroy_process_group()
    publish(args.output_dir / ('terminal-rank-%d.json' % rank),
            {'run_id': args.run_id, 'rank': rank, 'failed_any_case': failed,
             'preimport_release_sha256': pre_sha, 'mapping_release_sha256': mapping_sha,
             'source_pins_after': source_check(args.arm), 'main_cases_per_rank': 8,
             'child_only_cases_per_rank': 4, 'two_window_cases_per_rank': 2,
             'callback_count_observed': False})
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()