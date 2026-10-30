"""Synchronized native training measurement; no alternate model arithmetic."""
import json
import math
import os
import time


def require(value, message):
    if not value:
        raise RuntimeError(message)


def memory(torch):
    stats = torch.npu.memory_stats()
    return {
        'allocated_bytes': int(torch.npu.memory_allocated()),
        'reserved_bytes': int(torch.npu.memory_reserved()),
        'peak_allocated_bytes': int(torch.npu.max_memory_allocated()),
        'peak_reserved_bytes': int(torch.npu.max_memory_reserved()),
        'inactive_split_bytes': stats.get('inactive_split_bytes.all.current'),
        'allocation_retries': stats.get('num_alloc_retries'),
        'ooms': stats.get('num_ooms'),
    }


def storage_inventory(base):
    """Count live unique NPU storage; separately list aliases without double counting."""
    roles = []
    params = dict(base.model.named_parameters())
    names = {}
    for name, param in params.items():
        roles.append(('model', name, param))
        main = getattr(param, 'main_param', None)
        require(main is not None, 'missing main parameter')
        names[id(main)] = name
        roles.append(('main', name, main))
        if param.grad is not None:
            roles.append(('model_grad', name, param.grad))
        if main is not param and main.grad is not None:
            roles.append(('main_grad', name, main.grad))
    for leaf in base.optimizer.chained_optimizers:
        for param, slots in leaf.state.items():
            require(id(param) in names, 'unknown optimizer parameter')
            for slot, value in slots.items():
                if hasattr(value, 'untyped_storage'):
                    roles.append(('slot', names[id(param)] + '.' + str(slot), value))
    storage = {}
    records = []
    for role, name, value in roles:
        local = value.to_local() if hasattr(value, 'to_local') else value
        item = local.untyped_storage()
        key = (str(local.device), int(item.data_ptr()), int(item.nbytes()))
        storage.setdefault(key, []).append(role + ':' + name)
        records.append({'role': role, 'name': name, 'shape': list(local.shape),
                        'dtype': str(local.dtype), 'device': str(local.device),
                        'logical_bytes': local.numel() * local.element_size(),
                        'storage_key': list(key), 'storage_offset': int(local.storage_offset())})
    return {'records': records,
            'unique_npu_storage_bytes': sum(k[2] for k in storage if k[0].startswith('npu')),
            'unique_storages': [{'key': list(k), 'roles': v} for k, v in storage.items()],
            'scope': 'Live model/main/available grads/optimizer slots; excludes allocator cache, activations and library/HCCL memory.'}


def run(scope):
    """Run warmup and measurement from the externally audited native1500 state."""
    a = scope['a']
    torch = scope['torch']
    base = scope['base']
    trainer = scope['trainer']
    iterator = scope['iterator']
    rank = scope['rank']
    active = scope['active']
    warmup = 10
    require(a.stop_step == 1540 and a.source_step == 1500 and not a.save_final, 'measurement window differs')
    require(base.state.global_step == 1500 and base.num_micro_batches == scope['micros'], 'initial progress differs')
    path = a.output / ('rank-%d-perf.jsonl' % rank)
    require(not path.exists(), 'measurement output already exists')
    rows = []
    initial_memory = memory(torch)
    # Once per process, outside all measured steps. Fields may be unavailable.
    def clock_file(path):
        try:
            with open(path, encoding='utf-8') as stream: return {'value': stream.read().strip(), 'error': None}
        except OSError as exc: return {'value': None, 'error': str(exc)}
    try: time_namespace = {'value': os.readlink('/proc/self/ns/time'), 'error': None}
    except OSError as exc: time_namespace = {'value': None, 'error': str(exc)}
    ci = time.get_clock_info('perf_counter')
    clock_metadata = {'clock': 'perf_counter_ns', 'implementation': ci.implementation,
                      'resolution_seconds': ci.resolution, 'monotonic': ci.monotonic,
                      'adjustable': ci.adjustable, 'boot_id': clock_file('/proc/sys/kernel/random/boot_id'),
                      'time_namespace': time_namespace, 'time_namespace_offsets': clock_file('/proc/self/timens_offsets')}
    scope['save'](a.output / ('rank-%d-clock.json' % rank), clock_metadata)
    strategy = json.loads((a.config.parent / 'manifest.json').read_text())['tag']
    for step in range(1501, a.stop_step + 1):
        scope['loader_check'](step - 1)
        active.update(step=step, micro=0)
        if step == 1503:
            # Actual first two steps were checked against all four canonical input fields.
            # Later dataset indices remain observed by the existing guarded dataset.
            base.get_batch = scope['original_get_batch']
            for name, fn in scope['native_loss_entries'].items():
                setattr(scope['xing40_module'], name, fn)
            scope['optim_module'].clip_qk = scope['native_clip']
        torch.npu.synchronize()
        torch.npu.reset_peak_memory_stats()
        before = memory(torch)
        start = time.perf_counter_ns()
        result = trainer.train_step(iterator)
        torch.npu.synchronize()
        end = time.perf_counter_ns()
        elapsed = (end - start) * 1e-9
        after = memory(torch)
        require(base.state.global_step == step and math.isfinite(elapsed) and elapsed > 0, 'invalid native step')
        require(all(math.isfinite(float(result[k])) for k in ('loss', 'grad_norm')), 'nonfinite native result')
        loader = scope['loader_check'](step)
        row = {'step': step, 'phase': 'warmup' if step <= 1500 + warmup else 'measured',
               'rank': rank, 'start_perf_counter_ns': start, 'end_perf_counter_ns': end,
               'duration_seconds': elapsed, 'loss': float(result['loss']),
               'grad_norm': float(result['grad_norm']), 'memory_before': before, 'memory_after': after,
               'global_batch_size': 4, 'micro_batch_size': 1, 'num_micro_batches': base.num_micro_batches,
               'sequence_length': 4096, 'global_tokens_per_update': 16384, 'loader_after': loader,
               'scope': 'synchronized native train_step including normal data preparation, FB, optimizer, scheduler and callbacks; evidence I/O and restore excluded'}
        require(step > 1502 or active['micro'] == scope['micros'], 'verified micro count differs')
        rows.append(row)
        with path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
            stream.flush()
        if step == 1510:
            scope['save'](a.output / ('rank-%d-warm-storage.json' % rank), storage_inventory(base))
    from short_contract import slots, validate_group_progress
    expected_reads = [s['sample_index'] for step in range(1501, a.stop_step + 1)
                      for s in slots(scope['topology'], step, int(scope['mesh'].dp_rank),
                                     int(scope['mesh'].cp_rank), int(scope['mesh'].tp_rank))] if scope['owns'] else []
    require(scope['reads'] == expected_reads, 'actual dataset sample sequence differs')
    groups = scope['param_groups'](base.optimizer, dict(base.model.named_parameters()))
    validate_group_progress(scope['groups'], groups, a.stop_step)
    scope['save'](a.output / ('rank-%d-report.json' % rank), {
        'status': 'complete', 'run_id': a.run_id, 'rank': rank, 'source_step': 1500,
        'global_step': a.stop_step, 'training_updates': a.stop_step - 1500,
        'warmup_updates': warmup, 'measured_updates': a.stop_step - 1500 - warmup,
        'topology': scope['topology'], 'strategy': strategy,
        'actual_model_meshes': {name: record['mesh'] for name, record in scope['tensors']['model_local_parameters'].items()},
        'declared_fsdp': scope['cfg']['fsdp_config'], 'declared_recompute': scope['cfg']['activation_checkpoint'],
        'audit_report_sha256': scope['audit_sha'],
        'initial_memory': initial_memory, 'groups': groups, 'reads': scope['reads'],
        'verified_first_batches': scope['records'], 'qk_first_step': scope['clip_records'],
        'ce_first_step': scope['loss_entries'], 'final_loader': scope['normalize'](iterator.state_dict()),
        'final_storage': storage_inventory(base), 'final_memory': memory(torch),
        'checkpoint_saved': False, 'training_horizon': 2500,
        'scope': '40 actual native updates from sealed1500; 10 warmup+30 timed, no reference values injected.'})
    # No train-end checkpoint is requested for a performance sample; the old baseline is immutable.
    torch.distributed.barrier()
    # The original barrier has completed on all ranks; all timed steps are over.
    muon_before = scope['observe_muon'](base, scope['cfg'], rank, torch)
    raw_tensors = scope['annotate_offsets'](scope['capture'](base, scope['cb'], torch, scope['tensor_record']), rank)
    muon_after = scope['observe_muon'](base, scope['cfg'], rank, torch)
    require(muon_before == muon_after, 'Muon metadata changed during raw snapshot')
    raw_groups = scope['param_groups'](base.optimizer, dict(base.model.named_parameters()))
    momentum_by_owner = scope['link_momentum_hashes'](muon_before, raw_tensors)
    scope['save'](a.output / ('muon-post40-rank%d.json' % rank), {
        'schema': 'muon-post40-raw-observation-v1', 'run_id': a.run_id, 'rank': rank,
        'global_step': base.state.global_step, 'audit_report_sha256': scope['audit_sha'],
        'muon': muon_before, 'momentum_by_owner': momentum_by_owner,
        'raw_tensors': raw_tensors, 'groups': raw_groups,
        'scope': 'Post-barrier rank-local raw state; non-owner slots may be absent or stale; no extra state broadcast or cleanup.'})
    torch.distributed.destroy_process_group()
