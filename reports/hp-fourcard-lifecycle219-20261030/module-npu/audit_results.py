#!/usr/bin/env python3
"""Independently audit saved synthetic HCCL output by direct derivative, CPU-only."""
import argparse
import hashlib
import json
from pathlib import Path

RUN_ID = 'hp-pr1369-module-tree-125-green-v1'
SCHEDULER = 'hyper_parallel/core/fully_shard/hsdp_scheduler.py'
SCHEDULER_SHA = '219ce9346d528948dbe0755b50a261db025c87d1a6f207b893c22b8e9de0524d'
HEAD = '388ec460a163ae06e18b9189c490cf37da460f8a'
TREE = '1cc99a8cf4d2bc0615b75beebeefce8467f1294e'
MANIFEST_SHA = 'dd4c52de67f7617e416f67b5ea81de3d405580c1756a8d294a45aae9c5d35256'
WORKER_SHA = 'e5dfb59bb7702bb1896620fc878e88251a096db2de79c71b7c4e463d30060324'
SOURCE_KEYS = ('hyper_parallel/__init__.py', 'hyper_parallel/core/fully_shard/api.py', SCHEDULER,
               'hyper_parallel/core/fully_shard/hsdp_state.py', 'hyper_parallel/core/fully_shard/hsdp_param.py',
               'hyper_parallel/core/fully_shard/param_group.py', 'hyper_parallel/core/fully_shard/hsdp_utils.py',
               'hyper_parallel/core/fully_shard/utils.py', 'hyper_parallel/core/fully_shard/hook_function.py',
               'hyper_parallel/core/dtensor/device_mesh.py', 'hyper_parallel/core/optimizer/optimizer.py',
               'hyper_parallel/components/optim/mixed_precision_optimizer.py')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def vector(rank, micro):
    return [rank + 1, micro + 1, rank + micro + 1, 2]


def matrix(vec):
    return [[float(x)] * 4 for x in vec]


def sum_vectors(vectors):
    return [sum(v[i] for v in vectors) for i in range(4)]


def expected_shard(v, rank):
    return matrix(v[2 * rank:2 * rank + 2])


def weight_shard(name, rank):
    offset = {'common.weight': 1, 'early.weight': 3}[name]
    return [[(i * 4 + j + offset) / 16.0 for j in range(4)]
            for i in range(rank * 2, rank * 2 + 2)]


def loss(rank, micro, active):
    x = vector(rank, micro)
    offsets = {'common': [1], 'both': [1, 3], 'early': [3]}[active]
    return sum(x[i] * (i * 4 + j + offset) for offset in offsets for i in range(4) for j in range(4)) / 16.0


def read_result(folder, name, rank, fusion=None, expected_source=None):
    path = folder / name
    require(path.is_file() and not path.is_symlink(), name + ': missing/unsafe')
    raw = path.read_bytes()
    data = json.loads(raw)
    require(data.get('run_id') == RUN_ID and type(data.get('rank')) is int and data['rank'] == rank,
            name + ': run/rank differs')
    if fusion is not None:
        require(type(data.get('fusion')) is bool and data['fusion'] is fusion, name + ': fusion differs')
        require(data.get('status') == 'passed' and data.get('failures') == [], name + ': worker reported failure')
        require(data.get('source_pins') == expected_source, name + ': source key pins differ')
    return data, {'path': name, 'size': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def boundary(value, context):
    require(value.get('root_bp_state') is False, context + ': root flag remains active')
    require(value.get('scheduler_states') == {'root': None, 'common': None, 'early': None},
            context + ': scheduler state leaked')


def param(record, expected, context):
    require(record is not None and record.get('dtype') == 'torch.float32' and record.get('values') == expected,
            context + ': tensor differs from direct FP32 oracle')


def no_pending(row, context):
    require(row.get('pending_unsharded') is None and row.get('pending_reduce_partial') is None and
            row.get('pending_group_buckets') == [], context + ': pending remains')


def initial_new(data, rank, context):
    weights = data.get('initial_weights', {})
    require(set(weights) == {'common.weight', 'early.weight'}, context + ': initial names differ')
    for name in weights:
        want = weight_shard(name, rank)
        require(weights[name].get('actual') == want and weights[name].get('expected_local_fp32') == want and
                weights[name].get('equal_exact') is True, context + ': initial shard differs')


def inspect(folder):
    require(folder.is_dir() and not folder.is_symlink(), 'unsafe input directory')
    evidence, stats = [], {'main_rank_cases': 0, 'child_rank_cases': 0, 'window_rank_cases': 0,
                           'window_boundary_receipts': 0, 'rank_terminals': 0, 'backward_phases': 0}
    local = Path(__file__).resolve().parent
    manifest_raw = (local / 'inputs/source-manifest.json').read_bytes()
    require(hashlib.sha256(manifest_raw).hexdigest() == MANIFEST_SHA and
            hashlib.sha256((local / 'inputs/worker.py').read_bytes()).hexdigest() == WORKER_SHA,
            'local manifest or worker pin differs')
    manifest = json.loads(manifest_raw)
    require(manifest.get('head') == HEAD and manifest.get('tree') == TREE and len(manifest['files']) == 2679 and
            manifest['files'][SCHEDULER]['sha256'] == SCHEDULER_SHA, 'local manifest identity differs')
    expected_source = {key: manifest['files'][key]['sha256'] for key in SOURCE_KEYS}
    def gate_file(name):
        path = folder / name
        require(path.is_file() and not path.is_symlink(), name + ': missing/unsafe')
        raw = path.read_bytes();digest = hashlib.sha256(raw).hexdigest()
        evidence.append({'path': name, 'size': len(raw), 'sha256': digest})
        return json.loads(raw), digest
    pre, pre_sha = gate_file('preimport-release.json')
    mapping, mapping_sha = gate_file('physical-mapping-release.json')
    require(pre.get('kind') == 'root-verified-preimport-v1' and pre.get('run_id') == RUN_ID and
            pre.get('arm') == 'green' and pre.get('physical_cards') == [1, 2], 'preimport gate differs')
    require(mapping.get('kind') == 'root-verified-physical-device-map' and mapping.get('run_id') == RUN_ID and
            mapping.get('arm') == 'green' and mapping.get('physical_cards') == [1, 2] and
            mapping.get('cid') == pre.get('cid') and mapping.get('worker_sha256') == WORKER_SHA and
            mapping.get('source_manifest_sha256') == MANIFEST_SHA and mapping.get('scheduler_sha256') == SCHEDULER_SHA,
            'mapping gate differs')
    require(isinstance(pre.get('workers'), list) and len(pre['workers']) == 2 and
            isinstance(mapping.get('rank_proof'), list) and len(mapping['rank_proof']) == 2, 'gate rank count')
    ready_rows = []
    for rank in (0, 1):
        me, _ = gate_file('preimport-rank-%d.json' % rank)
        ready, _ = gate_file('device-ready-rank-%d.json' % rank)
        require(me.get('run_id') == RUN_ID and me.get('arm') == 'green' and me.get('cid') == pre.get('cid') and
                type(me.get('rank')) is int and me['rank'] == rank and me.get('local_rank') == rank and
                me.get('physical_cards') == [1, 2] and me.get('worker_sha256') == WORKER_SHA and
                me.get('source_manifest_sha256') == MANIFEST_SHA and me.get('scheduler_sha256') == SCHEDULER_SHA and
                me in pre['workers'], 'worker gate identity differs')
        matches = [x for x in mapping['rank_proof'] if type(x.get('rank')) is int and x['rank'] == rank]
        require(len(matches) == 1 and matches[0].get('physical_card') == rank + 1 and
                type(matches[0].get('host_pid')) is int and matches[0]['host_pid'] > 0 and
                all(matches[0].get(k) == me.get(k) for k in ('namespace_pid', 'starttick', 'pid_namespace')),
                'physical rank mapping differs')
        require(ready.get('me') == me and ready.get('logical_device') == rank and
                ready.get('preimport_release_sha256') == pre_sha and ready.get('torch') == '2.12.0+cpu' and
                ready.get('torch_npu') == '2.12.0.post3', 'device ready differs')
        ready_rows.append({'rank': rank, 'logical_device': rank, 'physical_card': rank + 1,
                           'torch': ready['torch'], 'torch_npu': ready['torch_npu']})
    for rank in (0, 1):
        for fusion in (False, True):
            for case in ('both_true', 'both_false', 'micro0_only_true', 'micro0_only_false'):
                name = 'rank-%d-fusion-%s-%s.json' % (rank, str(fusion).lower(), case)
                data, ref = read_result(folder, name, rank, fusion, expected_source);evidence.append(ref)
                require(data.get('case') == case and data.get('fusion_zero_copy') is False and
                        data.get('reduction') == 'SUM' and data.get('mesh_shape') == [2], name + ': case metadata')
                initial = {r['fqn']: r for r in data['initial']}
                require(set(initial) == {'common.weight', 'early.weight'}, name + ': initial names')
                for fqn, row in initial.items():param(row['parameter'], weight_shard(fqn, rank), name + ':initial ' + fqn)
                phases = data['phases'];require(len(phases) == 2, name + ': phase count')
                for micro, phase in enumerate(phases):
                    both = case.startswith('both') or micro == 0
                    require(phase['micro'] == micro and phase['input'] == [vector(rank, micro)] and
                            phase['use_early'] is both and phase['loss'] == loss(rank, micro, 'both' if both else 'common'),
                            name + ': inputs/loss')
                    boundary(phase['tree_boundary'], name);rows = {x['fqn']: x for x in phase['observation']}
                    require(set(rows) == {'common.weight', 'early.weight'}, name + ': observation names')
                    for fqn, row in rows.items():
                        sync_each = case.endswith('_true')
                        if micro == 0 and not sync_each:
                            param(row.get('pending_unsharded'), matrix(vector(rank, 0)), name + ':pending ' + fqn)
                            require(row.get('main_grad') is None, name + ': no-sync applied main_grad')
                        else:
                            micros = (0,) if micro == 0 or (fqn == 'early.weight' and not case.startswith('both')) else (0, 1)
                            vec = sum_vectors([vector(r, m) for r in (0, 1) for m in micros])
                            param(row.get('main_grad'), expected_shard(vec, rank), name + ':gradient ' + fqn)
                            no_pending(row, name + ':' + fqn)
                    stats['backward_phases'] += 1
                stats['main_rank_cases'] += 1
            for input_grad in (False, True):
                name = 'rank-%d-child-fusion-%s-input-grad-%s.json' % (rank, str(fusion).lower(), str(input_grad).lower())
                data, ref = read_result(folder, name, rank, fusion, expected_source);evidence.append(ref);initial_new(data, rank, name)
                require(data.get('input_requires_grad') is input_grad and data.get('fusion_zero_copy') is False and
                        data.get('callback_count_observed') is False and data.get('mesh_shape') == [2] and
                        data.get('reduction') == 'SUM', name + ': child metadata')
                require(len(data['phases']) == 2, name + ': phase count')
                for step, phase in enumerate(data['phases']):
                    require(phase['step'] == step and phase['input'] == [vector(rank, step)] and
                            phase['loss'] == loss(rank, step, 'early'), name + ': input/loss')
                    boundary(phase['tree_boundary'], name);rows = phase['observation']
                    require(set(rows) == {'common.weight', 'early.weight'} and
                            rows['common.weight']['main_grad'] is None, name + ': unused common gradient')
                    vec = sum_vectors([vector(r, m) for r in (0, 1) for m in range(step + 1)])
                    want = expected_shard(vec, rank);param(rows['early.weight']['main_grad'], want, name + ':gradient')
                    require(phase['expected_local_fp32'] == want and phase['equal_exact'] is True, name + ':reported oracle')
                    for row in rows.values():no_pending(row, name)
                    stats['backward_phases'] += 1
                stats['child_rank_cases'] += 1
            name = 'rank-%d-window-fusion-%s.json' % (rank, str(fusion).lower())
            data, ref = read_result(folder, name, rank, fusion, expected_source);evidence.append(ref);initial_new(data, rank, name)
            require(data.get('case') == 'two_windows_three_micro' and data.get('optimizer_step_performed') is False and
                    data.get('fusion_zero_copy') is False and data.get('mesh_shape') == [2] and
                    data.get('reduction') == 'SUM' and len(data['phases']) == 6, name + ': metadata')
            for index, phase in enumerate(data['phases']):
                window, micro = divmod(index, 3);both = micro == 0;inp = micro + window
                require(phase['window'] == window and phase['micro'] == micro and phase['input'] == [vector(rank, inp)] and
                        phase['use_early'] is both and phase['loss'] == loss(rank, inp, 'both' if both else 'common'), name + ':input/loss')
                boundary(phase['tree_boundary'], name);rows = phase['observation']
                require(set(rows) == {'common.weight', 'early.weight'}, name + ':names')
                for fqn, row in rows.items():
                    if micro < 2:
                        micros = range(micro + 1) if fqn == 'common.weight' else (0,)
                        want = matrix(sum_vectors([vector(rank, k + window) for k in micros]))
                        param(row.get('pending_unsharded'), want, name + ':pending ' + fqn)
                    else:
                        micros = range(3) if fqn == 'common.weight' else (0,)
                        vec = sum_vectors([vector(r, k + window) for r in (0, 1) for k in micros])
                        want = expected_shard(vec, rank)
                        param(row.get('main_grad'), want, name + ':gradient ' + fqn);no_pending(row, name + ':' + fqn)
                    require(phase['expected'][fqn]['expected_local_fp32'] == want and
                            phase['expected'][fqn]['equal_exact'] is True, name + ':reported oracle')
                stats['backward_phases'] += 1
            zero = data['after_window0_normal_zero_grad'];boundary(zero['tree_boundary'], name + ':zero')
            for row in zero['observation'].values():
                require(row['main_grad'] is None, name + ':zero main_grad');no_pending(row, name + ':zero')
            bname = 'rank-%d-window-fusion-%s-boundary0.json' % (rank, str(fusion).lower())
            rec, ref = read_result(folder, bname, rank, fusion, expected_source);evidence.append(ref)
            require(rec['last_micro'] == data['phases'][2], bname + ':pre-zero record differs')
            stats['window_boundary_receipts'] += 1;stats['window_rank_cases'] += 1
        name = 'terminal-rank-%d.json' % rank
        terminal, ref = read_result(folder, name, rank);evidence.append(ref)
        require(terminal.get('failed_any_case') is False and terminal.get('source_pins_after') == expected_source and
                terminal.get('main_cases_per_rank') == 8 and terminal.get('child_only_cases_per_rank') == 4 and
                terminal.get('two_window_cases_per_rank') == 2 and terminal.get('callback_count_observed') is False and
                terminal.get('preimport_release_sha256') == pre_sha and terminal.get('mapping_release_sha256') == mapping_sha,
                name + ': terminal differs')
        stats['rank_terminals'] += 1
    require(len(evidence) == 40 and stats == {'main_rank_cases': 16, 'child_rank_cases': 8,
            'window_rank_cases': 4, 'window_boundary_receipts': 4, 'rank_terminals': 2, 'backward_phases': 72},
            'result counts differ')
    return {'schema': 'pr1369-npu-independent-math-audit-v1', 'status': 'passed', 'run_id': RUN_ID,
            'source_head': HEAD, 'source_tree': TREE, 'source_manifest_sha256': MANIFEST_SHA,
            'worker_sha256': WORKER_SHA, 'scheduler_sha256': SCHEDULER_SHA, 'worker_source_key_pins': expected_source,
            'preimport_release_sha256': pre_sha, 'mapping_release_sha256': mapping_sha,
            'device_ready': ready_rows, 'stats': stats, 'files': evidence,
            'method': 'direct matrix derivative and per-rank SUM in ordinary CPU Python; exact synthetic FP32 values',
            'limitations': ['Callback count is not observed; state/flag are observed after backward.',
                            'Physical mapping is bound to the root release; separately inspect original live SMI and host source guard receipts.',
                            'Synthetic public-API HCCL module test, not whole-model, performance, grouped/checkpoint or partial-only coverage.']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists() and not args.output.is_symlink(), 'output exists')
    try:
        report = inspect(args.input_dir)
    except Exception as exc:
        report = {'schema': 'pr1369-npu-independent-math-audit-v1', 'status': 'failed',
                  'run_id': RUN_ID, 'error': repr(exc)}
    with args.output.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, sort_keys=True);f.write('\n')
    print(json.dumps({'status': report['status'], 'output': str(args.output)}))
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
