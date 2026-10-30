#!/usr/bin/env python3
"""Independent post-run check: direct derivatives, sealed input bytes; no Torch or device."""
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys

# Derived portable adaptation: original math/state checks stay unchanged.
# It rehashes the 40 published files; it does not rehash the omitted 107-item closed archive.
HERE = Path(__file__).resolve().parent.parent
OUT = HERE / 'output'
CANDIDATE = HERE
HEAD = '388ec460a163ae06e18b9189c490cf37da460f8a'
TREE = '1cc99a8cf4d2bc0615b75beebeefce8467f1294e'
MANIFEST_SHA = 'dd4c52de67f7617e416f67b5ea81de3d405580c1756a8d294a45aae9c5d35256'
WORKER_SHA = 'e5dfb59bb7702bb1896620fc878e88251a096db2de79c71b7c4e463d30060324'
SCHED = 'hyper_parallel/core/fully_shard/hsdp_scheduler.py'
SCHED_SHA = '219ce9346d528948dbe0755b50a261db025c87d1a6f207b893c22b8e9de0524d'
RUN = 'hp-pr1369-module-tree-125-green-v1'
CLOSED_SHA = 'ec55fff0d26439135ac5f42d2fe8aac9f7d2d850e829b57e4db4578017aee84e'
ROOT_AUDIT_SHA = 'b307fabf7217d7526ae9d41602e0c0ef39bcf88b127aea4598bf0186d3174825'
VERIFY_SHA = '1a34351007183da15d442e529b2ed754784728d184a01b3b12f377cc844acd20'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def demand(condition, message):
    if not condition:
        raise AssertionError(message)


def read(path):
    demand(path.is_file() and not path.is_symlink(), 'unsafe or missing: ' + str(path))
    s = path.stat();raw = path.read_bytes();t = path.stat()
    demand((s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns) == (t.st_dev, t.st_ino, t.st_size, t.st_mtime_ns),
           'changed while reading: ' + str(path))
    return raw


def x(rank, micro):
    return (rank + 1, micro + 1, rank + micro + 1, 2)


def w(row, col, name):
    return Fraction(4 * row + col + (1 if name == 'common.weight' else 3), 16)


def loss(rank, micro, included):
    xv = x(rank, micro)
    return float(sum((Fraction(xv[i]) * w(i, j, name) for name in included for i in range(4) for j in range(4)), Fraction(0)))


def gradient(rank_micro_pairs):
    # d sum_j(x @ W)_j / dW_ij = x_i. No runtime tensor or worker oracle is used.
    pairs = tuple(rank_micro_pairs)
    return tuple(sum(x(rank, micro)[i] for rank, micro in pairs) for i in range(4))


def full_matrix(vector):
    return [[v] * 4 for v in vector]


def shard(vector, rank):
    return full_matrix(vector[2 * rank:2 * rank + 2])


def weight_shard(name, rank):
    return [[float(w(i, j, name)) for j in range(4)] for i in range(2 * rank, 2 * rank + 2)]


def tensor(record, expected, label):
    demand(isinstance(record, dict) and record.get('dtype') == 'torch.float32' and
           record.get('values') == expected, label + ': actual FP32 values differ')
    if 'shape' in record:
        demand(record['shape'] == [len(expected), 4], label + ': shape differs')


def boundary(value, label):
    demand(value == {'root_bp_state': False, 'scheduler_states': {'root': None, 'common': None, 'early': None}},
           label + ': tree state/flag differs')


def no_pending(row, label):
    demand(row.get('pending_unsharded') is None and row.get('pending_reduce_partial') is None and
           row.get('pending_group_buckets') == [], label + ': unconsumed pending')


def rows(ph, list_mode):
    ob = ph['observation']
    value = {r['fqn']: r for r in ob} if list_mode else ob
    demand(set(value) == {'common.weight', 'early.weight'}, 'unexpected parameter set')
    return value


def validate_common(obj, name, rank, fusion, source_pins):
    demand(obj.get('run_id') == RUN and obj.get('arm') == 'green' and type(obj.get('rank')) is int and obj['rank'] == rank and
           type(obj.get('fusion')) is bool and obj['fusion'] is fusion and obj.get('status') == 'passed' and
           obj.get('failures') == [] and obj.get('source_pins') == source_pins, name + ': identity/status/pins')


def check_init_main(obj, rank, label):
    initial = {v['fqn']: v for v in obj['initial']}
    demand(set(initial) == {'common.weight', 'early.weight'}, label + ': initial FQNs')
    for key, row in initial.items():
        tensor(row['parameter'], weight_shard(key, rank), label + ': initial ' + key)
        demand(row.get('main_grad') is None and row.get('pending_unsharded') is None, label + ': initial gradients')
        demand((row.get('shard_world_size'), row.get('shard_rank'), row.get('shard_group_size')) == (2, rank, 2),
               label + ': shard group')


def check_init_aux(obj, rank, label):
    weights = obj['initial_weights']
    demand(set(weights) == {'common.weight', 'early.weight'}, label + ': initial FQNs')
    for key, row in weights.items():
        demand(row.get('actual') == weight_shard(key, rank) and row.get('equal_exact') is True,
               label + ': initial weight shard')


def check_phase(ph, rank, micro, active, label, window=None):
    expected_micro = micro if window is None else window + micro
    demand(ph['input'] == [list(x(rank, expected_micro))] and
           ph['loss'] == loss(rank, expected_micro, active), label + ': input/loss')
    boundary(ph['tree_boundary'], label)


def main():
    demand(len(sys.argv) == 2, 'usage: python -B analysis/independent_math_public.py <new-output.json>')
    original_raw = read(HERE / 'analysis/independent-math-original.json')
    demand(sha(original_raw) == 'f4b1f3707128261c806b3edbffbd00e8ca077f6e0060aee1d2217976e1b3efbf', 'original report SHA')
    original_report = json.loads(original_raw)
    demand(original_report['status'] == 'passed' and original_report['closed_sha256'] == CLOSED_SHA and
           original_report['closed_items_independently_hashed'] == 107 and
           original_report['source_before_after_identical'] is True, 'original report provenance')
    entries = [dict(x, path='output/' + x['path']) for x in original_report['output_files']]
    demand(len(entries) == 40, '40 original output entries')
    listed = {e['path']: e for e in entries};demand(len(listed) == 40, 'duplicate output path')
    for rel, entry in listed.items():
        path = HERE / rel
        demand(path.resolve().is_relative_to(OUT.resolve()), 'path escaped output')
        raw = read(path);demand(len(raw) == entry['size'] and sha(raw) == entry['sha256'], 'output bytes differ: ' + rel)
    manifest_raw = read(CANDIDATE / 'inputs/source-manifest.json')
    demand(sha(manifest_raw) == MANIFEST_SHA and sha(read(CANDIDATE / 'inputs/worker.py')) == WORKER_SHA, 'candidate pins')
    manifest = json.loads(manifest_raw)
    demand(manifest['head'] == HEAD and manifest['tree'] == TREE and len(manifest['files']) == 2679, 'source manifest')
    keys = ('hyper_parallel/__init__.py', 'hyper_parallel/core/fully_shard/api.py', SCHED,
            'hyper_parallel/core/fully_shard/hsdp_state.py', 'hyper_parallel/core/fully_shard/hsdp_param.py',
            'hyper_parallel/core/fully_shard/param_group.py', 'hyper_parallel/core/fully_shard/hsdp_utils.py',
            'hyper_parallel/core/fully_shard/utils.py', 'hyper_parallel/core/fully_shard/hook_function.py',
            'hyper_parallel/core/dtensor/device_mesh.py', 'hyper_parallel/core/optimizer/optimizer.py',
            'hyper_parallel/components/optim/mixed_precision_optimizer.py')
    expected_sources = {k: manifest['files'][k]['sha256'] for k in keys}
    terminal_raw = read(HERE / 'evidence/execute-terminal.json');terminal = json.loads(terminal_raw)
    demand(sha(terminal_raw) == original_report['terminal_sha256'] and terminal['state'] == 'exited' and
           type(terminal['child_exit_code']) is int and terminal['child_exit_code'] == 0 and
           terminal['run_id'] == RUN, 'run terminal differs')
    names = {e['path'][len('output/'):] for e in entries if e['path'].startswith('output/')}
    expected_names = {'preimport-release.json', 'physical-mapping-release.json'}
    for rank in (0, 1):
        expected_names.update({'preimport-rank-%d.json' % rank, 'device-ready-rank-%d.json' % rank,
                               'terminal-rank-%d.json' % rank})
        for fusion in (False, True):
            tag = str(fusion).lower()
            expected_names.update('rank-%d-fusion-%s-%s.json' % (rank, tag, case)
                                  for case in ('both_true', 'both_false', 'micro0_only_true', 'micro0_only_false'))
            expected_names.update('rank-%d-child-fusion-%s-input-grad-%s.json' % (rank, tag, str(flag).lower())
                                  for flag in (False, True))
            expected_names.update({'rank-%d-window-fusion-%s.json' % (rank, tag),
                                   'rank-%d-window-fusion-%s-boundary0.json' % (rank, tag)})
    demand(len(expected_names) == 40 and names == expected_names, '40 output inventory differs')
    def item(name):return json.loads(read(OUT / name))
    pregate = item('preimport-release.json');mapgate = item('physical-mapping-release.json')
    pregate_sha, mapgate_sha = sha(read(OUT/'preimport-release.json')), sha(read(OUT/'physical-mapping-release.json'))
    demand(pregate['kind'] == 'root-verified-preimport-v1' and mapgate['kind'] == 'root-verified-physical-device-map' and
           pregate['run_id'] == mapgate['run_id'] == RUN and pregate['arm'] == mapgate['arm'] == 'green' and
           pregate['cid'] == mapgate['cid'] == terminal['cid'] and pregate['physical_cards'] == mapgate['physical_cards'] == [1, 2] and
           mapgate['worker_sha256'] == WORKER_SHA and mapgate['source_manifest_sha256'] == MANIFEST_SHA and
           mapgate['scheduler_sha256'] == SCHED_SHA, 'gates differ')
    ready_summary = []
    for rank in (0, 1):
        me = item('preimport-rank-%d.json' % rank);ready = item('device-ready-rank-%d.json' % rank)
        demand(me in pregate['workers'] and me['rank'] == me['local_rank'] == rank and me['cid'] == terminal['cid'] and
               me['worker_sha256'] == WORKER_SHA and me['source_manifest_sha256'] == MANIFEST_SHA and me['scheduler_sha256'] == SCHED_SHA,
               'preimport identity')
        proof = [p for p in mapgate['rank_proof'] if p['rank'] == rank]
        demand(len(proof) == 1 and proof[0]['physical_card'] == rank + 1 and proof[0]['host_pid'] > 0 and
               all(proof[0][k] == me[k] for k in ('namespace_pid', 'starttick', 'pid_namespace')), 'mapping proof')
        demand(ready['me'] == me and ready['logical_device'] == rank and ready['preimport_release_sha256'] == pregate_sha and
               ready['torch'] == '2.12.0+cpu' and ready['torch_npu'] == '2.12.0.post3', 'ready identity')
        ready_summary.append({'rank': rank, 'logical': rank, 'physical_release': rank + 1, 'device_name': ready['device_name']})
    counts = {'main_rank_cases': 0, 'child_rank_cases': 0, 'window_rank_cases': 0, 'backward_phases': 0,
              'window_pre_zero_receipts': 0, 'rank_terminals': 0}
    examples = {}
    for rank in (0, 1):
        for fusion in (False, True):
            tag = str(fusion).lower()
            for case in ('both_true', 'both_false', 'micro0_only_true', 'micro0_only_false'):
                name = 'rank-%d-fusion-%s-%s.json' % (rank, tag, case);obj = item(name)
                validate_common(obj, name, rank, fusion, expected_sources);check_init_main(obj, rank, name)
                demand(obj['case'] == case and obj['mesh_shape'] == [2] and obj['reduction'] == 'SUM' and obj['fusion_zero_copy'] is False, name + ': metadata')
                for unit in ('root', 'common', 'early'):
                    unit_info = obj['model_types'][unit]
                    demand(unit_info['comm_fusion'] is fusion and unit_info['zero_copy'] is False and
                           unit_info['scheduler_type'] == 'hyper_parallel.core.fully_shard.hsdp_scheduler.HSDPSchedulerV2' and
                           unit_info['device'] == 'npu:%d' % rank, name + ': wrapper')
                demand(len(obj['phases']) == 2, name + ': phase count')
                for micro, ph in enumerate(obj['phases']):
                    active = ('common.weight','early.weight') if case.startswith('both') or micro == 0 else ('common.weight',)
                    demand(ph['micro'] == micro and ph['use_early'] is ('early.weight' in active), name + ': phase logic')
                    check_phase(ph, rank, micro, active, name);obs = rows(ph, True)
                    for fqn, row in obs.items():
                        demand(row['state'] is None and row['reduce_grads'] is (case.endswith('_true') or micro == 1), name + ': per-unit state/sync')
                        if micro == 0 and case.endswith('_false'):
                            tensor(row['pending_unsharded'], full_matrix(x(rank, 0)), name + ': pending')
                            demand(row['main_grad'] is None and row['pending_reduce_partial'] is None and row['pending_group_buckets'] == [], name + ': no-sync extra state')
                        else:
                            ms = (0,) if micro == 0 or (fqn == 'early.weight' and not case.startswith('both')) else (0,1)
                            v = gradient((r,m) for r in (0,1) for m in ms)
                            tensor(row['main_grad'], shard(v, rank), name + ': main grad');no_pending(row, name)
                    counts['backward_phases'] += 1
                if not fusion and case == 'micro0_only_false':
                    examples['main_rank%d' % rank] = {'micro0_early_pending_first_column': [r[0] for r in rows(obj['phases'][0],True)['early.weight']['pending_unsharded']['values']],
                                                    'final_early_shard_first_column': [r[0] for r in rows(obj['phases'][1],True)['early.weight']['main_grad']['values']]}
                counts['main_rank_cases'] += 1
            for input_grad in (False, True):
                name = 'rank-%d-child-fusion-%s-input-grad-%s.json' % (rank,tag,str(input_grad).lower());obj = item(name)
                validate_common(obj,name,rank,fusion,expected_sources);check_init_aux(obj,rank,name)
                demand(obj['input_requires_grad'] is input_grad and obj['fusion_zero_copy'] is False and obj['mesh_shape']==[2] and
                       obj['reduction']=='SUM' and obj['callback_count_observed'] is False and len(obj['phases'])==2, name + ': metadata')
                for step, ph in enumerate(obj['phases']):
                    demand(ph['step'] == step, name + ': child step');check_phase(ph,rank,step,('early.weight',),name)
                    obs=rows(ph,False);demand(obs['common.weight']['main_grad'] is None,name+': unused gradient')
                    v=gradient((r,m) for r in (0,1) for m in range(step+1));tensor(obs['early.weight']['main_grad'],shard(v,rank),name+': child grad')
                    for row in obs.values():no_pending(row,name)
                    counts['backward_phases']+=1
                counts['child_rank_cases']+=1
            name='rank-%d-window-fusion-%s.json'%(rank,tag);obj=item(name)
            validate_common(obj,name,rank,fusion,expected_sources);check_init_aux(obj,rank,name)
            demand(obj['case']=='two_windows_three_micro' and obj['fusion_zero_copy'] is False and obj['mesh_shape']==[2] and
                   obj['reduction']=='SUM' and obj['optimizer_step_performed'] is False and obj['callback_count_observed'] is False and
                   len(obj['phases'])==6,name+': metadata')
            for i,ph in enumerate(obj['phases']):
                window,micro=divmod(i,3);active=('common.weight','early.weight') if micro==0 else ('common.weight',)
                demand((ph['window'],ph['micro'])==(window,micro) and ph['use_early'] is (micro==0),name+': phase logic')
                check_phase(ph,rank,micro,active,name,window);obs=rows(ph,False)
                for fqn,row in obs.items():
                    if micro<2:
                        ms=range(micro+1) if fqn=='common.weight' else (0,)
                        tensor(row['pending_unsharded'],full_matrix(gradient((rank,k+window) for k in ms)),name+': pending')
                        demand(row['main_grad'] is None and row['pending_reduce_partial'] is None and row['pending_group_buckets']==[],name+': no-sync extra state')
                    else:
                        ms=range(3) if fqn=='common.weight' else (0,)
                        v=gradient((r,k+window) for r in (0,1) for k in ms)
                        tensor(row['main_grad'],shard(v,rank),name+': window grad');no_pending(row,name)
                counts['backward_phases']+=1
            zero=obj['after_window0_normal_zero_grad'];boundary(zero['tree_boundary'],name+': zero')
            for row in zero['observation'].values():demand(row['main_grad'] is None,name+': grad after zero');no_pending(row,name+': zero')
            pre_name='rank-%d-window-fusion-%s-boundary0.json'%(rank,tag);pre=item(pre_name)
            validate_common(pre,pre_name,rank,fusion,expected_sources);demand(pre['last_micro']==obj['phases'][2],pre_name+': pre-zero differs')
            counts['window_pre_zero_receipts']+=1;counts['window_rank_cases']+=1
            if not fusion:
                examples['window_rank%d'%rank]={'window0_common_shard_first_column':[r[0] for r in obj['phases'][2]['observation']['common.weight']['main_grad']['values']],
                                               'window1_common_shard_first_column':[r[0] for r in obj['phases'][5]['observation']['common.weight']['main_grad']['values']],
                                               'window0_early_shard_first_column':[r[0] for r in obj['phases'][2]['observation']['early.weight']['main_grad']['values']],
                                               'window1_early_shard_first_column':[r[0] for r in obj['phases'][5]['observation']['early.weight']['main_grad']['values']]}
        end=item('terminal-rank-%d.json'%rank)
        demand(end['rank']==rank and end['run_id']==RUN and end['failed_any_case'] is False and end['source_pins_after']==expected_sources and
               end['preimport_release_sha256']==pregate_sha and end['mapping_release_sha256']==mapgate_sha and
               (end['main_cases_per_rank'],end['child_only_cases_per_rank'],end['two_window_cases_per_rank'])==(8,4,2) and
               end['callback_count_observed'] is False,'rank terminal differs')
        counts['rank_terminals']+=1
    demand(counts == {'main_rank_cases':16,'child_rank_cases':8,'window_rank_cases':4,'backward_phases':72,
                      'window_pre_zero_receipts':4,'rank_terminals':2}, 'case totals')
    root_raw=read(HERE/'analysis/root-auditor-original.json');demand(sha(root_raw)==ROOT_AUDIT_SHA,'root audit SHA')
    root=json.loads(root_raw);demand(root['status']=='passed' and root['source_manifest_sha256']==MANIFEST_SHA and
                                     root['worker_sha256']==WORKER_SHA and root['scheduler_sha256']==SCHED_SHA, 'root audit identity')
    root_files={x['path']:(x['size'],x['sha256']) for x in root['files']}
    demand(set(root_files)==names and all(root_files[n]==(listed['output/'+n]['size'],listed['output/'+n]['sha256']) for n in names),
           'root audit file set differs')
    report={'schema':'pr1369-npu-independent-direct-derivative-v1','status':'passed','head':HEAD,'tree':TREE,
            'source_manifest_sha256':MANIFEST_SHA,'worker_sha256':WORKER_SHA,'scheduler_sha256':SCHED_SHA,
            'closed_sha256':CLOSED_SHA,'terminal_sha256':sha(terminal_raw),'root_audit_sha256':ROOT_AUDIT_SHA,
            'root_local_verification_sha256':VERIFY_SHA,
            'historical_closed_items_independently_hashed':original_report['closed_items_independently_hashed'],
            'closed_107_rehashed_in_this_portable_run':False,
            'source_before_after_original_report':original_report['source_before_after_identical'],
            'output_files_verified':len(names),'ready':ready_summary,
            'preimport_release_sha256':pregate_sha,'physical_mapping_release_sha256':mapgate_sha,
            'root_audit_same_40_files':True,'counts':counts,'examples':examples,
            'direct_global_gradient_vectors':{'two_micro_both':[6,6,8,8],'micro0_only':[3,2,3,4],
                    'window0_common':[9,12,15,12],'window0_early':[3,2,3,4],
                    'window1_common':[9,18,21,12],'window1_early':[3,4,5,4]},
            'output_files':[{'path':n,'size':listed['output/'+n]['size'],'sha256':listed['output/'+n]['sha256']} for n in sorted(names)],
            'method':'Exact rational input and weight arithmetic; directly derived d(sum(xW))/dW_ij=x_i; no worker CPU oracle or Torch.',
            'limits':['No callback count was observed; only post-backward scheduler state and shared flag.',
                      'Device mapping is verified against root release metadata, not independently adjudicated raw SMI.',
                      '388ec/e0 complete module source is not the 1a+219 full-model source tree.',
                      'Synthetic two-rank FP32 SUM, no activation checkpoint/recompute, grouped, zero-copy=true, partial-only, full-model or native full checkpoint coverage.']}
    output=Path(sys.argv[1])
    with output.open('x',encoding='utf-8',newline='\n') as f:json.dump(report,f,ensure_ascii=False,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps({'status':report['status'],'counts':counts,'historical_closed_items':original_report['closed_items_independently_hashed'],'output_files':len(names),
                      'math_sha256':sha(output.read_bytes()),'examples':examples},ensure_ascii=False))


if __name__ == '__main__':
    main()
