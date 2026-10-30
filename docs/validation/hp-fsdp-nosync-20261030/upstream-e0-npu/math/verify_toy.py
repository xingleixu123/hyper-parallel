#!/usr/bin/env python3
"""Independently recompute archived e0+51 HCCL cases, using exact rational arithmetic."""
import hashlib
import difflib
import json
import re
from fractions import Fraction as F
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
RUN = HERE / 'run'
INPUT = HERE / 'inputs-reviewed'
HEAD = 'e0cdf62c9d66f1f9ec48c6f15df87dcf4ce80a5f'
PATCH = '51abc1d50805960b50c334c8f5a5b5c4a5b2a5e0d7984686bae3d2e553499fe2'
BASE_SCHED = '2d311ed2ee490370118100d457dfbfe4f773c7d2ba5997f180238a10a1ad4cdd'
MANIFEST = '2ee3981b6643d6ffa27c3e670bf1c0fca7500379d436874af0a761e86d3b242a'
WORKER = '5bea6387d6ce46bad195fd6d13d1338d6877fe489c1ac54770865250d04fb00e'
CID = '00fd2bc683db6f98631f639a6c9e91eea6a7a0f1ddd2ba2a93e38d6cfca5250f'
RUNID = 'hp-upstream-nosync-e0-125-green-v2'
SCHED = 'hyper_parallel/core/fully_shard/hsdp_scheduler.py'

def ensure(test, what):
    if not test: raise AssertionError(what)
def rd(path): return json.loads(path.read_bytes())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def scalar(value): return F(str(value))
def matrix(rows): return [[scalar(v) for v in row] for row in rows]
def encode(mat): return [[float(v) for v in row] for row in mat]
def repetition(vec): return [[F(v)] * 4 for v in vec]
def input_vec(rank, micro): return [rank + 1, micro + 1, rank + micro + 1, 2]
def weights(offset): return [[F(4 * i + j + offset, 16) for j in range(4)] for i in range(4)]
def loss(x, common, early, use_early):
    return sum(F(x[i]) * (common[i][j] + (early[i][j] if use_early else 0)) for i in range(4) for j in range(4))
def records(rows):
    result = {r['fqn']:r for r in rows}
    ensure(len(rows) == 2 and set(result) == {'common.weight', 'early.weight'}, 'two unique managed FQNs')
    return result
def value(rec, shape):
    ensure(rec is not None and rec['dtype'] == 'torch.float32' and rec['shape'] == shape, 'FP32 shape')
    vals = matrix(rec['values'])
    ensure(len(vals)==shape[0] and all(len(r)==shape[1] for r in vals),'declared tensor shape')
    return vals

audit = rd(RUN/'closed-readback.json')
ensure(sha(RUN/'closed-readback.json')=='0b50fe294108a46c188a4f78d2d6bafa2abcb3ac28a4d3a821e68ad312164408','closed readback pin')
ensure(audit['cid']==CID and audit['run_id']==RUNID and len(audit['files'])==91,'closed identity/inventory')
inventory={row['path']:row for row in audit['files']}
ensure(len(inventory)==91 and all(not v.get('missing') for v in inventory.values()),'closed inventory uniqueness/completeness')
public_run=[p for p in RUN.rglob('*') if p.is_file() and p.name!='closed-readback.json']
for p in public_run:
    name=p.relative_to(RUN).as_posix(); row=inventory.get(name)
    ensure(row is not None and p.stat().st_size==row['size'] and sha(p)==row['sha256'],'published original versus closed inventory '+name)
ensure(len(public_run)==56,'published original RUN inventory')
pre = rd(RUN/'preflight.json')
env = pre['snapshot']['source_and_environment']
manifest = rd(INPUT/'inputs/source-manifest.json')
ensure(sha(INPUT/'inputs/source-manifest.json') == MANIFEST == env['source_manifest_sha256'], 'manifest pin')
ensure(manifest['head']==env['source_head']==HEAD and len(manifest['files'])==env['source_file_count']==2676,'official source tree')
ensure(manifest['files'][SCHED]['sha256']==BASE_SCHED,'base scheduler')
ensure(sha(INPUT/'inputs/patched_scheduler.py')==env['scheduler_sha256']==PATCH, 'scheduler overlay')
ensure(sha(INPUT/'inputs/worker.py')==WORKER, 'worker bytes')
worker=(INPUT/'inputs/worker.py').read_text(encoding='utf-8')
ensure("dist.init_process_group(backend='hccl'" in worker and "dist.get_backend() == 'hccl'" in worker and "model.set_reduce_op_type('sum')" in worker,'pinned HCCL/SUM code path')
source_dest='/work/sources/hyper-parallel/'
overlays=[b for b in pre['snapshot']['bind_mounts'] if str(b[1]).startswith(source_dest)]
ensure(len(overlays)==1 and overlays[0][1]==source_dest+SCHED and overlays[0][2] is False,'exact sole RO overlay')
entry=rd(RUN/'private/entry-evidence.json')
ensure(entry['file_hashes']['/work/input/worker.py']==WORKER and entry['file_hashes'][source_dest+SCHED]==PATCH and entry['mount_flags'][source_dest+SCHED] is True,'entry bytes and readonly mount')
ensure(entry['visible']=='0,1' and entry['python'].startswith('3.12.14 '),'entry Python/visible')
ensure(entry['seccomp'].split()[1]=='0' and entry['capeff'].split()[1]=='000001ffffffffff','entry security context')
for group in ('devices','memory'):
    ensure(entry['cgroups'][group]==pre['snapshot']['target_cgroups'][group], 'entry target cgroup '+group)
ensure(len(entry['cgroups'])==14,'entry complete cgroup observation')

original=(HERE/'source/official-e0-scheduler.py').read_bytes()
patched=(INPUT/'inputs/patched_scheduler.py').read_bytes()
ensure(hashlib.sha256(original).hexdigest()==BASE_SCHED and hashlib.sha256(patched).hexdigest()==PATCH,'base/overlay source bytes')
patch=''.join(difflib.unified_diff(original.decode().splitlines(keepends=True),patched.decode().splitlines(keepends=True),fromfile='a/hyper_parallel/core/fully_shard/hsdp_scheduler.py',tofile='b/hyper_parallel/core/fully_shard/hsdp_scheduler.py')).encode()
ensure(patch==(HERE/'source/scheduler.patch').read_bytes(),'scheduler patch reproduces sole byte change')

x={(r,m):input_vec(r,m) for r in range(2) for m in range(2)}
first=[sum(x[(r,0)][i] for r in range(2)) for i in range(4)]
second=[sum(x[(r,1)][i] for r in range(2)) for i in range(4)]
both=[first[i]+second[i] for i in range(4)]
ensure(first==[3,2,3,4] and second==[3,4,5,4] and both==[6,6,8,8],'derivation control')
w={'common.weight':weights(1),'early.weight':weights(3)}
labels=('both_true','both_false','micro0_only_true','micro0_only_false')
results=[]; pins_seen=None
for fusion in (False,True):
    for label in labels:
        for rank in (0,1):
            path=RUN/'output'/('rank-%d-fusion-%s-%s.json'%(rank,str(fusion).lower(),label))
            d=rd(path)
            ensure(d['schema']=='npu-hccl-nosync-two-rank-v1' and d['run_id']==RUNID and d['arm']=='green' and d['status']=='passed' and d['failures']==[],'case status/id')
            ensure(d['rank']==d['local_rank']==rank and d['device']=='npu:%d'%rank and d['fusion'] is fusion and d['fusion_zero_copy'] is False and d['mesh_shape']==[2] and d['reduction']=='SUM','case identity/mesh/policy')
            ensure(set(d['model_types'])=={'root','common','early'},'wrapper coverage')
            for info in d['model_types'].values():
                ensure(info['comm_fusion'] is fusion and info['zero_copy'] is False and info['device']=='npu:%d'%rank and info['scheduler_type']=='hyper_parallel.core.fully_shard.hsdp_scheduler.HSDPSchedulerV2' and info['state_type']=='hyper_parallel.core.fully_shard.hsdp_state.HSDPStateV2','wrapper policy')
            ensure(d['model_types']['root']['module_type']=='hyper_parallel.core.fully_shard.api.HSDPToy' and all(d['model_types'][name]['module_type']=='hyper_parallel.core.fully_shard.api.HSDPBlock' for name in ('common','early')),'public HSDP wrappers')
            pins=d['source_pins']
            ensure(len(pins)==10 and SCHED in pins and all(key in manifest['files'] and want==(PATCH if key==SCHED else manifest['files'][key]['sha256']) for key,want in pins.items()),'source key pins')
            ensure(all(entry['file_hashes'][source_dest+key]==want and env['files'][source_dest+key]==want for key,want in pins.items()),'entry and preflight key pins')
            if pins_seen is None: pins_seen=pins
            ensure(pins==pins_seen,'case source consistency')
            init=records(d['initial']); phases=d['phases']
            ensure(len(phases)==2,'two microbatches')
            for name,row in init.items():
                ensure(value(row['parameter'],[2,4])==w[name][2*rank:2*rank+2] and row['grad'] is None and row['main_grad'] is None,'initial weights or grads')
                ensure(row['pending_unsharded'] is None and row['pending_reduce_partial'] is None and not row['pending_group_buckets'],'initial pending')
                ensure(row['shard_rank']==rank and row['shard_world_size']==row['shard_group_size']==2,'shard identity')
            losses=[]; observed=[]; branch_both=label.startswith('both_'); sync_each=label.endswith('_true')
            for micro,phase in enumerate(phases):
                use_early=branch_both or micro==0
                expected_loss=loss(x[(rank,micro)],w['common.weight'],w['early.weight'],use_early)
                ensure(phase['micro']==micro and phase['input']==[x[(rank,micro)]] and phase['use_early'] is use_early and scalar(phase['loss'])==expected_loss,'actual input/branch/loss')
                for name,row in records(phase['observation']).items():
                    ensure(value(row['parameter'],[2,4])==w[name][2*rank:2*rank+2],'unchanged FP32 parameter')
                    ensure(row['shard_rank']==rank and row['shard_world_size']==row['shard_group_size']==2,'phase shard')
                losses.append(float(expected_loss)); observed.append(records(phase['observation']))
            for name in ('common.weight','early.weight'):
                a,b=observed[0][name],observed[1][name]
                ensure(a['reduce_grads'] is sync_each and a['pending_reduce_partial'] is None and not a['pending_group_buckets'],'micro0 flags/buckets')
                if sync_each:
                    ensure(a['pending_unsharded'] is None and value(a['main_grad'],[2,4])==repetition(first[2*rank:2*rank+2]),'micro0 synced gradient')
                else:
                    ensure(a['main_grad'] is None and value(a['pending_unsharded'],[4,4])==repetition(x[(rank,0)]),'micro0 local FP32 accumulated full gradient')
                expected=both if name=='common.weight' or branch_both else first
                ensure(b['reduce_grads'] is True and value(b['main_grad'],[2,4])==repetition(expected[2*rank:2*rank+2]),'final gradient')
                ensure(b['pending_unsharded'] is None and b['pending_reduce_partial'] is None and not b['pending_group_buckets'],'final pending drained')
            expected_early=both if branch_both else first
            results.append({'rank':rank,'fusion':fusion,'case':label,'source_file':str(path.relative_to(RUN)).replace(chr(92),'/'),'source_sha256':sha(path),'micro_inputs':[x[(rank,0)],x[(rank,1)]],'losses':losses,'expected_final_common':encode(repetition(both[2*rank:2*rank+2])),'expected_final_early':encode(repetition(expected_early[2*rank:2*rank+2])),'observed_final_common':observed[1]['common.weight']['main_grad']['values'],'observed_final_early':observed[1]['early.weight']['main_grad']['values'],'initial_micro0_pending':None if observed[0]['early.weight']['pending_unsharded'] is None else observed[0]['early.weight']['pending_unsharded']['values'],'final_pending_unsharded':observed[1]['early.weight']['pending_unsharded'],'status':'passed'})
ensure(len(results)==16,'rank case count')
pre_release=RUN/'output/preimport-release.json'; mapping_release=RUN/'output/physical-mapping-release.json'
for rank in range(2):
    t=rd(RUN/'output'/('terminal-rank-%d.json'%rank))
    ensure(t['rank']==rank and t['run_id']==RUNID and t['failed_any_case'] is False and t['source_pins_after']==pins_seen,'rank terminal and post source key pins')
    ensure(t['preimport_release_sha256']==sha(pre_release) and t['mapping_release_sha256']==sha(mapping_release),'terminal gate linkage')
terminal=rd(RUN/'execute-terminal.json')
ensure(terminal['state']=='exited' and terminal['child_exit_code']==0 and terminal['cid']==CID and sha(RUN/'execute-terminal.json')=='1dbd02153a81bb5e32847aff715049be1bd2df86cd5e9679b9cba908895086a5','execute exit')
mapr=rd(mapping_release); pre_r=rd(pre_release)
ensure(mapr['physical_cards']==pre_r['physical_cards']==[1,2] and mapr['cid']==pre_r['cid']==CID,'physical cards and CID')
for rel,obs,review in ((pre_r,'pre-obs-root-r1.json','preimport-root-review.json'),(mapr,'map-obs-root-r1.json','mapping-root-review.json')):
    ensure(rel['observation_sha256']==inventory[obs]['sha256'] and rel['root_review_sha256']==sha(RUN/review) and rd(RUN/review)['approved'] is True,'gates bind review and observation')
observed=rd(HERE/'gate-observation-summary.json')
ensure(observed['source_sha256']==mapr['observation_sha256'] and observed['source_included'] is False,'derived observation source hash')
for rank in range(2):
    proof=mapr['rank_proof'][rank]; ready=rd(RUN/'output'/('device-ready-rank-%d.json'%rank)); row=observed['ranks'][rank]; card=rank+1
    ensure(proof['rank']==row['rank']==rank and proof['physical_card']==row['physical_card']==card and proof['host_pid']==row['host_pid'] and proof['namespace_pid']==ready['me']['namespace_pid']==row['namespace_pids'][-1] and proof['starttick']==ready['me']['starttick']==row['starttick'],'rank mapping identity')
    ensure(ready['torch']=='2.12.0+cpu' and ready['torch_npu']=='2.12.0.post3' and ready['device_name']=='Ascend910B3' and ready['logical_device']==rank and ready['me']['cid']==CID,'device runtime')
    ensure(row['uid']==['0']*4 and row['seccomp']=='0' and row['cap_effective']=='000001ffffffffff','derived rank security')
    ensure(len(row['cgroups'])==14 and all(row['cgroups'][k]==pre['snapshot']['target_cgroups'][k] for k in ('devices','memory')),'derived rank cgroups')
for action in ('observe','release'):
    for stage,tag in (('preimport','pre-obs-root-r1'),('mapping','map-obs-root-r1')):
        for rank in range(2):
            card=rank+1; base=RUN/('smi-%s-%s-%s/card-%d'%(action,stage,tag,card))
            meta=rd(base.with_suffix('.json')); so=base.with_suffix('.stdout'); se=base.with_suffix('.stderr')
            ensure(meta['physical_card']==card and meta['returncode']==0 and meta['output_complete'] is True and meta['stdout_sha256']==sha(so) and meta['stderr_sha256']==sha(se) and meta['stdout_size']==so.stat().st_size and meta['stderr_size']==se.stat().st_size,'original SMI capture')
            raw=so.read_text(encoding='utf-8')
            if stage=='preimport': ensure('No process in device.' in raw,'preimport SMI idle')
            else: ensure(re.search(r'Process id:%d(?:\D|$)'%mapr['rank_proof'][rank]['host_pid'],raw) is not None,'mapping SMI host PID')
out={'schema':'upstream-e0-51-npu-two-rank-independent-direct-math-v1','status':'passed','method':'Exact rational derivative of loss=sum(x@W_common + branch*x@W_early): dloss/dW[i,j]=x[i]; SUM rank and active micros, then partition global rows 0:2 and 2:4. Never read or compare worker oracle field.','inputs':{str(r):[x[(r,m)] for m in range(2)] for r in range(2)},'logical_weights':{k:encode(v) for k,v in w.items()},'sum_rows':{'micro0':first,'micro1':second,'both':both},'rank_case_count':len(results),'passed_rank_cases':len(results),'failed_rank_cases':0,'cases':results,'source':{'official_head':HEAD,'base_file_count':2676,'base_manifest_sha256':MANIFEST,'base_scheduler_sha256':BASE_SCHED,'scheduler_overlay_sha256':PATCH,'worker_sha256':WORKER,'sole_readonly_scheduler_overlay':True,'per_rank_source_key_pins_after':True},'runtime':{'torch':'2.12.0+cpu','torch_npu':'2.12.0.post3','python':'3.12.14','device':'Ascend910B3','physical_rank_proof':mapr['rank_proof'],'backend_basis':'Pinned actual worker requires HCCL and world/rank before any case; artifacts are not a collective-operation trace.','rank_security':'Both observed ranks UID 0, Seccomp 0, CapEff 000001ffffffffff, 14 actual cgroup entries; devices and memory match target, remaining groups are not claimed to match the init container.'},'termination':{'child_exit_code':0,'rank_terminal_failed_any_case':[False,False],'gate_hashes_verified':True,'closed_readback_sha256':sha(RUN/'closed-readback.json'),'execute_terminal_sha256':sha(RUN/'execute-terminal.json')},'boundaries':'Two-rank, 1D row-sharded SUM, FP32 synthetic model, both fusion modes and zero-copy disabled; no optimizer step, full-model numerical acceptance, HSDP 2x2, or NPU timing/performance. Empty bucket observations are snapshots, not proof of internal fusion layout or individual collectives.'}
p=HERE/'math/independent-original.json'
ensure(sha(p)=='522c87fa11ecc90a7435c2c55286073a5e028a1977d5c03190e4ea13fe5a12ea','original independent stats pin')
ensure(rd(p)==out,'independent recomputation differs from archived calculation')
print(json.dumps({'status':'passed','passed_cases':len(results),'sum_rows':out['sum_rows'],'published_run_original_files':len(public_run),'closed_inventory_count':len(inventory),'independent_original_sha256':sha(p)},sort_keys=True))
