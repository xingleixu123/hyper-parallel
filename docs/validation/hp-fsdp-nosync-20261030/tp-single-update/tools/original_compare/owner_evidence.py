"""Validate independent restore-only Muon cache metadata against completed train runs.

This module never infers an owner from values or hashes. A result describes a
subsequent restore, not a direct recording of the earlier optimizer step.
"""
from pathlib import Path
import re
import compare_step as cmp

SOURCE='0f168e2ce3279cb315fc27d60d06693a17a5626a228fbf777146dda9da3e977d'
OVERLAY='51abc1d50805960b50c334c8f5a5b5c4a5b2a5e0d7984686bae3d2e553499fe2'
CONTRACT='1fd4ac373a20f3edd8620cb0ac0593cf8f0ca5876a0d7f7227089358ef5cb633'
OWNER_HELPER='eb66e16dcf5b8da0c7d2e1f8a18faf964513b22b57825edb5dbbe97e7668e4a0'
OWNER_WORKER='0f5e545b07e86217f2b45d9c8329f2744de928013c37f0e1dae8472edc5f2aeb'
TRAIN_WORKER='fa71e6432807c5549ab926023c0d6d3adc3be5d7f598c0e44c3612ff9c63760c'
CONFIG={'sync':'d779f298fe8e335a4bd9cfec081705eb6dbf177a2a9371d14a9a30bdce5fea7b',
        'deferred':'acf412733e3bed399510709a6aeb9456bc7c3fd6c24ec4993e69adc1891a3481'}
LAYOUT=('shape','dtype','global_shape','placements','mesh')
TARGET_KEYS=('tensors','bindings','groups','extra','loader','initial_sampler','mesh','source_contract_sha256')
MUON='hyper_parallel.core.optimizer.muon.Muon'


def require(x,msg):
    if not x: raise RuntimeError(msg)


def item_layout(actual, expected, what):
    require(isinstance(actual,dict) and all(actual.get(k)==expected.get(k) for k in LAYOUT),
            'owner target layout differs: '+what)


def axis(mesh,dim):
    ranks=mesh['rank_list']; c=mesh['coordinate']
    return [ranks[i*2+c[1]] for i in range(2)] if dim==0 else [ranks[c[0]*2+i] for i in range(2)]


def records_for_rank(doc,target,rank):
    assignment=doc['assignment']
    require(all(k in assignment for k in ('configured_replica_count','effective_replica_count',
                                         'explicit_flag_exists','momentum_update_fn'))
            and assignment.get('schema')=='tp-muon-existing-assignment-v1' and assignment.get('rank')==rank
            and assignment.get('configured_replica_count') is None
            and assignment.get('effective_replica_count') is None
            and assignment.get('explicit_flag_exists') is False, 'assignment identity differs')
    callback=assignment.get('momentum_update_fn')
    require(isinstance(callback,dict) and callback.get('present') is True and callback.get('is_none') is True,
            'Muon momentum callback unknown or custom')
    target_groups=[g for g in target['groups'] if g.get('optimizer_class')==MUON]
    groups=assignment.get('groups')
    require(isinstance(groups,list) and len(groups)==len(target_groups) and len(groups)>0,
            'Muon group count differs')
    by_fqn={}; all_fqn=[]
    for group_index,(group,target_group) in enumerate(zip(groups,target_groups)):
        require(group.get('group_index')==group_index,'Muon group order differs')
        members=target_group['members']
        require(len(members)==len(set(members)),'duplicate group FQN')
        momentum=group.get('momentum',{})
        target_momentum=target_group.get('options',{}).get('momentum')
        require(momentum.get('present') is True and momentum.get('kind')=='tuple'
                and target_momentum=={'tuple':momentum.get('value')},'Muon group momentum differs')
        group_records=[]
        for rec in group.get('no_comm',[]):
            require(rec.get('owner_kind')=='no_comm_local' and rec.get('current_owned') is True
                    and rec.get('record_index') is None,'no_comm metadata differs')
            group_records.append((rec,None))
        for batch in group.get('assignments',[]):
            require(type(batch.get('batch_group_index')) is int and type(batch.get('sub_batch_index')) is int,
                    'assignment batch index differs')
            for rec in batch.get('records',[]):
                index=rec.get('record_index')
                require(type(index) is int and 0<=index<len(members) and members[index]==rec.get('fqn'),
                        'record index/group order differs')
                group_records.append((rec,batch))
        require(len(group_records)==len(members) and {r['fqn'] for r,_ in group_records}==set(members),
                'Muon FQN group coverage differs')
        for rec,batch in group_records:
            name=rec['fqn']; require(name not in by_fqn,'duplicate Muon FQN')
            bind=rec['binding_and_layout']
            require(bind.get('main_is_model') is True
                    and target['bindings'][name].get('main_is_model') is True,'binding differs')
            item_layout(bind['model'],target['tensors']['model_local_parameters'][name],name+' model')
            item_layout(bind['main'],target['tensors']['optimizer_local_main_parameters'][name],name+' main')
            mesh=bind['main']['mesh']; require(mesh['shape']==[2,2] and mesh['rank_list']==[0,1,2,3]
                    and mesh['rank_list'][mesh['coordinate'][0]*2+mesh['coordinate'][1]]==rank,
                    'owner mesh differs')
            slot=target['tensors']['optimizer_local_slots'].get(name+'.momentum_buffer')
            state=rec['raw_state']
            require(state.get('momentum_buffer_present')==(slot is not None)
                    and state.get('audited_momentum_sha256')==(slot['local_sha256'] if slot else None),
                    'restored momentum presence/hash differs')
            normalized={'fqn':name,'group_index':group_index,'record_index':rec.get('record_index'),
                        'current_owned':rec.get('current_owned'),'owner_coord':None,'replica_pgs':[],
                        'batch':None,'is_shard':None,'raw_state':state,'binding_and_layout':bind}
            if batch is not None:
                sizes=batch['replicate_sizes']; coord=batch['replicate_coordinate']
                placements=mesh['placements']
                reps=[i for i,p in enumerate(placements) if p['class'].rsplit('.',1)[-1]=='Replicate']
                shards=[i for i,p in enumerate(placements) if p['class'].rsplit('.',1)[-1] in ('Shard','StridedShard')]
                require(len(reps)+len(shards)==len(placements)
                        and sizes==[mesh['shape'][i] for i in reps]
                        and coord==[mesh['coordinate'][i] for i in reps],
                        'replica coordinates differ')
                rpg=rec['replica_pgs'];spg=rec['shard_pgs']
                require(len(rpg)==len(reps) and len(spg)==len(shards)
                        and all(x is not None and x==axis(mesh,i) for x,i in zip(rpg,reps))
                        and all(x is not None and x==axis(mesh,i) for x,i in zip(spg,shards)),
                        'cached PG membership differs from mesh')
                owner=rec.get('owner_coord')
                require(owner is None and rec.get('current_owned') is False or
                        isinstance(owner,list) and len(owner)==len(coord)
                        and all(type(x) is int and 0<=x<n for x,n in zip(owner,sizes))
                        and rec.get('current_owned') is (owner==coord),'owner/cache membership differs')
                broadcast=rec.get('cached_broadcast')
                if owner is not None and any(n>1 for n in sizes):
                    require(isinstance(broadcast,dict) and broadcast.get('owner_coord')==owner
                            and broadcast.get('replica_pgs')==rpg,'broadcast owner differs')
                normalized.update(owner_coord=owner,replica_pgs=rpg,
                    batch={'batch_group_index':batch['batch_group_index'],'sub_batch_index':batch['sub_batch_index'],
                           'replicate_sizes':sizes,'replicate_coordinate':coord},is_shard=rec.get('is_shard'))
            by_fqn[name]=normalized
            all_fqn.append(name)
    require(len(all_fqn)==assignment.get('muon_parameter_count') and len(all_fqn)==len(set(all_fqn)),
            'Muon total coverage differs')
    return by_fqn


def read_arm(arm,train_run,owner_run,train_data):
    require(arm in CONFIG,'unknown arm'); train_run=Path(train_run);owner_run=Path(owner_run)
    expected=arm=='sync'
    train_create,train_create_sha=cmp.read_json(train_run/'create-receipt.json')
    owner_create,owner_create_sha=cmp.read_json(owner_run/'create-receipt.json')
    train_terminal,train_terminal_sha=cmp.read_json(train_run/'execute-terminal.json')
    owner_terminal,owner_terminal_sha=cmp.read_json(owner_run/'execute-terminal.json')
    train_id='tp-fullmodel-nosync-'+arm+'-125-r1'; owner_id='tp-owner-'+arm+'-125-r2'
    for create,terminal,run_id in ((train_create,train_terminal,train_id),(owner_create,owner_terminal,owner_id)):
        pins=create['pins']
        require(create.get('run_id')==terminal.get('run_id')==run_id and create.get('arm')==arm
                and create.get('cid')==terminal.get('cid') and terminal.get('status')=='complete'
                and terminal.get('child_exit_code')==0 and terminal.get('source_files_verified_after')==2661,
                'run terminal/create differs')
        require(pins.get('source_manifest_sha256')==SOURCE and pins.get('overlay_sha256')==OVERLAY
                and pins.get('config_sha256')==CONFIG[arm] and pins.get('source_contract_sha256')==CONTRACT,
                'run source/config differs')
        outputs=terminal.get('rank_outputs')
        require(isinstance(outputs,list) and len(outputs)==4 and {x['rank'] for x in outputs}==set(range(4)),
                'terminal rank coverage differs')
    require(train_create['pins'].get('worker_sha256')==TRAIN_WORKER
            and owner_create['pins'].get('worker_sha256')==OWNER_WORKER
            and owner_create['pins'].get('owner_sha256')==OWNER_HELPER,'worker/helper pin differs')
    tout={x['rank']:x for x in train_terminal['rank_outputs']}
    oout={x['rank']:x for x in owner_terminal['rank_outputs']}
    audit,audit_sha=cmp.read_json(owner_run/'root-external-audit-report.json')
    flags=('source_live_compared','target_live_compared','groups_compared','extra_compared','checkpoint_identity_stable')
    require(audit.get('status')=='passed' and audit.get('run_id')==owner_id and audit.get('source_step')==1500
            and audit.get('topology')=='tp2-dp2' and audit.get('source_contract_sha256')==CONTRACT
            and audit.get('global_readback_sha256')=='f8a11dbae22b44c1c9ce0edf7f3edb05899b36fe87e63343135ebe8e0cf76466'
            and audit.get('role_counts')==cmp.COUNTS and len(audit.get('comparisons',[]))==258
            and all(audit.get(x) is True for x in flags),'owner restore audit differs')
    result={}; sources={'train_create':train_create_sha,'train_terminal':train_terminal_sha,
                        'owner_create':owner_create_sha,'owner_terminal':owner_terminal_sha,
                        'owner_audit':audit_sha,'train_audit':train_data['audit_sha256'],'rank_documents':{}}
    for rank in range(4):
        row=train_data['rows'][rank]
        require(tout[rank].get('report_sha256')==row['report_sha256']
                and tout[rank].get('target_sha256')==row['target_sha256']
                and tout[rank].get('missing_trainable_gradients')==[],'training terminal rank differs')
        target,tsha=cmp.read_json(owner_run/f'target-live-rank-{rank}.json')
        report,rsha=cmp.read_json(owner_run/f'owner-rank-{rank}.json')
        require(oout[rank].get('report_sha256')==rsha and oout[rank].get('target_sha256')==tsha
                and report.get('schema')=='tp-muon-restore-only-rank-v1'
                and report.get('status')=='complete' and report.get('rank')==rank
                and report.get('run_id')==target.get('run_id')==owner_id
                and target.get('schema')=='native1500-final-short-target-v1'
                and target.get('source_step')==1500
                and report.get('source_step')==report.get('global_step')==1500
                and report.get('training_updates')==0 and report.get('requires_grad_sync_config') is expected
                and report.get('target_sha256')==tsha and report.get('audit_report_sha256')==audit_sha
                and audit['target_rank_sha256'][str(rank)]==tsha
                and report.get('optimizer_step_guard_retained') is True and report.get('data_reads')==[]
                and report.get('topology')==target.get('topology')=='tp2-dp2'
                and report.get('mesh')==target.get('mesh')
                and report.get('initial_sampler')==target.get('initial_sampler'),
                'owner rank closure differs')
        require(all(target.get(k)==row['target'].get(k) for k in TARGET_KEYS),
                'cross-run restored target differs')
        result[rank]=records_for_rank(report,target,rank)
        sources['rank_documents'][str(rank)]={'owner':rsha,'target':tsha}
    return result,sources


def verify_other_train(arm,run,data):
    create,create_sha=cmp.read_json(Path(run)/'create-receipt.json')
    terminal,terminal_sha=cmp.read_json(Path(run)/'execute-terminal.json')
    run_id='tp-fullmodel-nosync-'+arm+'-125-r1'
    pins=create['pins']
    require(create.get('arm')==arm and create.get('run_id')==terminal.get('run_id')==run_id
            and create.get('cid')==terminal.get('cid') and terminal.get('status')=='complete'
            and terminal.get('child_exit_code')==0 and terminal.get('source_files_verified_after')==2661,
            'other training terminal differs')
    require(pins.get('source_manifest_sha256')==SOURCE and pins.get('overlay_sha256')==OVERLAY
            and pins.get('config_sha256')==CONFIG[arm] and pins.get('source_contract_sha256')==CONTRACT
            and pins.get('worker_sha256')==TRAIN_WORKER,'other training pins differ')
    outputs=terminal.get('rank_outputs')
    require(isinstance(outputs,list) and len(outputs)==4 and {x['rank'] for x in outputs}==set(range(4)),
            'other terminal rank coverage')
    rows={x['rank']:x for x in outputs}
    rankdocs={}
    for rank in range(4):
        row=data['rows'][rank]
        require(rows[rank].get('report_sha256')==row['report_sha256']
                and rows[rank].get('target_sha256')==row['target_sha256']
                and rows[rank].get('missing_trainable_gradients')==[],
                'other training terminal rank differs')
        rankdocs[str(rank)]={'report':row['report_sha256'],'target':row['target_sha256']}
    return {'train_create':create_sha,'train_terminal':terminal_sha,'train_audit':data['audit_sha256'],
            'rank_documents':rankdocs}


def load_single(train_runs,owner_sync,train_data):
    """Build metadata evidence. It never authorizes a coverage exception by itself."""
    import hashlib,json
    sync,sources=read_arm('sync',train_runs['sync'],owner_sync,train_data['sync'])
    other=verify_other_train('deferred',train_runs['deferred'],train_data['deferred'])
    for rank in range(4):
        left=train_data['sync']['rows'][rank];right=train_data['deferred']['rows'][rank]
        require(all(left['target'].get(k)==right['target'].get(k) for k in TARGET_KEYS),
                'cross-arm native1500 target differs')
        for row in (left,right):
            grads=row['report']['preclip']['gradients']
            require(set(grads)==set(row['target']['tensors']['model_local_parameters'])
                    and all(g.get('trainable') is True and g.get('present') is True for g in grads.values()),
                    'trainable gradient inventory differs')
    payload={'schema':'tp-owner-observation-binding-v1','source_manifest_sha256':SOURCE,
             'overlay_sha256':OVERLAY,'source_contract_sha256':CONTRACT,'config_sha256':CONFIG,
             'owner_arm':'sync','rank_records':{str(k):v for k,v in sync.items()},
             'sync_evidence':sources,'deferred_evidence':other,
             'historical_scope':'new restore-only observation; old training assignment not directly recorded'}
    raw=json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    return {'payload':payload,'binding_sha256':hashlib.sha256(raw).hexdigest()}


REVIEW_CONDITIONS=('config_only_requires_grad_sync','same_source_environment_and_physical_cards',
                   'runtime_pg_members_and_order','requires_grad_at_cache_construction',
                   'batch_cap_order_and_replica_count','cache_lifecycle_until_training_step',
                   'default_muon_update_and_no_unreviewed_swap')


def accept_review(binding,review,review_sha):
    """Root reviews conditional equivalence; supplied artifacts remain unmodified."""
    import hashlib,json
    require(isinstance(review,dict) and review.get('schema')=='tp-owner-crossrun-root-review-v1'
            and review.get('status')=='conditionally_verified'
            and review.get('binding_sha256')==binding['binding_sha256']
            and review.get('source_manifest_sha256')==SOURCE
            and review.get('source_dependency_audit_sha256')==hashlib.sha256(
                (Path(__file__).parent/'source-dependency-audit.md').read_bytes()).hexdigest()
            and review.get('owner_scope')=='subsequent_restore_only_not_historical_recording',
            'root conditional review absent or mismatched')
    checks=review.get('checks'); rationale=review.get('rationale')
    require(isinstance(checks,dict) and set(checks)==set(REVIEW_CONDITIONS)
            and all(checks[x] is True for x in REVIEW_CONDITIONS)
            and isinstance(rationale,dict) and set(rationale)==set(REVIEW_CONDITIONS)
            and all(isinstance(rationale[x],str) and rationale[x].strip() for x in REVIEW_CONDITIONS),
            'root review must state every checked condition and evidence')
    require(isinstance(review_sha,str) and re.fullmatch('[0-9a-f]{64}',review_sha),
            'review hash unavailable')
    return {'binding_sha256':binding['binding_sha256'],'review_sha256':review_sha,
            'rank_records':binding['payload']['rank_records'],
            'scope':'conditional equivalence inferred from subsequent restore and reviewed lifecycle'}
