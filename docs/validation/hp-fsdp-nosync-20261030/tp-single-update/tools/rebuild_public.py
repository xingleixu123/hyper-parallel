#!/usr/bin/env python3
"""Rebuild a public aggregate from separately held, SHA-pinned private evidence.

This tool reads local files only. It emits an explicit aggregate whitelist:
no parameters, optimizer-state samples, training samples, RNG bytes, YAML,
process identity, filesystem paths or raw per-rank reports are exported.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import zipfile

PINS={
 'comparison':'42d66cc870378959224362060532853b8e790b6164a94d5c60196790eccf4480',
 'source':'02cd4c3f8856e594057b104f3438978087aa373bab7e46059c7da7f13b001bf2',
 'delta':'76469ba4198dafcbbaeeba6372bb178be0af323575f7115ba7d784636b7032ff',
 'result':'c4fb027ba94bc894e2af595da22889ca855535014b026e727f38e7c95864f69e',
 'terminal':'66ac1fbc09017b050f1c0ac426a58acba8b3dfd340dbf4d22ef753a3aa8fbf4e',
 'semantic_review':'499a94f55ce6662e9eca253f619efe086c89bbd8e1df9e38ce6ac987e6eb22e2',
 'owner_preflight':'8a0341713c3679fce2f35ed529da0b7f5db6c4a6109a34faefabf431989cf614',
 'owner_crossrun_review':'9f3aa3af415ce0ca80c1f7c0efb950ba2fde3df3ee266123e886411f7b442f31',
 'source_dependency_audit':'f3a115b210975a24319ac1e5aad8103953884cdb00d68f9d0936d5f610bb29df',
 'owner_terminal':'6f7ddf93c139b9b85bca71221027909d35b23f44f7b8c49f3a9c4dc284645bd7',
 'owner_physical_review':'39963c946b71d2ca8fbd675423e1f32d33030a257da8d82a0ac82657c7305922',
 'owner_initial_review':'cfaef39eb53a1de4394599a310e03af40d600673843022f69dbfb658837c2d17',
 'failure_zip':'ef584916a696def8ce7f847252a061acb12b17f74d89d16cb9bf3d1a22dbed98',
 'scalar_diagnostic':'174ba213a7d9f6272987a3ca95d333c9ce545bc6fb0e2ca75cb34844bf350318',
 'independent_fullcpu_summary':'6bb90659fd80acc60eae4de96405c1aadf10c3bbaffc995eab9bc5b4b04891de',
}
SOURCE_MANIFEST='0f168e2ce3279cb315fc27d60d06693a17a5626a228fbf777146dda9da3e977d'
OWNER_BINDING='de7e8915ec56d2e48e14a174501c32bb41dfe925a0833ebb285e3cb241412eea'
OWNER_REVIEW='9f3aa3af415ce0ca80c1f7c0efb950ba2fde3df3ee266123e886411f7b442f31'
AUDIT_REFERENCES={
 'source-excerpts-1a.txt':'f6629ae29037aafbedd1a9db2cb65df7fe07e7aa03009bf195ee888ff5ee86be',
 'source-excerpts-verification.json':'7cf60c5d555bdd033aa57b4f5418b3e2d6aacfc2e50ba8a8ba823fa320a7cf10',
}
REVIEW_CONDITIONS=('config_only_requires_grad_sync','same_source_environment_and_physical_cards',
                   'runtime_pg_members_and_order','requires_grad_at_cache_construction',
                   'batch_cap_order_and_replica_count','cache_lifecycle_until_training_step',
                   'default_muon_update_and_no_unreviewed_swap')


def require(value,message):
    if not value:raise ValueError(message)


def read(path,pin):
    raw=Path(path).read_bytes()
    require(hashlib.sha256(raw).hexdigest()==PINS[pin],'input SHA mismatch: '+pin)
    return raw


def document(path,pin):return json.loads(read(path,pin))


def max_metric(rows,key):
    valid=[r for r in rows if r.get(key) is not None]
    require(valid,'metric empty')
    row=max(valid,key=lambda x:x[key])
    # Do not export a singleton's absolute delta; a relative L2 is a ratio.
    value=None if row['elements']<16 and key!='relative_l2' else row[key]
    return {'value':value,'elements_in_worst_local_shard':row['elements'],
            'fqn':row['name'] if row['elements']>=16 else None,
            'rank':row['rank'] if row['elements']>=16 else None}


def main():
    ap=argparse.ArgumentParser()
    for key in PINS:
        if key=='source_dependency_audit':
            ap.add_argument('--source-dependency-audit',type=Path,
                            default=Path(__file__).parent/'original_compare'/'source-dependency-audit.md')
        else:ap.add_argument('--'+key.replace('_','-'),type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args()
    comp=document(a.comparison,'comparison');source=document(a.source,'source')
    delta=document(a.delta,'delta');result=document(a.result,'result')
    terminal=document(a.terminal,'terminal');review=document(a.semantic_review,'semantic_review')
    owner=document(a.owner_preflight,'owner_preflight');owner_term=document(a.owner_terminal,'owner_terminal')
    crossrun=document(a.owner_crossrun_review,'owner_crossrun_review')
    audit=read(a.source_dependency_audit,'source_dependency_audit')
    packaged_audit=read(Path(__file__).parent/'original_compare'/'source-dependency-audit.md','source_dependency_audit')
    require(audit==packaged_audit,'packaged audit differs')
    for name,pin in AUDIT_REFERENCES.items():
        data=(Path(__file__).parent/'original_compare'/name).read_bytes()
        require(hashlib.sha256(data).hexdigest()==pin and name.encode() in audit,'audit reference differs')
    physical=document(a.owner_physical_review,'owner_physical_review')
    initial=document(a.owner_initial_review,'owner_initial_review')
    scalar=document(a.scalar_diagnostic,'scalar_diagnostic')
    independent=document(a.independent_fullcpu_summary,'independent_fullcpu_summary')
    read(a.failure_zip,'failure_zip')
    with zipfile.ZipFile(a.failure_zip) as z:
        names=z.namelist()
        failure=json.loads(z.read('terminal.json'))
        compare_failure=json.loads(z.read('compare-terminal.json'))
        stderr=z.read('compare.stderr')
        require(failure.get('exit_code')==1 and compare_failure.get('exit_code')==1
                and hashlib.sha256(stderr).hexdigest()=='28a75940a8e6e41da71ff69e1243f528839663e1be95adb19026df6a2fb03fa6'
                and b'replicated region bytes differ' in stderr
                and not any(n.startswith('native/') for n in names),'original failed attempt differs')
    require(terminal.get('exit_code')==0 and terminal.get('oom_killed') is False
            and terminal.get('result_sha256')==PINS['result'] and terminal.get('result')==result
            and result.get('status')=='completed_no_acceptance','CPU terminal differs')
    require(result.get('sha256')=={'comparison':PINS['comparison'],'source':PINS['source'],'delta':PINS['delta']}
            and result.get('owner_binding_sha256')==OWNER_BINDING
            and result.get('owner_review_sha256')==OWNER_REVIEW,'CPU result binding differs')
    require(review.get('status')=='reviewed_for_performance_and_full1000_experiment'
            and review.get('single_update_semantics_reviewed') is True
            and review.get('final_training_acceptance') is False
            and review.get('comparison_sha256')==PINS['comparison']
            and review.get('native_source_sha256')==PINS['source']
            and review.get('native_delta_sha256')==PINS['delta']
            and review.get('cpu_terminal_sha256')==PINS['terminal']
            and review.get('cpu_owner_binding_sha256')==OWNER_BINDING
            and review.get('cpu_owner_review_sha256')==OWNER_REVIEW,'root technical review differs')
    require(comp.get('status')=='compared' and comp.get('nonfinite_elements')==0
            and comp.get('unresolved_gradients')==[] and len(comp['role_comparisons'])==804
            and len(comp['coverage'])==201 and comp['owner_binding_sha256']==OWNER_BINDING
            and comp['owner_review_sha256']==OWNER_REVIEW,'one-step coverage differs')
    require(source.get('status')=='source_regions_read_complete' and source.get('target_verified') is False
            and source.get('source_manifest_sha256')==SOURCE_MANIFEST
            and (source.get('global_tensors'),source.get('optimizer_byte_entries'),source.get('roles'))==(144,680,258)
            and delta.get('status')=='compared' and delta.get('comparison_records')==1152
            and delta.get('nonfinite_elements')==0 and delta.get('source_validation_sha256')==PINS['source'],
            'native results differ')
    require(owner.get('status')=='needs_root_review' and owner['owner_binding']['binding_sha256']==OWNER_BINDING
            and owner_term.get('status')=='complete' and owner_term.get('child_exit_code')==0
            and len(owner_term.get('rank_outputs',[]))==4 and owner_term.get('source_files_verified_after')==2661
            and physical.get('approved') is True and physical.get('physical_mapping_verified') is True
            and physical.get('physical_by_rank')==[4,5,6,7] and initial.get('status')=='passed',
            'restore-only owner evidence differs')
    require(crossrun.get('schema')=='tp-owner-crossrun-root-review-v1'
            and crossrun.get('status')=='conditionally_verified'
            and crossrun.get('binding_sha256')==owner['owner_binding']['binding_sha256']==OWNER_BINDING
            and crossrun.get('source_manifest_sha256')==SOURCE_MANIFEST
            and crossrun.get('source_dependency_audit_sha256')==PINS['source_dependency_audit']
            and crossrun.get('owner_scope')=='subsequent_restore_only_not_historical_recording',
            'actual cross-run review is not bound to source audit and owner observation')
    checks=crossrun.get('checks');rationale=crossrun.get('rationale')
    require(isinstance(checks,dict) and set(checks)==set(REVIEW_CONDITIONS)
            and all(checks[x] is True for x in REVIEW_CONDITIONS)
            and isinstance(rationale,dict) and set(rationale)==set(REVIEW_CONDITIONS)
            and all(isinstance(rationale[x],str) and rationale[x].strip() for x in REVIEW_CONDITIONS),
            'cross-run review conditions incomplete')
    rr=owner['owner_binding']['payload']['rank_records']
    require(set(rr)=={'0','1','2','3'} and all(len(rr[str(r)])==28 for r in range(4)),
            'owner records incomplete')
    coverage={}
    for arm in ('sync','deferred'):
        coverage[arm]={}
        for kind in ('gradients','model','buffers','slots'):
            rows=[x for x in comp['coverage'] if x['kind']==kind]
            require(all(x[arm] is not None for x in rows),'coverage missing')
            replicate=sum(x[arm]['replicated_region_count'] for x in rows)
            owner_rows=[(x['name'],z) for x in rows for z in x[arm].get('regions',[])
                        if z.get('mode')=='conditional_owner_cache']
            require(all(y['post_matches_initial'] for _,z in owner_rows for y in z['ranks_observed']
                        if not y['restore_only_current_owned']),'nonowner initial mismatch')
            coverage[arm][kind]={'logical_fqns':len(rows),'replicated_regions':replicate,
                'strict_replicated_regions':replicate-len(owner_rows),
                'conditional_owner_regions':len(owner_rows),'conditional_owner_fqns':len({n for n,_ in owner_rows})}
    require(all(coverage[x]['slots']['conditional_owner_regions']==38
                and coverage[x]['slots']['conditional_owner_fqns']==19 for x in coverage),'owner replica coverage')
    # This uses complete comparison ordering and stored post hashes only to identify
    # the initial comparator failure; owner selection came from the later snapshot.
    disagreements=[(x['name'],arm) for x in comp['coverage'] for arm in ('sync','deferred')
                   for z in x[arm].get('regions',[]) if z.get('mode')=='conditional_owner_cache'
                   and len({r['post_sha256'] for r in z['ranks_observed']})>1]
    require(disagreements and disagreements[0]==('model.hc_head.hc_fn.momentum_buffer','sync'),
            'first conditional live-slot discrepancy differs')
    cross={}
    for kind in ('gradients','model','buffers','slots'):
        rows=[r for r in comp['role_comparisons'] if r['kind']==kind]
        require(all(r['nonfinite_sync']==0 and r['nonfinite_deferred']==0 for r in rows),'nonfinite role')
        cross[kind]={'rows':len(rows),'unique_fqns':len({r['name'] for r in rows}),
            'different_rows':sum(r['different_elements']>0 for r in rows),
            'empty_shards':sum(r['elements']==0 for r in rows),
            'max_absolute_difference':max_metric(rows,'max_abs'),
            'max_relative_l2':max_metric(rows,'relative_l2')}
    native={}
    for arm in ('sync','deferred'):
        native[arm]={}
        for kind in ('model','buffers','slots'):
            rows=[r for r in delta['per_role_rank_arm'] if r['arm']==arm and r['kind']==kind]
            require(all(r['nonfinite_initial']==0 and r['nonfinite_post']==0 for r in rows),'nonfinite native role')
            native[arm][kind]={'rows':len(rows),'empty_shards':sum(r['elements']==0 for r in rows),
                'unchanged_nonempty_rows':sum(r['all_values_unchanged'] is True for r in rows),
                'max_absolute_native_to_post_update':max_metric(rows,'max_abs_update')}
    require(scalar.get('status')=='diagnostic_only_no_acceptance'
            and (scalar.get('sha_constrained_initial_scalar_values'),scalar.get('fp32_trajectory_slot_matches'),
                 scalar.get('initial_equals_post_parameters'),scalar.get('conservative_bounds_below_half_ulp'))==(60,40,20,20),
            'scalar diagnostic incomplete')
    sv=independent['source_validation']
    require(independent['cpu_terminal_exit_code']==0 and sv['target_verified'] is False
            and sv['target_sha_and_post_sha_linked_rows']==1152
            and (sv['target_region_also_listed_in_generic_source_cache'],
                 sv['target_region_not_listed_in_generic_source_cache'])==(1104,48),
            'independent metadata association differs')
    clips=comp['clip_evidence']
    require(len(clips)==4 and len({(x['norm_abs_difference'],x['scale_abs_difference']) for x in clips})==1,
            'clipping aggregate differs')
    result_public={'schema':'tp-single-step-public-aggregate-v2',
       'status':'bounded_one_step_evidence_not_final_training_acceptance',
       'source':{'git_head':'1a39572fef23819a2d2651b4bd53047a05a4892f',
                 'source_manifest_sha256':SOURCE_MANIFEST,
                 'sole_scheduler_sha256':'51abc1d50805960b50c334c8f5a5b5c4a5b2a5e0d7984686bae3d2e553499fe2',
                 'verified_unchanged_dependency_git_blob_sha256':{
                   'optimizer.py':'9e24e38e602e929ef8cc3a214710602452aa99abb2f94980a26afa29d7bf63c6',
                   'muon.py':'27e81dc93bcbbc4c3a288977ff3cab81f8e27c4f537d466d41c2830566aa4f10',
                   'sharding_category.py':'d418c2dbdb18dbaef9197342aba40430dbd90e6a352d5c08277ca9f9cf0a32ae',
                   'muon_shard.py':'b33e6ce958e8633c148e0e3fc209f4aebafbf4f3786a0e91ffe8b92ce5c079c4'}},
       'provenance_sha256':dict(PINS),
       'source_audit_references_sha256':dict(AUDIT_REFERENCES),
       'additional_private_evidence_fingerprints':{
          'closed_full_cpu_archive':'24a782795f0c9ab4cfbe0f807dc8386febaa295b25e4beaf955949888ca8372f',
          'failed_compare_stderr':'28a75940a8e6e41da71ff69e1243f528839663e1be95adb19026df6a2fb03fa6',
          'scalar_native_supplement':'f3b8da000ef60dcccb8e86646ed9f5fbd49e90993b1c6d2d56a6e2537b9bd11b',
          'executed_native_delta_reader':'4ada4484d60b5fcdf7c90ed413cd62b04a315e2a5b30481c2ee2ccd077d009b1',
          'executed_native_delta':'9be93b198213bca5af0189464885992708bc1c2a0912921d746cd3f770ebdd41',
          'executed_owner_aware_compare':'31189017fe363bde5f4d1e220d4857fb6550aa4c396f538ef4b313ebd360dc14',
          'owner_evidence_dependency':'96b6c454f0ec5d2e1c1e00fa819bbefa8ddecaa216cfff40b54b5bc18b8d40fb',
          'native_delta_compare_dependency':'3d0cb68731d0b253a0c90a6885bf255b7b20e78e3be757dcb84dd17e262ab874',
          'source_regions_dependency':'f50803a1370b68308bcfdb7e883c09a55f085ab0f8c2125ba442ff964b67fd88'},
       'initial_attempt':{'exit_code':1,'phase':'comparison_coverage',
          'reason':'same-region live Muon momentum was assumed byte-identical',
          'first_fqn':'model.hc_head.hc_fn.momentum_buffer','native_phase_started':False},
       'owner_observation':{'restore_only_npu':True,'training_updates':0,'observed_ranks':4,
          'muon_fqns_per_rank':28,'physical_mapping_reviewed':True,'initial_258_roles_audited':True,
          'conditional_later_restore_not_historical_recording':True,'binding_sha256':OWNER_BINDING,
          'root_owner_review_sha256':OWNER_REVIEW},
       'completed_cpu':{'exit_code':0,'oom_killed':False,'raw_comparison_rows':804,
          'logical_coverage_roles':201,'global_native_tensors':144,'optimizer_bytes_entries':680,
          'audited_roles':258,'native_delta_records':1152,'missing_gradients':0,'nonfinite_elements':0,
          'generic_source_cache_target_verified':False,'target_slices_validated_by_separate_delta':1152,
          'target_slices_listed_in_generic_cache':1104,'target_slices_not_listed_in_generic_cache':48},
       'coverage_per_arm':coverage,'cross_arm_summary':cross,'native_update_summary':native,
       'clipping_absolute_differences':{'returned_norm':clips[0]['norm_abs_difference'],
                                         'scale':clips[0]['scale_abs_difference']},
       'scalar_diagnostic':scalar,
       'future_results':{'same_source_tp40_performance':'pending_unpopulated',
                         'deferred_full_1000_loss':'pending_unpopulated'},
       'limitations':['later owner cache only conditionally linked to earlier step',
                      'compared is an evidence status, not a final numerical acceptance',
                      'no independent full-model optimizer oracle, NPU kernel trace, long-run loss or performance',
                      'row counts include replicated local shards']}
    raw=(json.dumps(result_public,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    with a.out.open('xb') as f:f.write(raw)
    print(json.dumps({'status':result_public['status'],'output_sha256':hashlib.sha256(raw).hexdigest()},sort_keys=True))


if __name__=='__main__':main()
