"""Rebuild the release whitelist from pinned private reports, never emit private data files."""
import argparse
import ast
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import yaml

PINS = {
 'baseline_loss':'790f667ad93dfa2bfe572c5184523ef979755bff8ad32ca4613a963e4495ae81',
 'historical_tp_audit':'c6ed3880d05a80e3eb5da06de4a188207459a2373c1dd9a3101546a60108cfe2',
 'historical_tp_worker':'17935eeec9cc4b830191ff58e3e101a05adfc780fa70310e17a4d5d0eb206f0b',
 'historical_tp_measure':'8f50fa38f9a49616ec9e3fbaa3db2b1d94ea51857325cafe498454a390133acd',
 'tp_timed_loss':'494610cee4671714d8e3914c8e5f7cbcaf5cb82c08b15281be0a16027cd4ae99',
 'tp_continuation_loss':'bfbe662811d0a615fdff8a23efea56a2d086adabf0551e093becd2292a59efe2',
 'independent_audit':'6a26d915ddbd29bd4441830e98c253a20fb411b39f7acacd453786c432999b72',
 'root_comparison':'7ba3cbc18a763ec72d8fd0aa91876f2dcde768ea75a2280986b9480074acd372',
 'paired_summary':'363a723e4e0ae9990d1887faa50ce408bd791512e863db5a8abebc914c8715b3',
 'paired_steps':'0bc754888686ea65de24314026fa7bab603442cb92e144c40ba7c94dc4305404',
 'paired_memory':'318c5308cfd5f6788bf447b63ea350503474dc8416766b534b1b429e4fa9785e',
 'baseline_config':'1643450890283aa811b9243d4e6d9ffee5b4b56bdae517c7cc3030b88cd0a7ea',
 'tp_config':'acf412733e3bed399510709a6aeb9456bc7c3fd6c24ec4993e69adc1891a3481',
 'baseline_worker':'fc5ed096723c4da2c602280ea469d98d03273014a126be57628cab0e4131195c',
 'baseline_measure':'55c98860ae603272d7b2c6b707c062813d6ebb9445d540a8ec19d5d89affe431',
 'tp_worker':'dbc6b326742175a5af4b57febdc9fb4d3513b6360158be2d34fd3310d804d957',
 'tp_measure':'d00836c34de96174aea5dea42f15b89aa973b3b24059859e391da8d884d469d8',
 'tp_short_contract':'d69319a853497b3818dcc0e87e6498db70edaef51e565090e336793ed5d232c5',
 'tp_post_observer':'f8d34b706defb771f764082cfd4ba8813e1a0d95fed144b93a684d4f370e87cc',
 'scheduler_overlay':'51abc1d50805960b50c334c8f5a5b5c4a5b2a5e0d7984686bae3d2e553499fe2',
 'tp_source_manifest':'0f168e2ce3279cb315fc27d60d06693a17a5626a228fbf777146dda9da3e977d',
 'tp_terminal':'a3cdb7ee168cb7d7ea1828502e2b2e3d82d3ad6713a099ff53a8066858c604cb',
 'sync_terminal':'c8e8ffa95d1611a4ceacbd1193c06b1ee5794b0c96e562bcc9f1b369e0c68010',
}
HEAD='1a39572fef23819a2d2651b4bd53047a05a4892f'
BASE_SOURCE='b18fef99d3d55649c439d865718db90cfc1d0b1f6742bad1f7c87fd7d9a9a8cb'
GIT_FILES={
 'hyper_parallel/trainer/base.py':'9c13d21d1d009c9caf266be1e103ab9bd5f6db4ff1d6acd1a01aa684b0945cd1',
 'hyper_parallel/trainer/text_trainer.py':'5130e931e595fa1bb97edb5c608432ba8b56524dccc660d3b2cae7c731960f88',
 'hyper_parallel/trainer/runtime/metrics.py':'83202395a352a7d7c4d27ed5f346c59fa69a1700e29affeb73966b1f6b7d71fb',
}


def require(v,msg):
 if not v:raise ValueError(msg)
def sha(data):return hashlib.sha256(data).hexdigest()
def readpin(p,key):
 raw=p.read_bytes();require(sha(raw)==PINS[key],'input SHA mismatch: '+key);return raw

def jswrite(p,x):
 with p.open('x',encoding='utf-8',newline='\n') as f:json.dump(x,f,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
def csvwrite(p,fields,rows):
 with p.open('x',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='raise',lineterminator='\n');w.writeheader();w.writerows(rows)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--project-root',type=Path,required=True);ap.add_argument('--output-dir',type=Path,required=True);a=ap.parse_args();p=a.project_root;g=p/'grad-sync-fix-v1';out=a.output_dir
 require(out.is_dir() and not out.is_symlink(),'output directory must already exist')
 for name in ('loss-pairs-1000.csv','first40.csv','memory.csv','metrics.json','source-lineage.json'):
  require(not (out/name).exists(),'output already exists: '+name)
 base=p/'sac-strict-1000-adopt-v2/original-selective';runs=g/'tp-perf-acceptance-closed-v1';scripts=g/'tp-nosync-perf-acceptance-reviewed-v2/inputs/deferred';paired=g/'tp-perf-paired40-review-max-v1';independent=g/'tp-full1000-independent-audit-max-v1'
 paths={
  'baseline_loss':base/'run/rank-0-perf.jsonl',
  'historical_tp_audit':p/'tp2-paired-125-audit-v2/audit.json',
  'historical_tp_worker':p/'tp2-paired-125-v2/tp2-conservative-full/scripts/worker.py',
  'historical_tp_measure':p/'tp2-paired-125-v2/tp2-conservative-full/scripts/perf_measure.py',
  'tp_timed_loss':runs/'deferred/run/rank-0-perf.jsonl',
  'tp_continuation_loss':runs/'deferred/run/rank-0-continuation.jsonl','independent_audit':independent/'summary.json',
  'root_comparison':independent/'root-comparison.json','paired_summary':paired/'summary.json',
  'paired_steps':paired/'steps.csv','paired_memory':paired/'memory.csv','baseline_config':base/'scripts/config.yaml',
  'tp_config':scripts/'config.yaml','baseline_worker':base/'scripts/worker.py','baseline_measure':base/'scripts/perf_measure.py',
  'tp_worker':scripts/'worker.py','tp_measure':scripts/'perf_measure.py','tp_short_contract':scripts/'short_contract.py',
  'tp_post_observer':scripts/'post_training_observation.py','scheduler_overlay':g/'repo-patch/hyper_parallel/core/fully_shard/hsdp_scheduler.py',
  'tp_source_manifest':scripts/'actual-source-manifest.json','tp_terminal':runs/'deferred/run/execute-terminal.json',
  'sync_terminal':runs/'sync/run/execute-terminal.json'}
 raw={k:readpin(path,k) for k,path in paths.items()};audit=json.loads(raw['independent_audit']);review=json.loads(raw['root_comparison']);ps=json.loads(raw['paired_summary'])
 historical=json.loads(raw['historical_tp_audit']);historic_mean=historical['runs']['tp2-conservative-full']['mean_max_rank_seconds']
 require(historic_mean==2.019935879666667,'historical TP audit differs')
 def first40_loop(blob):
  tree=ast.parse(blob);run=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='run')
  loop=next(x for x in ast.walk(run) if isinstance(x,ast.For) and isinstance(x.target,ast.Name) and x.target.id=='step')
  return ast.dump(ast.Module(body=loop.body[:3],type_ignores=[]),include_attributes=False)
 require(first40_loop(raw['historical_tp_measure'])==first40_loop(raw['tp_measure'])
         and PINS['historical_tp_worker']!=PINS['tp_worker'] and PINS['historical_tp_measure']!=PINS['tp_measure'],
         'historical code relationship differs')
 require(audit['status']=='descriptive_mean_bound_checked_not_full_model_oracle' and audit['absolute_signed_mean_bound']==0.005
         and review['status']=='exact_1000_points_and_aggregate_match' and review['points_matched']==1000
         and ps['status']=='first40_descriptive_not_final_acceptance','input result scope')
 baseline=[json.loads(x) for x in raw['baseline_loss'].splitlines()];timed=[json.loads(x) for x in raw['tp_timed_loss'].splitlines()];tail=[json.loads(x) for x in raw['tp_continuation_loss'].splitlines()];candidate=timed+tail
 require(len(baseline)==len(candidate)==1000 and len(timed)==40 and len(tail)==960,'loss coverage')
 require(all(x['rank']==0 and x['step']==1501+i for i,x in enumerate(baseline))
         and all(x['rank']==0 and x['step']==1501+i for i,x in enumerate(candidate)),'loss rank/step')
 deltas=[];pairs=[]
 for x,y in zip(baseline,candidate):
  require(math.isfinite(x['loss']) and math.isfinite(y['loss']),'nonfinite loss')
  delta=y['loss']-x['loss'];deltas.append(delta)
  pairs.append({'step':x['step'],'reference_loss':x['loss'],'tp_deferred_loss':y['loss'],
                'delta':delta,'cumulative_signed_mean':math.fsum(deltas)/len(deltas)})
 observed={'n':1000,'signed_mean_delta':math.fsum(deltas)/1000,'absolute_signed_mean_delta':abs(math.fsum(deltas)/1000),
           'mean_absolute_error':math.fsum(abs(x) for x in deltas)/1000,'positive':sum(x>0 for x in deltas),
           'negative':sum(x<0 for x in deltas),'zero':sum(x==0 for x in deltas),
           'max_absolute_delta':max(map(abs,deltas)),
           'max_absolute_delta_step':1501+max(range(1000),key=lambda i:abs(deltas[i]))}
 require(observed==audit['requested_metric'] and audit['absolute_signed_mean_within_bound'] is True,
         'independent loss aggregate differs')
 term=json.loads(raw['tp_terminal']);sterm=json.loads(raw['sync_terminal'])
 require(term['status']=='complete' and term['child_exit_code']==0 and term['source_files_verified_after']==2661
         and sterm['status']=='complete' and sterm['child_exit_code']==0,'terminal mismatch')
 r0=next(x for x in term['rank_outputs'] if x['rank']==0)
 require(r0['perf_sha256']==PINS['tp_timed_loss'] and r0['continuation_sha256']==PINS['tp_continuation_loss'],
         'terminal does not bind rank0 files')
 bcfg=yaml.safe_load(raw['baseline_config']);tcfg=yaml.safe_load(raw['tp_config'])
 for cfg,mode in ((bcfg,'selective'),(tcfg,'full')):
  m=cfg['model']['config'];require(m['num_hidden_layers']==3 and m['first_k_dense_replace']==2
    and m['num_nextn_predict_layers']==0 and m['mtp_loss_factor']==0 and cfg['training']['train_iters']==2500
    and cfg['training']['global_batch_size']==4 and cfg['training']['micro_batch_size']==1
    and cfg['dataset']['sequence_length']==4096 and cfg['activation_checkpoint']['mode']==mode
    and cfg['accelerator']['sequence_parallel'] is False,'model or training contract')
 require(all(bcfg[k]==tcfg[k] for k in ('model','optimizer','training','lr_scheduler','dataset','model_init_dtype'))
         and bcfg['fsdp_config']['mix_precision']==tcfg['fsdp_config']['mix_precision']
         and tcfg['fsdp_config']['requires_grad_sync'] is False,'cross-config invariant')
 lines={k:v.decode('utf-8-sig').splitlines() for k,v in raw.items() if k in
        ('baseline_worker','baseline_measure','tp_worker','tp_measure','tp_short_contract','tp_post_observer','scheduler_overlay')}
 for k,n,needle in [('baseline_measure',104,'synchronize'),('baseline_measure',105,'reset_peak_memory_stats'),
                    ('baseline_measure',108,'trainer.train_step'),('baseline_measure',118,"result['loss']"),
                    ('baseline_worker',127,'mtp_loss_factor'),('tp_measure',102,'for step in range'),
                    ('tp_measure',113,'synchronize'),('tp_measure',114,'reset_peak_memory_stats'),
                    ('tp_measure',117,'trainer.train_step'),('tp_measure',127,"result['loss']"),
                    ('tp_measure',140,'trainer.train_step'),('tp_measure',144,"result['loss']"),
                    ('tp_measure',154,"scope['reads']"),('tp_worker',105,'mtp_loss_factor'),
                    ('tp_worker',175,'prepared_four_fields_checked'),('scheduler_overlay',236,'set_requires_grad_sync'),
                    ('scheduler_overlay',251,'unsharded_accumulated_grad'),
                    ('tp_short_contract',14,'def canonical_subset'),('tp_short_contract',34,'def slots'),
                    ('tp_short_contract',49,'def expected_loader'),('tp_short_contract',87,'def validate_records'),
                    ('tp_short_contract',116,'def validate_group_progress'),
                    ('tp_post_observer',20,'def examine'),('tp_post_observer',39,'def observe')]:
  require(needle in lines[k][n-1],'source line anchor differs: '+k+':'+str(n))
 manifest=json.loads(raw['tp_source_manifest']);entries={v['relative_path']:v for v in manifest['files']}
 require(manifest['base_head']==HEAD and manifest['dirty_files']==['hyper_parallel/core/fully_shard/hsdp_scheduler.py']
         and len(entries)==2661 and entries[manifest['dirty_files'][0]]['sha256']==PINS['scheduler_overlay'],'source manifest')
 git={}
 for name,expected in GIT_FILES.items():
  blob=subprocess.check_output(['git','-C',str(p/'source-sac-npu-policy'),'show',HEAD+':'+name]);require(sha(blob)==expected==entries[name]['sha256'] and len(blob)==entries[name]['size'],'Git blob differs')
  git[name]={'sha256':expected,'git_commit':HEAD}
 git['hyper_parallel/trainer/text_trainer.py']['original_lines']=[249,275,280,301]
 git['hyper_parallel/trainer/base.py']['original_lines']=[750,754,760]
 git['hyper_parallel/trainer/runtime/metrics.py']['original_lines']=[35,40,45,46,77,78,127,133,145]
 steps=list(csv.DictReader(io.StringIO(raw['paired_steps'].decode('utf-8'))));mem=list(csv.DictReader(io.StringIO(raw['paired_memory'].decode('utf-8'))))
 require(len(steps)==40 and len(mem)==4,'performance rows')
 perfrows=[]
 for i,row in enumerate(steps):
  require(int(row['step'])==1501+i and row['phase']==('warmup' if i<10 else 'measured'),'performance step')
  s=float(row['sync_max_rank_s']);d=float(row['deferred_max_rank_s'])
  require(math.isfinite(s) and math.isfinite(d) and s>0 and d>0,'invalid duration')
  perfrows.append({'step':1501+i,'phase':row['phase'],'sync_max_seconds':s,'deferred_max_seconds':d,
                   'deferred_minus_sync_seconds':d-s,'deferred_over_sync':d/s})
 require(math.isclose(math.fsum(r['sync_max_seconds'] for r in perfrows[10:])/30,ps['arms']['sync']['measured']['mean_seconds'],abs_tol=1e-15)
         and math.isclose(math.fsum(r['deferred_max_seconds'] for r in perfrows[10:])/30,ps['arms']['deferred']['measured']['mean_seconds'],abs_tol=1e-15),'performance mean mismatch')
 memrows=[]
 for i,row in enumerate(mem):
  require(int(row['rank'])==i and int(row['physical_device'])==4+i,'memory rank')
  selected={'rank':i,'physical_device':4+i}
  for k in ('measured_peak_allocated_bytes','measured_peak_reserved_bytes','warm_unique_npu_storage_bytes'):
   s=int(row['sync_'+k]);d=int(row['deferred_'+k]);require(d-s==int(row['difference_'+k]) and d/s==float(row['ratio_'+k]),'memory ratio mismatch')
   selected['sync_'+k]=s;selected['deferred_'+k]=d;selected['difference_'+k]=d-s;selected['ratio_'+k]=d/s
  memrows.append(selected)
 sample_scope={'all_dataset_indices_observed':4000,'raw_owner_rank_counts':[2000,0,2000,0],
   'first_two_steps_prepared_four_field_rank_micro_records':16,'later_prepared_four_field_per_sample_hashes':'not_recorded',
   'post_roles_per_rank':258,'post_finite_observed_nonfinite_count':0}
 require(audit['candidate']['canonical_actual_read_count']==4000 and audit['candidate']['first_two_step_prepared_records']==16
         and all(x['recorded_roles']==258 and x['all_finite_nonfinite_count']==0 for x in audit['candidate']['post_rank_summaries']),
         'audit sample/role scope')
 metrics={'schema':'tp-perf-final-public-metrics-v1','status':'descriptive_data_plus_requested_mean_bound_only',
   'loss_1000':{**observed,'threshold_on_absolute_signed_mean':0.005,'within_mean_threshold':True},
   'performance_30_steps':{'sync':ps['arms']['sync']['measured'],'deferred':ps['arms']['deferred']['measured'],
                           'paired':ps['paired_measured']},'per_rank_memory':memrows,
   'background_observations':ps['background_observations'],
   'historical_tp_sync':{'mean_max_rank_seconds':historic_mean,
      'source_manifest_sha256':BASE_SOURCE,'worker_sha256':PINS['historical_tp_worker'],
      'measurement_sha256':PINS['historical_tp_measure'],'first40_loop_first_three_statements_ast_equal':True,
      'scope':'approximate historical context only; different worker and measurement bytes, not same-source repeat'},
   'related_sac_fixed_near_step_1804':audit['related_sac_fixed_selective']['near_peak'],
   'related_sac_fixed_minus_original_max':{'absolute':audit['related_sac_fixed_selective']['fixed_minus_original']['max_absolute_delta'],
                                           'step':audit['related_sac_fixed_selective']['fixed_minus_original']['max_absolute_delta_step']},
   'sample_and_post_scope':sample_scope,'mtp':{'status':'not_applicable','reason':'MTP disabled; no MTP objective was evaluated'},
   'repeat40':{'status':'not_included_in_this_bundle','future_placement':'separate versioned sibling package'},
   'limitations':['single sequential timing pair; host background and run order are not controlled',
      'TP/full-recompute/deferred versus original FSDP/selective is a combined training comparison',
      '0.005 is a signed-mean bound only, not a per-step, MAE or full-weight bound',
      'no cross-run full-weight oracle or cross-mesh RNG equivalence is asserted']}
 lineage={'schema':'tp-perf-final-source-lineage-v1','training_head':HEAD,'baseline_source_manifest_sha256':BASE_SOURCE,
   'candidate_source_manifest_sha256':PINS['tp_source_manifest'],'candidate_sole_overlay_sha256':PINS['scheduler_overlay'],
   'fixed_git_files':git,'measurement_files':{
      'baseline_perf_measure.py':{'sha256':PINS['baseline_measure'],'original_lines':[104,105,108,109,118,134]},
      'candidate_perf_measure.py':{'sha256':PINS['tp_measure'],'original_lines':[102,112,113,114,116,117,118,119,125,127,140,144,151,154]},
      'baseline_worker.py':{'sha256':PINS['baseline_worker'],'original_lines':[127]},
      'candidate_worker.py':{'sha256':PINS['tp_worker'],'original_lines':[70,73,105,153,175,178]},
      'candidate_short_contract.py':{'sha256':PINS['tp_short_contract'],'original_lines':[14,18,34,42,49,87,89,99,116]},
      'candidate_post_training_observation.py':{'sha256':PINS['tp_post_observer'],'original_lines':[20,21,29,34,37,39,41,45,55,61]},
      'candidate_hsdp_scheduler.py':{'sha256':PINS['scheduler_overlay'],'original_lines':[236,249,250,251,253,254]}},
   'published_loss_csv_one_based_line_for_step':'step - 1499 (header is line 1)',
   'rank0_raw_lineage':{
      'reference_perf_jsonl':{'sha256':PINS['baseline_loss'],'rows':1000,'one_based_line_for_step':'step - 1500'},
      'candidate_perf_jsonl':{'sha256':PINS['tp_timed_loss'],'rows':40,'steps':[1501,1540],'one_based_line_for_step':'step - 1500'},
      'candidate_continuation_jsonl':{'sha256':PINS['tp_continuation_loss'],'rows':960,'steps':[1541,2500],'one_based_line_for_step':'step - 1540'}},
   'historical_tp_reference_sha256':{k:PINS[k] for k in ('historical_tp_audit','historical_tp_worker','historical_tp_measure')},
   'related_sac_fixed_rank0_perf_sha256':audit['related_sac_fixed_selective']['rank0_perf_sha256'],
   'related_sac_fixed_source_manifest_sha256':audit['related_sac_fixed_selective']['source_manifest_sha256'],
   'background_evidence_sha256':{
      'sync_audit_review':next(v for k,v in ps['input_sha256'].items() if 'active-sync' in k and k.endswith('/audit-review.json')),
      'pre_deferred_card0':next(v for k,v in ps['input_sha256'].items() if k.endswith('/background-before-timing.json')),
      'pre_deferred_cards0_3':next(v for k,v in ps['input_sha256'].items() if k.endswith('/background-before-timing-v2.json'))},
   'derived_fields':{'delta':'candidate rank0 loss - reference rank0 loss',
      'cumulative_signed_mean':'math.fsum of delta for steps 1501 through current step divided by (step - 1500)',
      'timing':'max of four rank-local synchronized step durations; not a global start-to-finish span'},
   'evidence_sha256':{k:PINS[k] for k in ('independent_audit','root_comparison','paired_summary','paired_steps',
                                          'paired_memory','baseline_config','tp_config','tp_terminal','sync_terminal')},
   'private_inputs_not_packaged':'original rank files, configs, role/state records and archive bodies remain private'}
 csvwrite(out/'loss-pairs-1000.csv',list(pairs[0]),pairs)
 csvwrite(out/'first40.csv',list(perfrows[0]),perfrows)
 csvwrite(out/'memory.csv',list(memrows[0]),memrows)
 jswrite(out/'metrics.json',metrics);jswrite(out/'source-lineage.json',lineage)
 print(json.dumps({'status':metrics['status'],'loss_rows':len(pairs),'first40_rows':len(perfrows),'memory_rows':len(memrows)},sort_keys=True))


if __name__=='__main__':main()
