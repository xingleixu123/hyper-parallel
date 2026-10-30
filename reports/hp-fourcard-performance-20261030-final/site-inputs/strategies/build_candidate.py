#!/usr/bin/env python3
"""Prepare grouped website input files from sealed real per-rank data, no website access."""
import csv
import hashlib
import io
import json
import math
import statistics
from pathlib import Path

HERE=Path(__file__).resolve().parent
G=HERE.parent
P=G.parent
PUBLIC=P/'performance-final-public-v3/public/data'
PAIR=P/'replica-count-fix-v1/fullmodel-pair-125-v4-summary/step-rank-data.csv'
PAIR_RESULT=P/'replica-count-fix-v1/fullmodel-pair-125-v4-result.json'
SOURCE_EXPECTED={'perf_steps':'794a740c20c38c62092a46b6d529c84f81de72060471a48a86686e76e2ae747b','perf_summary':'562cddd396cf33d5a15ba27c778d500a38bf5f5d39c8e319a6ae7dea6c2df276','replica_steps':'7057f1f5e0e2f220746b4366f2004b99e8a206257c557c0d63883012b2a5b0c2','replica_result':'e2b5a871ca22bc23831a13cc4085f4745fd415133d2731db397575f3a1e620ff','matrix_report':'025144d2f28ba64757adc1d4ac8cb3e9eed8b3fbb90154c44db0446c6401713e'}
def sha(b):return hashlib.sha256(b).hexdigest()
def read_expected(p,key):
 b=p.read_bytes();assert sha(b)==SOURCE_EXPECTED[key],key;return b
def put(name,data):
 p=HERE/name;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as f:f.write(data)
 assert p.read_bytes()==data

def read_csv_with_raw_lines(raw):
 lines=raw.splitlines(keepends=True);assert lines and all(line.endswith((b'\n',b'\r')) for line in lines[:-1]);header=next(csv.reader([lines[0].decode('utf-8-sig')]))
 out=[]
 for number,line in enumerate(lines[1:],2):
  parsed=next(csv.reader([line.decode('utf-8')]))
  assert len(parsed)==len(header),'CSV multiline or unexpected columns'
  out.append((number,line,dict(zip(header,parsed))))
 return out

perf_csv=read_expected(PUBLIC/'perf-steps.csv','perf_steps');perf_doc=json.loads(read_expected(PUBLIC/'perf-summary.json','perf_summary'));pair_csv=read_expected(PAIR,'replica_steps');pair_doc=json.loads(read_expected(PAIR_RESULT,'replica_result'));read_expected(G/'final-completion-matrix-review-max-v1.md','matrix_report')
perf_raw=read_csv_with_raw_lines(perf_csv);pair_raw=read_csv_with_raw_lines(pair_csv);assert len(perf_raw)==420 and len(pair_raw)==160 and len(perf_doc['runs'])==14 and len(perf_doc['memory_by_rank'])==56 and pair_doc['status']=='complete'
perf_meta={x['run']:x for x in perf_doc['runs']};perf_rows={}
for n,line,row in perf_raw:
 key=(row['run'],int(row['step']));assert key not in perf_rows
 rank_times=[float(row['rank%d_seconds'%r]) for r in range(4)];mx=max(rank_times);assert math.isfinite(mx) and mx>0 and abs(mx-float(row['max_rank_seconds']))<1e-12
 perf_rows[key]={'seconds':float(row['max_rank_seconds']),'source_line_numbers':[n],'source_line_sha256':[sha(line)],'rank_values':rank_times}
pair_rows={}
for n,line,row in pair_raw:
 step=int(row['step']);rank=int(row['rank']);key=(step,rank);assert key not in pair_rows and rank in range(4)
 pair_rows[key]=(n,line,row)
assert len(pair_rows)==160

# Shared keys are used only for the explicitly audited pairs; non-concurrent references
# and independent recompute runs get separate keys even when hosted together.
GROUPS=[
 {'id':'g01_raf_reverse_125','kind':'audited_sequential_pair','description':'125 1a/full, reverse-order no-RAF vs full-RAF; same window, one pair','series':[('no_raf','raf-no',None),('full_raf','raf-full',None)]},
 {'id':'g02a_hsdp_fsdp_129_pair1','kind':'audited_sequential_pair','description':'129 1a/full HSDP then FSDP, pair 1','series':[('hsdp','hsdp-1',None),('fsdp','fsdp-1',None)]},
 {'id':'g02b_hsdp_fsdp_129_pair2','kind':'audited_sequential_pair','description':'129 1a/full HSDP then FSDP, pair 2; distinct pair key','series':[('hsdp','hsdp-2',None),('fsdp','fsdp-2',None)]},
 {'id':'g03_tp_fsdp_125','kind':'audited_sequential_pair','description':'125 1a/full conservative TP2 vs FSDP4; one pair','series':[('tp2','tp2',None),('fsdp4','tp-fsdp',None)]},
 {'id':'g04_cp_fixed_129','kind':'nonconcurrent_references','description':'129 03ea/selective CP2 and FSDP4: nonconcurrent reference, separate metric names','series':[('cp2','cp2-fixed','cp2'),('fsdp4_reference','fsdp-fixed','fsdp4_reference')]},
 {'id':'g05_ep2_129','kind':'nonconcurrent_references','description':'129 1a/full EP2 retest with both historical FSDP4 references; separate metric names','series':[('ep2_retest','ep2-2','ep2_retest'),('fsdp_reference1','fsdp-1','fsdp_reference1'),('fsdp_reference2','fsdp-2','fsdp_reference2')]},
 {'id':'g06_recompute_125','kind':'separate_policy_runs','description':'125 1a original-policy full/off/selective; independent runs with separate metric names','series':[('full','recompute-full','full'),('off','recompute-off','off'),('selective','recompute-selective','selective')]},
 {'id':'g07_hsdp_replica_125','kind':'audited_sequential_pair','description':'125 1a+Muon-a3 HSDP default vs explicit1, same-source paired 40 steps','series':[('default','default',None),('explicit1','explicit1',None)]},
]
source_paths={'perf_steps':'../../performance-final-public-v3/public/data/perf-steps.csv','perf_summary':'../../performance-final-public-v3/public/data/perf-summary.json','replica_steps':'../../replica-count-fix-v1/fullmodel-pair-125-v4-summary/step-rank-data.csv','replica_result':'../../replica-count-fix-v1/fullmodel-pair-125-v4-result.json','matrix_report':'../final-completion-matrix-review-max-v1.md'}
provenance=[];memory=[];groupdocs=[];statdocs=[];all_keys={}
for group in GROUPS:
 entries=[]
 for alias,run,suffix in group['series']:
  stem=group['id']+(('_'+suffix) if suffix else '')
  seconds_key=stem+'_max_rank_step_seconds';throughput_key=stem+'_nominal_tokens_per_second'
  for key in (seconds_key,throughput_key):
   prior=all_keys.setdefault(key,group['id']);assert prior==group['id'],'metric key reused across groups'
  outname='runs/'+group['id']+'/'+alias+'.jsonl';data=[];times=[];idx=[]
  for step in range(1511,1541):
   if group['id']=='g07_hsdp_replica_125':
    gathered=[pair_rows[(step,rank)] for rank in range(4)];col='default_seconds' if run=='default' else 'explicit1_seconds';vals=[float(r[col]) for _,_,r in gathered];assert all(r['phase']=='measured' for _,_,r in gathered)
    secs=max(vals);entry=next(x for x in pair_doc[run]['curve'] if x['step']==step);assert entry['phase']=='measured' and entry['per_rank_seconds']==vals and entry['max_rank_seconds']==secs
    source_name='replica_steps';numbers=[v[0] for v in gathered];hashes=[sha(v[1]) for v in gathered]
   else:
    x=perf_rows[(run,step)];secs=x['seconds'];source_name='perf_steps';numbers=x['source_line_numbers'];hashes=x['source_line_sha256']
   assert math.isfinite(secs) and secs>0;rate=16384/secs;assert math.isfinite(rate) and rate>0
   data.append((json.dumps({'step':step,seconds_key:secs,throughput_key:rate},ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n').encode());times.append(secs)
   provenance.append({'group':group['id'],'series':alias,'output_path':outname,'output_line':step-1510,'step':step,'source_file':source_paths[source_name],'source_file_sha256':SOURCE_EXPECTED[source_name],'source_line_numbers':'|'.join(map(str,numbers)),'source_line_sha256s':'|'.join(hashes)})
  content=b''.join(data);put(outname,content)
  mean=statistics.mean(times);expected=(pair_doc[run]['stats']['mean_seconds'] if group['id']=='g07_hsdp_replica_125' else perf_meta[run]['mean_max_rank_seconds']);assert abs(mean-expected)<1e-12
  if group['id']=='g07_hsdp_replica_125':
   meta=pair_doc[run]; run_id=meta['run_id']; pins=meta['pins'];head='1a39572fef23819a2d2651b4bd53047a05a4892f';source=pins['source_manifest_sha256'];host='125';conf=pins['config_sha256'];mm=meta['rank_memory'];mem_file='replica_result'
   for ri,v in enumerate(mm):
    t=v['timed_memory_after_max_bytes'];physical=v['physical_card'];memory.append({'group':group['id'],'series':alias,'source_run':run,'host':host,'physical_card':physical,'peak_allocated_gib':t['peak_allocated_bytes']/2**30,'peak_reserved_gib':t['peak_reserved_bytes']/2**30,'warm_unique_storage_gib':v['warm_storage_unique_npu_bytes']/2**30,'peak_allocated_bytes':t['peak_allocated_bytes'],'peak_reserved_bytes':t['peak_reserved_bytes'],'warm_unique_storage_bytes':v['warm_storage_unique_npu_bytes'],'source_file':source_paths[mem_file],'source_sha256':SOURCE_EXPECTED[mem_file],'source_json_pointer':'/'+run+'/rank_memory/'+str(ri),'scope':'static per-card timed maxima; warm snapshot, not step-level memory'})
  else:
   meta=perf_meta[run];run_id=meta['run_id'];head=meta['source_head'];source=meta['source_manifest_sha256'];host=meta['host'];conf=meta['config_sha256']
   for ri,v in enumerate(perf_doc['memory_by_rank']):
    if v['run']!=run:continue
    memory.append({'group':group['id'],'series':alias,'source_run':run,'host':host,'physical_card':v['physical_card'],'peak_allocated_gib':v['peak_allocated_gib'],'peak_reserved_gib':v['peak_reserved_gib'],'warm_unique_storage_gib':v['warm_deduplicated_gib'],'peak_allocated_bytes':'','peak_reserved_bytes':'','warm_unique_storage_bytes':'','source_file':source_paths['perf_summary'],'source_sha256':SOURCE_EXPECTED['perf_summary'],'source_json_pointer':'/memory_by_rank/'+str(ri),'scope':'static per-card timed maxima; warm snapshot, not step-level memory'})
  entry={'alias':alias,'source_run':run,'run_id':run_id,'jsonl':outname,'jsonl_sha256':sha(content),'metric_seconds':seconds_key,'metric_nominal_tokens_per_second':throughput_key,'source_head':head,'source_manifest_sha256':source,'config_sha256':conf,'host':host,'steps':[1511,1540],'points':30}
  if group['id']=='g07_hsdp_replica_125':
   entry.update({'source_baseline_manifest_sha256':pins['baseline_manifest_sha256'],'source_overlay_sha256':pins['overlay_sha256'],'observation_scope':'See paired result and original receipts; no inference of continuous exclusive use.'})
  else:
   entry.update({'activation_checkpointing':meta['ac'],'observation_scope':meta['wave'],'env_fingerprint_sha256':meta['env_fingerprint_sha256']})
  entries.append(entry)
  statdocs.append({'group':group['id'],'series':alias,'source_run':run,'run_id':run_id,'n':30,'mean_max_rank_seconds':mean,'median_max_rank_seconds':statistics.median(times),'population_std_seconds':statistics.pstdev(times),'cv':statistics.pstdev(times)/mean,'min_seconds':min(times),'max_seconds':max(times),'nominal_tokens_per_second_from_mean_duration':16384/mean,'arithmetic_mean_of_step_nominal_tokens_per_second':statistics.mean(16384/x for x in times),'source_summary_expected_mean_seconds':expected})
 assert len({x['jsonl'] for x in entries})==len(entries)
 if group['kind']=='audited_sequential_pair':assert len({x['metric_seconds'] for x in entries})==len({x['metric_nominal_tokens_per_second'] for x in entries})==1
 else:assert len({x['metric_seconds'] for x in entries})==len(entries)
 groupdocs.append({'group':group['id'],'kind':group['kind'],'description':group['description'],'series':entries,'auto_difference_safe_scope':'Within this one paired group only; an observed paired difference is not a universal causal estimate.' if group['kind']=='audited_sequential_pair' else 'Distinct metric keys; no automatic cross-file difference.'})
assert len(groupdocs)==8 and sum(len(x['series']) for x in groupdocs)==18 and len(provenance)==540 and len(memory)==72

def csv_bytes(rows):
 s=io.StringIO(newline='');w=csv.DictWriter(s,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows);return s.getvalue().encode()
put('source-index.csv',csv_bytes(provenance));put('memory-peaks.csv',csv_bytes(memory))
put('groups.json',(json.dumps({'schema':'grouped-website-inputs-v1','groups':groupdocs},sort_keys=True,ensure_ascii=False,indent=2)+'\n').encode())
put('summary.json',(json.dumps({'schema':'grouped-website-input-summary-v1','window':[1511,1540],'steps_per_run':30,'runs':statdocs,'throughput_caveat':'Per-step 16384/time values differ from 16384/mean(time); the latter matches the archived official window summary.','time_scope':'maximum of four rank-local synchronized durations; not a global makespan','memory_scope':'static per-card aggregates only'},sort_keys=True,ensure_ascii=False,indent=2)+'\n').encode())
README='''# 全策略网站输入候选（本地，未绘图）\n\n本包只整理已经实际训练的 1511–1540 共30个 step，**不连接网站、不绘图、不发布**。18 份 JSONL 各30行，字段为真实 step、四 rank 各自同步计时的最大值及 `16384/该步时长` 的名义 tokens/s；不平滑、不删除点、不将最大 rank 时长改称全局 makespan，也不将名义 tokens 当作有效 mask tokens。\n\n8组配置见 [groups.json](groups.json)：125 noRAF/full反向配对；129 HSDP/FSDP两对分别隔离；125旧TP/FSDP配对；129 CP fixed 与FSDP fixed非同期参考；129 EP2复测及两个非同期 FSDP参考；125 原1a full/off/selective；125 修复后 HSDP default/explicit1配对。配对组内才共享指标名；CP、EP参考和独立重算配置均各用不同指标名，避免网站自动计算跨条件差值。不同组的指标名全部不同；不能将它们汇总为无条件速度排名。各run为离散监测或监测不完整等实际状态，不证明计时窗口连续独占。EP的两个FSDP参考有意从已公布的两次记录重复引用，并没有新实验。\n\n[summary.json](summary.json) 给每组每run的mean/median/CV、极值和两种吞吐统计：官网报表对应 `16384 / mean(seconds)`，不等于对逐步 `16384 / seconds` 求均值。[source-index.csv](source-index.csv) 将每个输出点绑定到原 CSV 文件 SHA、精确原行号及每行原字节 SHA；HSDP 每个点绑定该 step 的四个 rank 行，并逐点与已封存 max 曲线核对。\n\n[memory-peaks.csv](memory-peaks.csv) 是每个run每张物理卡的**静态** timed peak allocated、peak reserved 与 warm 唯一 storage，保留原始汇总的单位和JSON指针；仅部分来源有原始 bytes。它不是逐步内存曲线，缺失 bytes 不得反算冒充原始精度。reserved 不是物理HBM，peak差或reserved−allocated不能当作 activation、碎片或泄漏。\n\n每组真实 source、host、配置指纹、run ID、metric 及适用限制均在 groups.json；文件完整哈希见 [SHA256SUMS](SHA256SUMS)。在新空同级目录复制 `build_candidate.py` 并以 `python -B build_candidate.py` 可从任务内固定原始表重建；parser-check.json 仅为离线 helper 解析验证，不是网站解析或可交互页面验收。\n'''
put('README.md',README.encode())
print(json.dumps({'status':'built_pending_offline_parser','groups':len(groupdocs),'jsonl_files':18,'step_points':len(provenance),'static_memory_rows':len(memory)},ensure_ascii=False))