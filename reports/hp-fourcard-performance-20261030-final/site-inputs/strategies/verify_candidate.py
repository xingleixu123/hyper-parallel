#!/usr/bin/env python3
"""Independently verify each grouped point against its pinned original byte lines."""
import csv, hashlib, json, math, statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sha = lambda b: hashlib.sha256(b).hexdigest()
loads = lambda p: json.loads(p.read_text(encoding='utf-8'))
groups = loads(HERE/'groups.json')['groups']
summaries = loads(HERE/'summary.json')['runs']
entries = {(g['group'], s['alias']): (g, s) for g in groups for s in g['series']}
assert len(groups) == 8 and len(entries) == 18
keys = defaultdict(set)
values = {}
for key, (group, item) in entries.items():
    file = HERE/item['jsonl']
    raw = file.read_bytes()
    assert sha(raw) == item['jsonl_sha256'] and raw.endswith(b'\n')
    lines = [json.loads(x) for x in raw.splitlines()]
    assert len(lines) == 30 and [x['step'] for x in lines] == list(range(1511,1541))
    metric_keys = {item['metric_seconds'],item['metric_nominal_tokens_per_second']}
    assert len(metric_keys) == 2
    for k in metric_keys: keys[k].add(group['group'])
    for line in lines:
        assert set(line) == metric_keys | {'step'}
        secs=line[item['metric_seconds']]; rate=line[item['metric_nominal_tokens_per_second']]
        assert math.isfinite(secs) and secs > 0 and rate == 16384 / secs
    if group['kind'] == 'audited_sequential_pair':
        assert len(group['series']) == 2
        assert len({x['metric_seconds'] for x in group['series']}) == 1
    else:
        assert len({x['metric_seconds'] for x in group['series']}) == len(group['series'])
    values[key]=lines
assert all(len(v)==1 for v in keys.values())

with (HERE/'source-index.csv').open(encoding='utf-8',newline='') as f: indices=list(csv.DictReader(f))
assert len(indices)==540
cache={}
seen=set()
for record in indices:
    key=(record['group'],record['series']); _, item=entries[key]
    n=int(record['output_line']); step=int(record['step']); assert 1<=n<=30 and step==1510+n
    assert record['output_path']==item['jsonl'] and (key,step) not in seen
    seen.add((key,step))
    source=(HERE/record['source_file']).resolve()
    if source not in cache:
        b=source.read_bytes(); lines=b.splitlines(keepends=True)
        header=next(csv.reader([lines[0].decode('utf-8-sig')]))
        cache[source]=(sha(b),lines,header)
    digest,lines,header=cache[source]; assert digest==record['source_file_sha256']
    nums=list(map(int,record['source_line_numbers'].split('|')))
    hashes=record['source_line_sha256s'].split('|'); assert len(nums)==len(hashes)
    originals=[]
    for num, digest_expected in zip(nums,hashes):
        assert num >= 2 and num <= len(lines)
        raw=lines[num-1]; assert sha(raw)==digest_expected
        cols=next(csv.reader([raw.decode('utf-8')])); assert len(cols)==len(header)
        originals.append(dict(zip(header,cols)))
    if len(originals)==1:
        x=originals[0]; assert x['run']==item['source_run'] and int(x['step'])==step
        four=[float(x['rank%d_seconds'%r]) for r in range(4)]
        assert float(x['max_rank_seconds'])==max(four)
    else:
        assert len(originals)==4 and {int(x['rank']) for x in originals}==set(range(4))
        assert all(int(x['step'])==step and x['phase']=='measured' for x in originals)
        column='default_seconds' if item['source_run']=='default' else 'explicit1_seconds'
        four=[float(x[column]) for x in originals]
    assert values[key][n-1][item['metric_seconds']]==max(four)
assert len(seen)==18*30 and len(cache)==2

for s in summaries:
    key=(s['group'],s['series']); _, item=entries[key]
    ts=[x[item['metric_seconds']] for x in values[key]]
    assert s['n']==30 and s['mean_max_rank_seconds']==statistics.mean(ts)
    assert s['median_max_rank_seconds']==statistics.median(ts)
    assert s['population_std_seconds']==statistics.pstdev(ts)
    assert s['min_seconds']==min(ts) and s['max_seconds']==max(ts)
    assert s['nominal_tokens_per_second_from_mean_duration']==16384/statistics.mean(ts)
    assert s['arithmetic_mean_of_step_nominal_tokens_per_second']==statistics.mean(16384/x for x in ts)
    assert abs(s['source_summary_expected_mean_seconds']-statistics.mean(ts))<1e-12
assert len(summaries)==18

with (HERE/'memory-peaks.csv').open(encoding='utf-8',newline='') as f: memory=list(csv.DictReader(f))
assert len(memory)==72
memseen=set(); doccache={}
for row in memory:
    key=(row['group'],row['series']); _, item=entries[key]
    assert row['source_run']==item['source_run'] and row['host']==item['host']
    assert (key,row['physical_card']) not in memseen;memseen.add((key,row['physical_card']))
    p=(HERE/row['source_file']).resolve()
    if p not in doccache:
        raw=p.read_bytes();doccache[p]=(sha(raw),json.loads(raw))
    digest,doc=doccache[p];assert digest==row['source_sha256']
    target=doc
    for part in row['source_json_pointer'].strip('/').split('/'):
        target=target[int(part)] if isinstance(target,list) else target[part]
    assert str(target['physical_card'])==row['physical_card']
    if 'timed_memory_after_max_bytes' in target:
        t=target['timed_memory_after_max_bytes']
        for field, value in [('peak_allocated',t['peak_allocated_bytes']),('peak_reserved',t['peak_reserved_bytes']),('warm_unique_storage',target['warm_storage_unique_npu_bytes'])]:
            assert int(row[field+'_bytes'])==value and float(row[field+'_gib'])==value/2**30
    else:
        assert target['run']==item['source_run']
        for field,sourcefield in [('peak_allocated','peak_allocated_gib'),('peak_reserved','peak_reserved_gib'),('warm_unique_storage','warm_deduplicated_gib')]:
            assert row[field+'_bytes']=='' and float(row[field+'_gib'])==target[sourcefield]
    assert 'not step-level memory' in row['scope']
assert len(memseen)==18*4 and len(doccache)==2

parser=loads(HERE/'parser-check.json')
assert parser['status']=='passed' and parser['files']==18 and parser['groups']==8 and parser['step_points']==540
report={'schema':'independent-website-input-audit-v1','status':'passed','groups':len(groups),'series':len(entries),'step_points':len(seen),'memory_rows':len(memseen),'independent_byte_line_sha_rechecks':sum(len(x['source_line_numbers'].split('|')) for x in indices),'source_csv_files':len(cache),'source_summary_files':len(doccache),'per_step_rate_formula':'16384 / max(four recorded rank seconds)','static_memory_only':True,'cross_group_metric_collisions':0,'offline_helper_parser_status':parser['status'],'limitations':['No site or visualization used.','Maximum rank-local step time is not a global makespan.','Nominal token count is not effective unmasked tokens.','Discrete or incomplete observations do not establish continuous exclusive device use.']}
with (HERE/'independent-check.json').open('x',encoding='utf-8',newline='\n') as f:json.dump(report,f,indent=2,ensure_ascii=False);f.write('\n')
print(json.dumps(report,ensure_ascii=False))
