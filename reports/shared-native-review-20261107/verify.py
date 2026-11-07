import hashlib,json,re,statistics
from pathlib import Path
R=Path(__file__).resolve().parent
def load(p): return json.loads((R/p).read_text(encoding='utf-8'))
def sha(b): return hashlib.sha256(b).hexdigest()
def check_item(x): assert sha(x['text'].encode('utf-8'))==x['sha256'],x['original_path']
m=load('manifest.json'); actual={p.relative_to(R).as_posix():sha(p.read_bytes()) for p in R.rglob('*') if p.is_file() and p!=R/'manifest.json'}; assert m['files']==actual,'manifest mismatch'
ids=load('source/identities.json'); assert ids['base']=='c086152596226c99f5c06597491ee94b4c26d26c' and ids['swap_head']=='bd037f88994d41aa3bde45823266dcf32ae3ba22' and ids['cp_head']=='af64e123e509085e69dc31578eee712155813538'
prod='hyper_parallel/core/optimizer/swap_optimizer_base.py'; cprod='hyper_parallel/distributed/context_parallel/mla_context_parallel.py'; launcher='tests/torch/context_parallel/test_mla_cp.py'; worker='tests/torch/context_parallel/_test_mla_cp.py'
bench=load('cpu/results.json'); assert bench['source_sha256']=={'base':ids['files']['swap_base'][prod]['sha256'],'head':ids['files']['swap_head'][prod]['sha256']}; assert bench['batch_iterations']==200 and bench['batches_per_variant']==12
for c in bench['cases'].values():
 for v in ('base','head'): assert abs(statistics.median(c['samples_ns_per_construction'][v])-c['summary'][v]['median'])<1e-7
sw=load('swap/runs.json'); walls={'base':[],'head':[]}; calls={'base':[],'head':[]}
for n,run in enumerate(sw['runs']):
 for x in run.values(): check_item(x)
 rec=json.loads(run['receipt']['text']); rows=[json.loads(s) for s in run['monitor']['text'].splitlines()]; log=run['log']['text']; v=['base','head','head','base'][n]
 assert rec['variant']==v and rec['status']=='completed' and rec['returncode']==0 and rec['source_archive_compare']['exit']==0 and rec['cards']==[4,5,6,7] and len(rows)==10 and all(not x['unknown'] and not x['foreign'] for x in rows)
 assert rec['source_commit']==(ids['base'] if v=='base' else ids['swap_head']); assert '1 passed' in log
 walls[v].append(rec['wall_seconds']);calls[v].append(float(re.search(r'([0-9.]+)s call\s+.*test_swap_muon_1',log).group(1)))
workers=load('swap/workers.json'); assert len(workers)==16
for w in workers: check_item(w); assert re.search(r'=+\s*1 passed.*? in [0-9.]+s\s*=+',w['text'])
cp=load('cp/run.json')
for k in ('receipt','monitor','log'): check_item(cp[k])
r=json.loads(cp['receipt']['text']); rows=[json.loads(s) for s in cp['monitor']['text'].splitlines()]; assert r['status']=='completed' and r['returncode']==0 and r['source_archive_compare']['exit']==0 and r['cards']==[4,5,6,7] and len(rows)==16 and all(not x['unknown'] and not x['foreign'] for x in rows)
assert r['production_sha256']==ids['files']['cp_head'][cprod]['sha256'] and r['launcher_sha256']==ids['files']['cp_head'][launcher]['sha256'] and r['st_worker_sha256']==ids['files']['cp_head'][worker]['sha256']; assert '1 passed' in cp['log']['text']
workers=load('cp/workers.json'); assert len(workers)==4
for w in workers: check_item(w); assert re.search(r'=+\s*1 passed.*? in [0-9.]+s\s*=+',w['text'])
for x in load('cpu/validation-logs.json'): check_item(x)
logs=load('cpu/validation-logs.json'); assert len(logs)==3 and '130 passed, 26 subtests passed' in logs[0]['text'] and '9 passed, 18 subtests passed' in logs[1]['text'] and 'exit_code=0' in logs[2]['text']
for file in ("swap/workers.json","cp/workers.json"):
 for item in load(file):
  assert not re.search(r"\b[1-9][0-9]* (?:failed|skipped|xfailed|xpassed)\b",item["text"]),item["original_path"]
records=[]
for item in load("cp/workers.json"):
 for line in item["text"].splitlines():
  pos=line.find('{"tp":')
  if pos>=0:records.append(json.JSONDecoder().raw_decode(line[pos:])[0])
assert len(records)==3
assert {(x["strategy"],x["overlap"]) for x in records}=={("expanded_ulysses",False),("expanded_ulysses",True),("latent_kv_head",True)}
assert all(x["tp"]==x["cp"]==2 and x["backward_passes"]==2 and x["gradient_relative_l2"]<0.025 for x in records)
assert max(x["gradient_relative_l2"] for x in records)==0.00379561724947224

print('Verified manifest, source identity cross-checks, 20 workers and five controller receipts.')
for v in ('base','head'): print(v,'walls',walls[v],'mean',statistics.mean(walls[v]),'calls',calls[v],'mean',statistics.mean(calls[v]))
print('CPU medians (ns)',{k:{v:x['summary'][v]['median'] for v in ('base','head')} for k,x in bench['cases'].items()})

