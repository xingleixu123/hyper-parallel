import hashlib,json,subprocess,tarfile,re,sys
from pathlib import Path
R=Path(__file__).resolve().parent.parent; O=R/'public-final'; G=r'C:\Program Files\Git\cmd\git.exe'
base='c086152596226c99f5c06597491ee94b4c26d26c'; swap='bd037f88994d41aa3bde45823266dcf32ae3ba22'; cp='af64e123e509085e69dc31578eee712155813538'
def sha(b): return hashlib.sha256(b).hexdigest()
def out(path,x):
 p=O/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
def raw(p): return p.read_bytes()
def item(path,b): return {'original_path':str(path).replace('\\','/'),'sha256':sha(b),'text':b.decode('utf8')}
def git(repo,*args): return subprocess.check_output([G,'-C',str(R/repo),*args])
files=['tests/ut/core/optimizer/test_swap_optimizer_base.py','hyper_parallel/core/optimizer/swap_optimizer_base.py','tests/torch/swap_optimizer/test_swap_muon.py','tests/torch/swap_optimizer/swap_muon.py','tests/common/parallel_case.py','tests/common/distributed_launcher.py']; cpfiles=['hyper_parallel/models/deepseek_v3/adapter/distributed/context_parallel.py','tests/ut/distributed/context_parallel/test_mla_context_parallel.py','hyper_parallel/distributed/context_parallel/mla_context_parallel.py','tests/torch/context_parallel/test_mla_cp.py','tests/torch/context_parallel/_test_mla_cp.py']
ids={'base':base,'swap_head':swap,'cp_head':cp,'files':{}}
for variant,repo,commit,paths in [('swap_base','git-swap',base,files),('swap_head','git-swap',swap,files),('cp_base','git-cp',base,cpfiles),('cp_head','git-cp',cp,cpfiles)]:
 ids['files'][variant]={}
 for f in paths:
  b=git(repo,'show',f'{commit}:{f}'); blob=git(repo,'rev-parse',f'{commit}:{f}').decode().strip(); ids['files'][variant][f]={'git_blob':blob,'sha256':sha(b)}
out('source/identities.json',ids)
archive=R/'NPU-timeout/host129/muon-abba-evidence.tar.gz'; names=['run-base-20261107T044601Z-1b10e851','run-head-20261107T044643Z-005b8043','run-head-20261107T044743Z-509a92a4','run-base-20261107T044826Z-63f74e5f']
with tarfile.open(archive) as t:
 def read(n): return t.extractfile(n).read()
 runs=[]
 for name in names:
  d='outputs/'+name; runs.append({k:item(d+'/'+f,read(d+'/'+f)) for k,f in [('receipt','receipt.json'),('monitor','monitor.jsonl'),('log','run.log')]})
 workers=[item(m.name,read(m.name)) for m in t.getmembers() if m.isfile() and m.name.startswith('sources/') and m.name.endswith('/stdout.log')]
assert len(workers)==16
out('swap/runs.json',{'runs':runs,'independent_verification':json.loads(raw(R/'abba-parent-verification.json'))});out('swap/workers.json',workers)
d=R/'NPU-ST-helper/run-cp-20261107T034525Z-e655efb6'; cp_run={k:item('run-cp-20261107T034525Z-e655efb6/'+f,raw(d/f)) for k,f in [('receipt','receipt.json'),('monitor','monitor.jsonl'),('log','run.log')]};cp_run['independent_verification']=json.loads(raw(R/'native-helper-st-verification.json'));out('cp/run.json',cp_run)
logs=sorted((R/'NPU-ST-helper/logs').rglob('stdout.log'));assert len(logs)==4
out('cp/workers.json',[item(p.relative_to(R/'NPU-ST-helper'),raw(p)) for p in logs]);out('cp/complexity.json',json.loads(raw(R/'complexity-fix.json')))
(O/'cpu').mkdir(exist_ok=True);(O/'cpu/bench.py').write_bytes(raw(R/'CPU-timeout/bench.py'));(O/'cpu/results.json').write_bytes(raw(R/'CPU-timeout/results.json'))
out('cpu/validation-logs.json',[item(p.relative_to(R),raw(p)) for p in [R/'CPU-validation/swap-ut.log',R/'CPU-validation/complexity-cp-ut.log',R/'CPU-validation/complexity-cp-pylint.log']])
print('prepared data')

