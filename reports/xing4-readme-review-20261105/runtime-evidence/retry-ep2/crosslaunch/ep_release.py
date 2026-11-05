"""Read-only cross-layout gate. Writes release only after bounded CPU DCP verification."""
import argparse,hashlib,json,os,signal,subprocess,time
from pathlib import Path

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def atomic(p,obj):
 t=p.with_name(p.name+'.tmp')
 with t.open('x') as f:json.dump(obj,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(t,p)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--spec',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();c=json.loads(Path(a.spec).read_text());out=Path(a.output)
 source=Path(c['baseline_output']);bundle=Path(c['crosslaunch_path']); spec=Path(c['source_verify_spec']);source_receipt=Path(c['source_verify_receipt'])
 for q,key in [(spec,'source_verify_spec_sha256'),(source_receipt,'source_verify_receipt_sha256')]:
  if q.is_symlink() or not q.is_file() or sha(q)!=c[key]:raise ValueError('bad source verification identity')
 prior=json.loads(source_receipt.read_text());seed=json.loads(spec.read_text())
 if prior.get('status')!='passed' or prior.get('source_step')!=5 or prior.get('spec_sha256')!=c['source_verify_spec_sha256']:raise ValueError('source DCP not independently verified')
 deadline=time.monotonic()+7200
 proofs=[out/('rank-'+str(i))/'restored-state.json' for i in range(4)]
 while not all(q.is_file() and not q.is_symlink() for q in proofs):
  if time.monotonic()>deadline:raise TimeoutError('target rank proofs')
  time.sleep(1)
 inventories=[json.loads((out/('rank-'+str(i))/'inventory.json').read_text()) for i in range(4)]
 dp=[x['mesh_sizes']['dp_rank'] for x in inventories]
 if sorted(dp)!=list(range(4)) or any(x['mesh_sizes']['ep_size']!=2 or x['mesh_sizes']['dp_size']!=4 or x['mesh_sizes']['tp_size']!=1 or x['mesh_sizes']['cp_size']!=1 or x['microbatches']!=1 for x in inventories):raise ValueError('target topology/data DP mismatch')
 states=[json.loads(q.read_text()) for q in proofs]; rank0=states[0]
 for row in seed['tensors']:
  for b in row['targets']:
   name=b['name'];section=b['section'];rec=rank0['model'][name] if section=='model' else rank0['slots'][name][b['slot']]
   b['plain_full']=rec['mesh'] is None
 for key,decl in seed['optimizer_bytes'].items():
  rem=key.removeprefix('optimizer.param_groups.');matches=[(i,k) for i,g in enumerate(rank0['groups']) for n in g['parameters'] for k in g['options'] if rem==n+'.'+k]
  if len(matches)!=1:raise ValueError('target optimizer byte mapping differs '+key)
  decl['group_index'],decl['field']=matches[0]
 seed['target_rank_proofs']={str(i):{'path':'/target/rank-'+str(i)+'/restored-state.json','sha256':sha(q)} for i,q in enumerate(proofs)}
 target=out/'target-verify-spec.json';atomic(target,seed);target_sha=sha(target)
 label='xing4-ep-gate-'+c['run_id']; cid=out/'gate-container.cid'
 cmd=['docker','run','--rm','--cidfile',str(cid),'--label','codex.task='+label,'--network','none','--cpus=2','--memory=12g','--read-only','--tmpfs','/tmp:rw,size=1g','-v',str(source)+':/evidence:ro','-v',str(out)+':/target','-v',str(bundle)+':/audit:ro','-v',c['paths']['source']+':/source:ro','-v',c['paths']['env']+':/env:ro','-v',c['paths']['python']+':/usr/local/python3.12.14:ro','-e','PYTHONPATH=/source:/audit','-e','HYPER_PARALLEL_PLATFORM=torch','-e','TORCH_DEVICE_BACKEND_AUTOLOAD=0','-e','LD_LIBRARY_PATH=/usr/local/python3.12.14/lib','-e','OMP_NUM_THREADS=2','-e','MKL_NUM_THREADS=2','--entrypoint','/env/bin/python',c['image'],'-B','/audit/verify_dcp.py','--spec','/target/target-verify-spec.json','--spec-sha256',target_sha,'--max-tensor-bytes','2500000000','--output','/target/target-verify.json']
 def interrupted(signum, frame):
  raise RuntimeError('gate interrupted signal '+str(signum))
 signal.signal(signal.SIGTERM,interrupted); signal.signal(signal.SIGINT,interrupted)
 try:
  subprocess.run(cmd,check=True,timeout=7200)
 finally:
  if cid.is_file():
   container_id=cid.read_text().strip()
   probe=subprocess.run(['docker','inspect',container_id],capture_output=True,text=True)
   if probe.returncode==0:
    info=json.loads(probe.stdout)[0]
    if info.get('Config',{}).get('Labels',{}).get('codex.task')!=label:raise RuntimeError('gate container label mismatch')
    subprocess.run(['docker','stop','--time','3',container_id],capture_output=True,check=True)
 verification=out/'target-verify.json'; result=json.loads(verification.read_text())
 if result.get('status')!='passed' or result.get('spec_sha256')!=target_sha:raise ValueError('target full DCP verification failed')
 if any(sha(q)!=seed['target_rank_proofs'][str(i)]['sha256'] for i,q in enumerate(proofs)):raise ValueError('target proof changed')
 tag=json.loads((out/'physical-mapping-release.json').read_text())['tag']
 atomic(out/'restore-release.json',{'status':'full_state_verified','run_id':c['run_id'],'tag':tag,'rank_proof_sha256':{str(i):sha(q) for i,q in enumerate(proofs)},'source_verify_sha256':sha(source_receipt),'target_verify_sha256':sha(verification),'target_spec_sha256':target_sha})
if __name__=='__main__':main()
