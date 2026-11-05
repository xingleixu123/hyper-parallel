"""Reviewed candidate: run on target host; no action until explicitly invoked with spec."""
import argparse,fcntl,hashlib,importlib.util,json,os,re,shutil,signal,socket,stat,subprocess,sys,time,traceback
from pathlib import Path, PurePosixPath
import candidate
IMAGE=candidate.IMAGE
candidate.CARDS=(0,1,2,3)
HERE=Path(__file__).resolve().parent
PINS={'historical_launcher.py':'c650fde4ba7917b7750bbf97f16a4825ea00eb40d37746dae617f10623cd4646','fourcard_guards_review.py':'a94a8d3a3d2fcdc7ea136f97a1b7df553d962ce153d62a1f847c56e3ff8699e7'}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def require(ok,msg):
 if not ok: raise RuntimeError(msg)
def load(name):
 p=HERE/name; require(sha(p)==PINS[name],name+' hash mismatch'); spec=importlib.util.spec_from_file_location(name.replace('.py',''),p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
def run(args,timeout=30): return subprocess.run(args,check=True,text=True,capture_output=True,timeout=timeout).stdout
def atomic(p,obj):
 tmp=p.with_name(p.name+'.tmp-'+str(os.getpid()))
 with tmp.open('x') as f: json.dump(obj,f,indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
 os.replace(tmp,p)
def memavailable():
 for row in Path('/proc/meminfo').read_text().splitlines():
  if row.startswith('MemAvailable:'): return int(row.split()[1])*1024
 raise RuntimeError('MemAvailable missing')
def port_free(port):
 s=socket.socket(); s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,0)
 try: s.bind(('0.0.0.0',port)); return True
 except OSError: return False
 finally: s.close()
def scan_ready(helper,pidns,groups,tag,rows):
 candidates=helper.scan_workers(pidns,groups,tag); result=[]
 for rank,row in enumerate(rows):
  require(row.get('rank')==rank and row.get('local_rank')==rank and row.get('tag')==tag,'rank receipt mismatch')
  matched=[]
  for x in candidates:
   proc=Path('/proc')/str(x['host_pid'])
   try:
    env=dict(z.split(b'=',1) for z in (proc/'environ').read_bytes().split(b'\0') if b'=' in z)
    status=(proc/'status').read_text(); m=re.search(r'^NSpid:\s+([\d\s]+)$',status,re.M)
    if m and int(m.group(1).split()[-1])==row['pid'] and x['starttick']==row['starttick'] and env.get(b'RANK')==str(rank).encode() and env.get(b'LOCAL_RANK')==str(rank).encode() and env.get(b'WORLD_SIZE')==b'4' and helper.identity_alive(x): matched.append(x)
   except (FileNotFoundError,ProcessLookupError): continue
  require(len(matched)==1,'rank not uniquely mapped to namespace/cgroup/tag: '+str(rank))
  result.append({'rank':rank,'namespace_pid':row['pid'],'host_pid':matched[0]['host_pid'],'starttick':matched[0]['starttick']})
 require(len({x['host_pid'] for x in result})==4,'rank PID collision'); return result
def checkpoint_location(raw, runid, base=Path('/nfs/staging/hp-mf-master-sixpr-20261102/fourcard-checkpoints')):
 if not isinstance(raw,str) or not raw or ',' in raw: raise ValueError('checkpoint output invalid')
 p=PurePosixPath(raw)
 if not p.is_absolute() or '..' in p.parts or str(p)!=raw or p!=PurePosixPath(str(base))/runid: raise ValueError('checkpoint output must be task root/run_id')
 root=Path(base)
 for part in [*reversed(root.parents),root]:
  if part.is_symlink(): raise ValueError('symlink in checkpoint parent: '+str(part))
 if not root.is_dir() or root.resolve(strict=True)!=root: raise ValueError('checkpoint parent missing or not canonical')
 target=Path(raw)
 if os.path.lexists(target): raise FileExistsError(target)
 if shutil.disk_usage(root).free<100*2**30: raise ValueError('checkpoint filesystem free <100 GiB')
 return target
def proc_identity(proof, proc_root=Path('/proc')):
 """Only ENOENT or a readable zombie with the original tick is retired."""
 path=proc_root/str(proof['host_pid'])/'stat'
 try: raw=path.read_text()
 except FileNotFoundError: return {'kind':'absent','raw_stat':None,'expected_starttick':proof['starttick']}
 except OSError as e: raise RuntimeError('unknown proc read: '+repr(e))
 try:
  fields=raw.rsplit(') ',1)[1].split(); status=fields[0]; tick=int(fields[19])
 except (IndexError,ValueError) as e: raise RuntimeError('unknown proc stat: '+repr(e))
 require(tick==proof['starttick'],'PID starttick mismatch')
 require(status in 'RSDTtXZPIWK','unknown proc status')
 return {'kind':'zombie' if status=='Z' else 'live','state':status,'starttick':tick,'raw_stat':raw}
def perf_completion(folder, c, allow_missing_global=False):
 hashes={}
 for i in range(4):
  rank=folder/f"rank-{i}"; complete=rank/"complete.json"
  if not complete.exists():
   if allow_missing_global: return {"rank_sha256":None,"global_sha256":None}
   raise FileNotFoundError(complete)
  v=json.loads(complete.read_text()); rows=json.loads((rank/"steps.json").read_text())
  require(v.get("rank")==i and v.get("run_id")==c["run_id"] and v.get("status")=="100_updates_complete" and v.get("updates")==100 and v.get("horizon")==7500,"rank completion invalid")
  require([x.get("step") for x in rows]==list(range(6,106)) and all(len(x.get("microbatches", [])) == 1 and x["microbatches"][0].get("dataset_fields_verified") is True and x["microbatches"][0].get("logical_position") == (x["step"]-1)*4+i for x in rows),"step evidence invalid")
  for name in ("complete.json","steps.json","inventory.json","restore-verification.json","restored-state.json"):
   f=rank/name; require(f.is_file() and not f.is_symlink(),"rank evidence missing"); hashes[str(f.relative_to(folder))]=sha(f)
 return {"rank_sha256":hashes,"global_sha256":"all-ranks-complete"}
def drain_until_complete(folder,c,child,guard,clock=time.monotonic,sleeper=time.sleep,limit=60):
 deadline=clock()+limit
 while True:
  current=perf_completion(folder,c,allow_missing_global=True)
  rc=child.poll(); require(rc is None or rc==0,"completion drain torchrun nonzero")
  if current["global_sha256"] is not None and rc==0: return {"completion":current,"torchrun_returncode":rc}
  remaining=deadline-clock(); require(remaining>0,"completion drain timeout")
  guard(remaining); sleeper(min(1,remaining))
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--spec',required=True); a=ap.parse_args(); c=json.loads(Path(a.spec).read_text())
 runid=c['run_id']; cards=tuple(c.get('cards',[0,1,2,3])); require(cards==(0,1,2,3),'only cards 0-3 are permitted')
 require(re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,63}',runid) is not None,'unsafe run id')
 output_root=Path(c["output_root"])
 require(output_root.is_absolute() and ".." not in output_root.parts and output_root.is_dir() and not output_root.is_symlink() and output_root.resolve()==output_root,"unsafe output root")
 require(c['mode']=='resume' and c['stop']==105,'mode/stop')
 master=int(c['master_port']); hccl=int(c['hccl_start']); require(1024<=master<=65535 and 1024<=hccl<=65520 and not hccl<=master<hccl+16,'ports invalid')
 paths=c['paths']; required={'source','env','python','cann','driver','vendor','pyext','data_bin','data_idx','cache','scripts'}; require(set(paths)==required,'bad path map')
 for k,v in paths.items():
  p=PurePosixPath(v); require(p.is_absolute() and '..' not in p.parts and ',' not in v and Path(v).exists(),'unsafe/missing path '+k)
 require(re.fullmatch(r'[A-Za-z0-9_.-]+\.yaml',c['config']) is not None,'config filename'); require((Path(paths['scripts'])/'resume-fourcard/entry.py').is_file() and (Path(paths['scripts'])/c['config']).is_file(),'worker/config missing')
 require(sha(Path(paths['scripts'])/'resume-fourcard/entry.py')==c['entry_sha256'] and sha(Path(paths['scripts'])/c['config'])==c['config_sha256'],'worker/config SHA mismatch')
 bundle=Path(paths["scripts"])
 for name in ("issue12_hp_data.py","ordered_factory.py"):
  q=bundle/name; require(q.is_file() and not q.is_symlink() and sha(q)==c["asset_sha256"][name],"asset hash mismatch: "+name)
 manifest_path=Path(c["source_manifest"])
 require(manifest_path.resolve()==(bundle/"source-manifest.json").resolve() and sha(manifest_path)==c["source_manifest_sha256"],"source manifest SHA/path mismatch")
 manifest=json.loads(manifest_path.read_text(encoding="utf-8-sig"))
 require(manifest.get("schema")=="xing_4-source-snapshot-v1" and manifest.get("commit")==c["source_head"] and manifest.get("tree")==c["source_tree"],"source identity mismatch")
 source=Path(paths["source"]); expected=manifest["files"]
 require(isinstance(expected,dict) and bool(expected),"empty source manifest")
 for name,row in expected.items():
  rel=PurePosixPath(name); require(not rel.is_absolute() and ".." not in rel.parts,"unsafe source entry")
  q=source/name; require(not q.is_symlink() and q.is_file() and q.stat().st_size==row["size"] and sha(q)==row["sha256"],"source file mismatch "+name)
 for key in ("data_bin","data_idx"):
  require(sha(paths[key])==c[key+"_sha256"],key+" SHA mismatch")
 for key in ("checkpoint_path", "reference_path"):
  q=Path(c[key]); require(q.is_absolute() and q.is_dir() and not q.is_symlink() and q.resolve()==q, "unsafe " + key)
 require(c.get("checkpoint_sha256") and isinstance(c["checkpoint_sha256"],dict), "checkpoint inventory required")
 checkpoint=Path(c["checkpoint_path"])
 actual_files={str(q.relative_to(checkpoint)) for q in checkpoint.rglob("*") if q.is_file()}
 require(actual_files==set(c["checkpoint_sha256"]), "checkpoint inventory mismatch")
 for name,digest in c["checkpoint_sha256"].items():
  rel=PurePosixPath(name); require(not rel.is_absolute() and ".." not in rel.parts and not (checkpoint/name).is_symlink() and sha(checkpoint/name)==digest,"checkpoint SHA mismatch: "+name)
 require(".metadata" in actual_files and all("extra_state/extra_state_rank_"+str(i)+".pt" in actual_files for i in range(4)), "checkpoint incomplete")
 for i in range(4):
  q=Path(c["reference_path"])/("rank-"+str(i))/"checkpoint-5-live-state.json"
  require(q.is_file() and not q.is_symlink() and sha(q)==c["reference_sha256"][str(i)],"reference SHA mismatch")
 root=Path(c['output_root']); root.mkdir(parents=True,exist_ok=True); folder=root/runid; folder.mkdir(mode=0o700)
 state={'status':'starting','run_id':runid,'mode':c['mode'],'stop':c['stop'],'returncode':None}; holder=None; child=None; locks=[]; guardlocks=[]
 def save(): atomic(folder/'controller.json',state)
 try:
  require(shutil.disk_usage(folder).free>=int(c['min_disk_bytes']),'insufficient disk'); require(memavailable()>=int(c['min_mem_bytes']),'insufficient RAM'); [require(port_free(port),'port in use: '+str(port)) for port in [master,*range(hccl,hccl+16)]]
  helper=load('historical_launcher.py'); guards=load('fourcard_guards_review.py'); guards.CARDS=cards; guardlocks=guards.acquire_locks()
  for card in cards:
   fd=os.open('/tmp/hp-mf-formal-single-card-'+str(card)+'.lock',os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW|os.O_CLOEXEC,0o600); locks.append(fd); st=os.fstat(fd); require(stat.S_ISREG(st.st_mode) and st.st_uid==0 and st.st_nlink==1 and not (stat.S_IMODE(st.st_mode)&0o077),'invalid lock'); fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
  for card in cards:
   raw=run(['npu-smi','info','-t','proc-mem','-i',str(card),'-c','0']); require(candidate.physical_pids(raw)==set(),'card occupied')
  require(subprocess.run(['docker','image','inspect',IMAGE],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0,'image absent')
  mounts=[(c['checkpoint_path'],'/work/checkpoint',True),(c['reference_path'],'/work/reference',True),(paths['source'],'/work/sources/hyper-parallel',True),(paths['env'],'/work/env',True),(paths['python'],'/usr/local/python3.12.14',True),(paths['cann'],'/work/runtime/cann920/cann-9.2.0',True),(paths['driver'],'/usr/local/Ascend/driver',True),(paths['vendor'],'/work/task/vendor-opp',True),(paths['pyext'],'/work/pyext',True),(paths['data_bin'],'/datasets/mmap_deepseekv3_datasets_text_document.bin',True),(paths['data_idx'],'/datasets/mmap_deepseekv3_datasets_text_document.idx',True),(paths['cache'],'/work/source-mf/cache',True),(paths['scripts'],'/work/scripts',True),(str(folder),'/work/output',False)]
  name='sixpr-fourcard-'+runid; label='sixpr-fourcard-'+runid; args=['docker','run','-d','--rm','--cidfile',str(folder/'container.cid'),'--name',name,'--label','codex.task='+label,'--read-only','--network','host','--cpus','16','--memory','192g','--tmpfs','/tmp:rw,nosuid,nodev,size=4g','--workdir','/work/output']
  for d in ['/dev/davinci'+str(n) for n in cards]+['/dev/davinci_manager','/dev/devmm_svm','/dev/hisi_hdc']: args+=['--device',d+':'+d]
  for src,dst,ro in mounts: args+=['--mount','type=bind,src='+src+',dst='+dst+(',readonly' if ro else '')]
  args+=['--entrypoint','/usr/bin/sleep',IMAGE,'infinity']; state['holder_argv']=args; save(); run(args)
  holder=json.loads(run(['docker','inspect',name]))[0]; pid=holder['State']['Pid']; require(holder['Image']==IMAGE and holder['Config']['Labels']['codex.task']==label,'holder identity')
  actual={x['Destination']:(os.path.realpath(x['Source']),x['RW']) for x in holder['Mounts'] if x['Type']=='bind'}; require(actual=={dst:(os.path.realpath(src),not ro) for src,dst,ro in mounts},'mount mismatch')
  groups=helper.cgroup_paths(pid); require(holder['Id'] in groups['devices'] and holder['Id'] in groups['memory'],'holder cgroup mismatch'); guards.verify_device_cgroup(groups['devices'])
  pidns=os.readlink('/proc/'+str(pid)+'/ns/pid'); hostmnt=os.readlink('/proc/self/ns/mnt'); tag='sixpr-fourcard-'+runid+'-'+holder['Id'][:12]; (folder/'home').mkdir()
  env=['HP_PR310_ID='+tag,'HP_FOURCARD_TAG='+tag,'HP_FOURCARD_RUN_ID='+runid,'HOME=/work/output/home','TMPDIR=/tmp','PYTHONDONTWRITEBYTECODE=1','HYPER_PARALLEL_PLATFORM=torch','HF_HUB_OFFLINE=1','TRANSFORMERS_OFFLINE=1','PATH=/work/env/bin:/usr/local/python3.12.14/bin:/usr/bin:/bin','LD_LIBRARY_PATH=/usr/local/python3.12.14/lib:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/driver','ASCEND_RT_VISIBLE_DEVICES=0,1,2,3','ASCEND_CUSTOM_OPP_PATH=','PYTHONPATH=/work/sources/hyper-parallel:/work/scripts:/work/pyext','TORCH_DEVICE_BACKEND_AUTOLOAD=0','OMP_NUM_THREADS=2','MKL_NUM_THREADS=2','HCCL_NPU_SOCKET_PORT_RANGE='+str(hccl)+'-'+str(hccl+15),'ASCEND_WORK_PATH=/work/output/ascend','EXPECTED_HOST_MNT_NS='+hostmnt]
  worker=['/work/env/bin/python','-B','-m','torch.distributed.run','--nnodes=1','--node-rank=0','--nproc-per-node=4','--master-addr=127.0.0.1','--master-port='+str(master),'/work/scripts/resume-fourcard/entry.py','--config','/work/scripts/'+c['config'],'--mode',c['mode'],'--stop',str(c['stop'])]
  cmd=['nsenter','-t',str(pid),'--root','--pid','--uts','--ipc','--net','--wd','--','/usr/bin/env','-i',*env,'bash','--noprofile','--norc','-c','source /work/runtime/cann920/cann-9.2.0/set_env.sh && source /work/task/vendor-opp/vendors/omni_training_custom_transformer/bin/set_env.bash && exec "$@"','fourcard-entry',*worker]
  state.update(holder_id=holder['Id'],holder_pid=pid,hostconfig=holder['HostConfig'],cgroups=groups,pid_namespace=pidns,host_mount_namespace=hostmnt,entry_argv=cmd,tag=tag); save()
  def join():
   helper.cg_join(groups)
   for controller in ('cpu','cpuacct','cpuset','pids'):
    parent=Path('/sys/fs/cgroup')/controller
    if controller in groups and parent.exists():
     target=(parent/groups[controller].lstrip('/')).resolve(); require(target.is_relative_to(parent.resolve()),'cgroup escapes'); (target/'cgroup.procs').write_text(str(os.getpid()))
  with (folder/'stdout.log').open('x') as stdout,(folder/'stderr.log').open('x') as stderr:
   child=subprocess.Popen(cmd,stdout=stdout,stderr=stderr,preexec_fn=join,start_new_session=True); state['entry_pid']=child.pid; save(); deadline=time.monotonic()+180
   ready=[folder/('rank-'+str(i))/'device-ready.json' for i in range(4)]
   while not all(p.is_file() for p in ready):
    require(child.poll() is None,'torchrun exited before ranks ready'); require(time.monotonic()<deadline,'rank handshake timeout'); time.sleep(.2)
   rows=[json.loads(p.read_text()) for p in ready]; proof=scan_ready(helper,pidns,groups,tag,rows)
   observations={card:run(['npu-smi','info','-t','proc-mem','-i',str(card),'-c','0']) for card in cards}; mapping=candidate.require_bijection(proof,observations)
   require(scan_ready(helper,pidns,groups,tag,rows)==proof,'rank identity changed'); state['mapping']=mapping; state['npu_smi_initial']=observations; save(); atomic(folder/'physical-mapping-release.json',{'run_id':runid,'tag':tag,**mapping})
   deadline=time.monotonic()+12*3600
   while child.poll() is None:
    require(time.monotonic()<deadline,'job timeout 4h'); time.sleep(5)
    if child.poll() is not None: break
    with (folder/'device-observations.jsonl').open('a') as log:
     sample={str(card):run(['npu-smi','info','-t','proc-mem','-i',str(card),'-c','0']) for card in cards}
     log.write(json.dumps({'time':time.time(),'cards':sample})+'\n'); log.flush()
    # Check every sample for foreign PIDs, even if torchrun exits during sampling.
    seen_by_rank=[candidate.physical_pids(sample[str(card)]) for card in cards]
    for rank,seen in enumerate(seen_by_rank):
     require(not (seen-{proof[rank]['host_pid']}),'external/unexpected device PID during run')
    if child.poll() is not None: break
    try:
     identities=[proc_identity(item) for item in proof]
    except BaseException as exc:
     raw_after=[]
     for item in proof:
      try: raw_after.append({'host_pid':item['host_pid'],'raw_stat':(Path('/proc')/str(item['host_pid'])/'stat').read_text()})
      except OSError as err: raw_after.append({'host_pid':item['host_pid'],'error':repr(err)})
     atomic(folder/'exit-identity-error.json',{'run_id':runid,'proof':proof,'sample':sample,'proc_observed_after_error':raw_after,'error':repr(exc)})
     raise
    if any(x['kind']!='live' for x in identities):
     receipt={'run_id':runid,'proof':proof,'sample':sample,'proc':identities,'started_at':time.time(),'status':'checking_completion'}
     try:
      receipt['drain_checks']=[]
      def drain_guard(remaining):
       check={'time':time.time(),'cards':{},'proc':[]}; receipt['drain_checks'].append(check)
       for rank,card in enumerate(cards):
        remaining_now=60-(time.monotonic()-drain_started); require(remaining_now>0,'completion drain timeout before device query')
        raw=run(['npu-smi','info','-t','proc-mem','-i',str(card),'-c','0'],timeout=min(5,remaining_now));check['cards'][str(card)]=raw
        seen=candidate.physical_pids(raw);require(not (seen-{proof[rank]['host_pid']}),'external/unexpected device PID during drain')
       check['proc']=[proc_identity(item) for item in proof]
      drain_started=time.monotonic()
      receipt.update(drain_until_complete(folder,c,child,drain_guard))
      receipt['status']='drain_complete_rc0'
     except BaseException as e:
      receipt['error']=repr(e); raise
     finally: atomic(folder/'exit-drain.json',receipt)
     break
   state['returncode']=child.returncode; require(child.returncode==0,'torchrun nonzero'); state['completion']=perf_completion(folder,c); state['status']='finished_process_exit_0'
 except BaseException as exc:
  state.update(status='controller_failed',error=repr(exc),traceback=traceback.format_exc())
 finally:
  if child is not None and child.poll() is None:
   os.killpg(child.pid,signal.SIGTERM)
   try: child.wait(timeout=10)
   except subprocess.TimeoutExpired: os.killpg(child.pid,signal.SIGKILL); child.wait()
   state['returncode']=child.returncode
  if holder is None and (folder/'container.cid').is_file():
   cid=(folder/'container.cid').read_text().strip(); r=subprocess.run(['docker','inspect',cid],capture_output=True,text=True)
   if r.returncode==0:
    obj=json.loads(r.stdout)[0]; require(obj['Config']['Labels'].get('codex.task')==label,'cleanup label mismatch'); holder=obj
  if holder is not None:
   require(holder['Config']['Labels'].get('codex.task')==label,'cleanup ownership mismatch'); state['stop_rc']=subprocess.run(['docker','stop','--time','3',holder['Id']],capture_output=True,text=True).returncode; state['holder_removed']=subprocess.run(['docker','inspect',holder['Id']],capture_output=True).returncode!=0
  for fd in reversed(locks+guardlocks): os.close(fd)
  save()
 print(json.dumps({'output':str(folder),'status':state['status'],'returncode':state['returncode']})); return 0 if state['status']=='finished_process_exit_0' and state.get('holder_removed') else 1
if __name__=='__main__': sys.exit(main())
