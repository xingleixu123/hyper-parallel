#!/usr/bin/env python3
"""REVIEW DRAFT: HP state stages; fail closed until independently reviewed inputs exist."""
import argparse, base64, errno, fcntl, hashlib, json, os, re, signal, socket, stat, subprocess, sys, time
from pathlib import Path
R=Path('/home/xxl/hp-mf-cann92-pr310-20261028'); S=R/'hp-e499-formal-scripts-v4'; SRC=R/'sources/hp-pr310-e499-33096-v1'; VENV=R/'envs/hp-pta212-hf521-cpu'; PY=R/'runtime/python3.12.14'; CANN=Path('/home/xxl/hp-mf-cann92-pr301-20261027/runtime/cann920'); DR=Path('/usr/local/Ascend/driver'); NFS=Path('/nfs/staging/hp-mf-cann92-pr310-20261028-01a17d83'); TMP=Path('/tmp/hp-mf-cann92-pr301-20261027'); LOCK=TMP/'module-card1.lock'; D='/home/xxl/mmap_deepseekv3_datasets_text_document'
IMAGE='sha256:5f9e677f439d11ab4e7c888f54b142335a27eee7411cbd2ce894a4798a70146d'; HEAD='33096a1030012777a5f71502dee009a7dd409bf9'; DEVS=['/dev/davinci1','/dev/davinci_manager','/dev/devmm_svm','/dev/hisi_hdc']; GiB=1024**3
PINS={'examples/training_demo/train_text.py':'a012e86f7976b15ac5140044420676100ea0140cecc1125ea7f6603a55d5481e','hyper_parallel/models/xing40/modeling_xing40.py':'97289847512006383c578fc60ea8d7c6eb38e274d717dad2d8c6cad7df0f45bd','hyper_parallel/models/xing40/adapter/optim.py':'225f7331717ee8b5dd1c3a54e8341951fc30f7f3584fa7416458eae1db71c61a','hyper_parallel/models/xing40/adapter/builder.py':'75115b565098ef47c14d9e28e0f7db22e0cce032da6a8da5c19363d8aa553ab1'}
DATA={'.bin':(4645101328,'6b925f342504c3eae80c5bde3163ff5924ad36b262bc6a276839a6f3f817ea3c'),'.idx':(18610322,'d535e502d62368e7a195b27a9a717543534ee1eaa9b4f9ce45f4d08d746cb5cd')}
def fail(s): raise RuntimeError(s)
def run(a,timeout=60):
 p=subprocess.run(a,text=True,capture_output=True,timeout=timeout)
 if p.returncode: fail(f'{a[0]} exit {p.returncode}: {p.stderr.strip()}')
 return p.stdout
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''): h.update(b)
 return h.hexdigest()
def pin(p,h):
 if p.is_symlink() or not p.is_file() or sha(p)!=h: fail('pin differs: '+str(p))
def port(n):
 # TCPStore wildcard listener snapshot; closing it does not reserve the port.
 try:
  s=socket.socket(socket.AF_INET6,socket.SOCK_STREAM)
 except OSError as exc:
  if exc.errno not in (errno.EAFNOSUPPORT,errno.EPROTONOSUPPORT):raise
  with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as v4:
   v4.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);v4.bind(('0.0.0.0',n));v4.listen(1)
  return 'ipv4-wildcard'
 with s:
  s.setsockopt(socket.IPPROTO_IPV6,socket.IPV6_V6ONLY,0)
  if s.getsockopt(socket.IPPROTO_IPV6,socket.IPV6_V6ONLY)!=0:fail('IPv6 socket is not dual stack')
  s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
  try:s.bind(('::',n))
  except OSError as exc:
   if exc.errno!=errno.EADDRNOTAVAIL:raise
   with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as v4:
    v4.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);v4.bind(('0.0.0.0',n));v4.listen(1)
   return 'ipv4-wildcard'
  s.listen(1)
 return 'ipv6-dual-wildcard'
def cg(pid):
 lines=Path('/proc/%s/cgroup'%pid).read_text().splitlines(); paths=[x.split(':',2)[2] for x in lines if ':memory:' in x]
 if len(paths)!=1:fail('expected v1 memory cgroup')
 base=Path('/sys/fs/cgroup/memory').resolve(); cur=(base/paths[0].lstrip('/')).resolve(); out=[]
 if not cur.is_relative_to(base):fail('cgroup escaped root')
 while True:
  lim=int((cur/'memory.limit_in_bytes').read_text()); use=int((cur/'memory.usage_in_bytes').read_text()); free=lim-use;out.append([str(cur),lim,use,free])
  if free<96*GiB:fail('memory cgroup headroom <96 GiB at '+str(cur))
  if cur==base:break
  cur=cur.parent
 return out
def env_fingerprint():
 base=VENV/'lib/python3.12/site-packages'; fields={}
 for name in ['torch-2.12.0+cpu.dist-info','torch_npu-2.12.0.post3.dist-info','transformers-5.21.0.dist-info','tokenizers-0.23.3.dist-info','hyper_parallel-1.2.0.dev0.dist-info']:
  p=base/name/'METADATA'
  if p.is_symlink() or not p.is_file():fail('missing metadata '+str(p))
  match=re.search(r'(?m)^Version:\s*(\S+)',p.read_text())
  if not match:fail('metadata version absent '+str(p))
  fields[name]={'version':match.group(1),'sha256':sha(p)}
 for key,path in [('pta_direct_url',base/'torch_npu-2.12.0.post3.dist-info/direct_url.json'),('hp_direct_url',base/'hyper_parallel-1.2.0.dev0.dist-info/direct_url.json'),('hp_pth',base/'__editable__.hyper_parallel-1.2.0.dev0.pth'),('pyvenv',VENV/'pyvenv.cfg'),('cann_env',CANN/'cann-9.2.0/set_env.sh'),('python',PY/'bin/python3.12')]:fields[key]=sha(path)
 return {'fields':fields,'sha256':hashlib.sha256(json.dumps(fields,sort_keys=True,separators=(',',':')).encode()).hexdigest()}
def identity(path):
 st=path.stat();return [st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns]
def regular_below(root,relative):
 if not isinstance(relative,str) or not relative or Path(relative).is_absolute() or any(x in ('','..','.') for x in relative.split('/')):fail('invalid relative input path')
 p=root
 for item in relative.split('/'):
  p=p/item
  if p.is_symlink():fail('symlink in input path')
 if not p.is_file() or not p.resolve().is_relative_to(root):fail('input file missing or escapes root')
 return p
def complete_job(report_path,report_sha,create_path,create_sha,terminal_path,terminal_sha,expected_mode):
 for value in (report_sha,create_sha,terminal_sha):
  if not re.fullmatch('[0-9a-f]{64}',value or ''):fail('diagnostic/export provenance SHA required')
 report_path=Path(report_path);create_path=Path(create_path);terminal_path=Path(terminal_path)
 for p,h in ((report_path,report_sha),(create_path,create_sha),(terminal_path,terminal_sha)):pin(p,h)
 if report_path.name!='report.json' or report_path.parent.name!=expected_mode or not report_path.resolve().parent.parent.is_relative_to(NFS.resolve()) or create_path.resolve().parent!=report_path.resolve().parent.parent or terminal_path.resolve().parent!=report_path.resolve().parent.parent:fail('provenance files must match task NFS run and stage')
 report=json.loads(report_path.read_text());create=json.loads(create_path.read_text());term=json.loads(terminal_path.read_text());pid=term.get('supervisor_pid');tick=term.get('supervisor_starttick')
 if term.get('state')!='complete' or term.get('exit_code')!=0 or term.get('container_id')!=create.get('id') or create.get('mode')!=expected_mode or not isinstance(pid,int) or pid<=1 or not isinstance(tick,int) or tick<=0:fail('producer terminal/create provenance differs')
 try:
  if starttick(pid)==tick:fail('producer still exists')
 except FileNotFoundError:pass
 return report,create,term
def formal_inputs(a):
 for value in (a.worker_sha256,a.worker_manifest_sha256,a.ready_sha256,a.collector_sha256,a.hp_config_sha256,a.data_bin_sha256,a.data_idx_sha256):
  if not re.fullmatch('[0-9a-f]{64}',value or ''):fail('external reviewed pins required')
 if a.worker_sha256!='8d6a398c8476a00e8abff776242c60f3a3af9542a75759705de1ede4df02517b' or a.worker_manifest_sha256!='8316a6b91d024d0be54967c958377897122f4b9c31ea282a9dfe586ed95557dd':fail('formal v4 script pins differ')
 worker=S/'pr310_formal_candidate_v4/worker.py';pin(worker,a.worker_sha256);wm=Path(a.worker_manifest);pin(wm,a.worker_manifest_sha256);doc=json.loads(wm.read_text());tree=S.resolve();entries=doc.get('files');worker_files={}
 if doc.get('schema')!='pr310-worker-files-v1' or not isinstance(entries,list) or not entries:fail('worker manifest schema differs')
 for item in entries:
  rel=item.get('relative_path');size=item.get('size');digest=item.get('sha256')
  if rel in worker_files or not isinstance(size,int) or size<0 or not re.fullmatch('[0-9a-f]{64}',digest or ''):fail('invalid script manifest item')
  p=regular_below(tree,rel);before=identity(p)
  if before[2]!=size or sha(p)!=digest or identity(p)!=before:fail('script file changed '+str(p))
  worker_files[rel]=digest
 actual=set()
 for directory,dirs,names in os.walk(tree,followlinks=False):
  base=Path(directory)
  for name in dirs:
   if (base/name).is_symlink():fail('symlink scripts dir')
  for name in names:
   p=base/name
   if p.is_symlink() or not p.is_file():fail('non-regular scripts input')
   actual.add(p.relative_to(tree).as_posix())
 if actual!=set(worker_files) or worker_files.get('pr310_formal_candidate_v4/worker.py')!=a.worker_sha256 or not all(any(k.startswith(x+'/') for k in worker_files) for x in ('pr310_state_candidate_v4','pr310_data','data_contract','netbase_b2_hp')):fail('script tree differs')
 collector=Path(a.collector);pin(collector,a.collector_sha256)
 ready=Path(a.ready_dir)
 if ready.is_symlink() or not ready.is_dir() or not ready.resolve().is_relative_to(NFS.resolve()):fail('ready root outside task NFS')
 ready=ready.resolve();readyfile=regular_below(ready,'ready-spec.json');pin(readyfile,a.ready_sha256);spec=json.loads(readyfile.read_text())
 if spec.get('schema')!='mf1500-ready-v1' or spec.get('status')!='anchor_and_natural_1501_ready' or spec.get('collector_sha256')!=a.collector_sha256:fail('ready schema or collector differs')
 common=spec.get('common',{})
 if common.get('global_step')!=1500 or common.get('consumed_samples')!=3000 or common.get('global_batch_size')!=2:fail('MF1500 progress differs')
 source=Path(spec.get('run_root',''))
 if source.is_symlink() or not source.is_dir() or not source.resolve().is_relative_to(NFS.resolve()) or source.resolve()==ready or ready.is_relative_to(source.resolve()):fail('source root unsafe or overlaps ready')
 source=source.resolve();checked={}
 for label,key in (('anchor','anchor_manifest'),('constructor','constructor'),('first1501','first_1501_file')):
  item=spec[key];p=regular_below(ready,{'anchor':'anchor-manifest.json','constructor':'constructor-inventory.json','first1501':'first-1501.json'}[label]);pin(p,item['sha256']);checked[label]=item['sha256']
 anchor=json.loads((ready/'anchor-manifest.json').read_text());first=json.loads((ready/'first-1501.json').read_text())
 if anchor.get('passed') is not True or anchor.get('rng_saved_equal_live') is not True or anchor.get('common')!=common or first!=spec.get('first_1501') or first.get('global_step')!=1501 or first.get('consumed_samples')!=3002 or len(first.get('microbatches',[]))!=2:fail('MF1500/1501 ready evidence differs')
 items={'model':spec['tensors']['model'],'optimizer':spec['tensors']['optimizer'],'common':spec['common_file'],'rng':spec['rng_file']}
 for name,item in spec['indices'].items():items['index:'+name]=item
 if set(spec['indices'])!={'document_index.npy','sample_index.npy','shuffle_index.npy'}:fail('MF index set differs')
 if anchor.get('common_sha256')!=items['common']['sha256'] or anchor.get('rng_file_sha256')!=items['rng']['sha256'] or any(anchor.get(k,{}).get('sha256')!=items[k]['sha256'] for k in ('model','optimizer')):fail('anchor file hashes differ')
 for label,item in items.items():
  path=Path(item['path'])
  if not path.is_absolute() or not path.is_relative_to(source):fail('ready source path escaped run')
  p=regular_below(source,path.relative_to(source).as_posix());before=identity(p)
  if before!=item['identity'] or (a.stage!='resume' or label not in ('model','optimizer')) and sha(p)!=item['sha256'] or identity(p)!=before:fail('frozen source file changed '+label)
  checked[label]={'path':str(p),'producer_file_sha256':item['sha256'],'identity':before,'integrity':('deferred_to_export_provenance' if a.stage=='resume' and label in ('model','optimizer') else 'verified')}
 mfconfig=Path(a.mf_config);hpconfig=Path(a.hp_config);pin(mfconfig,spec['config']['sha256']);pin(hpconfig,a.hp_config_sha256)
 if not mfconfig.resolve().is_relative_to(R.resolve()) or not hpconfig.resolve().is_relative_to(R.resolve()):fail('config outside task')
 for ext,wanted in (('.bin',a.data_bin_sha256),('.idx',a.data_idx_sha256)):
  p=Path(D+ext);pin(p,wanted)
 # Require same known source data, not a caller-selected replacement.
 if a.data_bin_sha256!=DATA['.bin'][1] or a.data_idx_sha256!=DATA['.idx'][1]:fail('data pin differs from task dataset')
 if a.hp_config_sha256!='e459f0626ac081a046eb81fdd2fda0c57eacbd72980d2d995f5d2d9f2ca2b183' or a.diag_report_sha256!='dddd5c7607ee08d6309e548f0ae882d325d5e7c2372a5f326ebf4c8635905016':fail('latest short/config pin differs')
 diag,_,_=complete_job(a.diag_report,a.diag_report_sha256,a.diag_create,a.diag_create_sha256,a.diag_terminal,a.diag_terminal_sha256,'resume')
 if diag.get('schema')!='pr310-hp-resume6-10-v1' or diag.get('status')!='complete' or diag.get('hp_head')!=HEAD or diag.get('step_start')!=5 or diag.get('step_end')!=10 or diag.get('training_updates')!=5:fail('successful HP5->10 diagnostic required')
 info={'ready_root':str(ready),'ready_sha256':a.ready_sha256,'collector_sha256':a.collector_sha256,'source_root':str(source),'source_checks':checked,'mf_config':str(mfconfig.resolve()),'mf_config_sha256':spec['config']['sha256'],'hp_config':str(hpconfig.resolve()),'hp_config_sha256':a.hp_config_sha256,'diagnostic_report':str(Path(a.diag_report).resolve()),'diag_report_sha256':a.diag_report_sha256,'diag_create_sha256':a.diag_create_sha256,'diag_terminal_sha256':a.diag_terminal_sha256,'worker_sha256':a.worker_sha256,'worker_manifest_sha256':a.worker_manifest_sha256,'worker_files':worker_files}
 if a.stage=='resume':
  root=Path(a.export_root)
  if root.is_symlink() or not root.is_dir() or not root.resolve().is_relative_to(NFS.resolve()):fail('HP1500 export root outside NFS')
  root=root.resolve(); report,_,_=complete_job(root/'report.json',a.export_report_sha256,a.export_create,a.export_create_sha256,a.export_terminal,a.export_terminal_sha256,'export')
  if report.get('schema')!='pr310-hp1500-export-v1' or report.get('status')!='complete' or report.get('hp_head')!='1825953e91989184ace5d4a37a975a18fb2aa5c2' or report.get('ready_sha256')!=a.ready_sha256 or report.get('source_step')!=1500 or report.get('training_updates')!=0:fail('HP1500 export report differs')
  index=regular_below(root,'mapped-state-index.json');pin(index,report.get('mapped_index_sha256',''))
  if len(json.loads(index.read_text()))!=report.get('tensor_count'):fail('mapped-index count differs')
  rel=report.get('checkpoint_relative_path');check=root
  if not rel or Path(rel).is_absolute() or any(x in ('','..','.') for x in rel.split('/')):fail('DCP path invalid')
  for part in rel.split('/'):
   check=check/part
   if check.is_symlink():fail('symlink DCP path')
  if not check.is_dir() or not check.resolve().is_relative_to(root):fail('DCP missing')
  found={}
  for directory,dirs,names in os.walk(check,followlinks=False):
   base=Path(directory)
   for n in dirs:
    if (base/n).is_symlink():fail('symlink in DCP')
   for n in names:
    p=base/n
    if p.is_symlink() or not p.is_file():fail('non-regular DCP')
    before=identity(p);relname=p.relative_to(check).as_posix();expected=report.get('checkpoint_files',{}).get(relname)
    if not isinstance(expected,dict) or before[2]!=expected.get('size'):fail('DCP file size/inventory differs')
    if p.suffix=='.safetensors':
     found[relname]={'size':before[2],'identity':before,'producer_file_sha256':expected['sha256'],'integrity':'deferred_to_live_state_after_native_load'}
    elif p.name in ('.metadata','_rank0_.bytes','extra_state_rank_0.pt'):
     digest=sha(p)
     if digest!=expected.get('sha256'):fail('small DCP hash differs')
     found[relname]={'size':before[2],'identity':before,'producer_file_sha256':expected['sha256'],'actual_sha256':digest,'integrity':'verified'}
    else:fail('unrecognized DCP file role')
    if identity(p)!=before:fail('DCP changed')
  if not found or set(found)!=set(report.get('checkpoint_files',{})):fail('DCP file inventory differs')
  info.update(export_root=str(root),export_create_path=str(Path(a.export_create).resolve()),export_terminal_path=str(Path(a.export_terminal).resolve()),export_report_sha256=a.export_report_sha256,export_create_sha256=a.export_create_sha256,export_terminal_sha256=a.export_terminal_sha256,export_index_sha256=report['mapped_index_sha256'],export_files=found)
 return info
def starttick(pid):
 data=Path('/proc/%s/stat'%pid).read_text(); return int(data[data.rfind(')')+2:].split()[19])
def actual_checks(obj,m,name):
 pid=obj['State']['Pid']; ident=obj['Id']; lines=Path('/proc/%s/cgroup'%pid).read_text().splitlines()
 paths=[x.split(':',2)[2] for x in lines if ':devices:' in x]
 if len(paths)!=1 or ident not in paths[0]:fail('target devices cgroup identity differs')
 base=Path('/sys/fs/cgroup/devices').resolve(); group=(base/paths[0].lstrip('/')).resolve()
 if not group.is_relative_to(base):fail('devices cgroup escaped root')
 wanted={(os.major(os.stat(x).st_rdev),os.minor(os.stat(x).st_rdev)) for x in DEVS}; cardmajor=os.major(os.stat(DEVS[0]).st_rdev); cardminor=os.minor(os.stat(DEVS[0]).st_rdev); matched=False
 for rule in (group/'devices.list').read_text().splitlines():
  kind,numbers,perms=rule.split(); maj,mn=numbers.split(':')
  if kind not in ('c','a') or not set(perms)&set('rw'):continue
  if maj=='*' or (maj.isdigit() and int(maj)==cardmajor):
   if mn=='*' or not mn.isdigit() or int(mn)!=cardminor:fail('devices cgroup permits another accelerator')
   matched=matched or set('rw').issubset(perms)
 if not matched:fail('devices cgroup lacks card1')
 ro=['/']+[d for _,d,v in m if v]; rw=[d for _,d,v in m if not v]
 code='import os,json; ro='+repr(ro)+'; rw='+repr(rw)+'; v={p:bool(os.statvfs(p).f_flag & os.ST_RDONLY) for p in ro+rw}; assert all(v[p] for p in ro) and all(not v[p] for p in rw),v; print(json.dumps(v,sort_keys=True))'
 flags=json.loads(run(['docker','exec',name,'env','-i','PYTHONDONTWRITEBYTECODE=1','/work/envs/hp-pta212/bin/python','-I','-c',code]))
 return {'mount_flags':flags,'devices_cgroup':str(group),'memory_cgroup':cg(pid)}
def persist(path,obj):
 temp=path.with_name(path.name+'.tmp-'+str(os.getpid()))
 with temp.open('x') as f:json.dump(obj,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(temp,path)
def cgroup_paths(pid):
 groups={}
 for line in Path('/proc/%s/cgroup'%pid).read_text().splitlines():
  _,controllers,path=line.split(':',2)
  for name in controllers.split(','):groups[name]=path
 if not groups.get('devices') or not groups.get('memory'):fail('missing v1 target cgroup')
 return groups
def cg_join(groups):
 # Called only in Popen child, never the supervising process.
 for name in ('devices','memory'):
  base=Path('/sys/fs/cgroup')/name; folder=(base/groups[name].lstrip('/')).resolve()
  if not folder.is_relative_to(base.resolve()):raise RuntimeError('invalid '+name+' cgroup')
  (folder/'cgroup.procs').write_text(str(os.getpid()))
def scan_workers(pidns,groups,tag):
 matches=[]
 for item in Path('/proc').iterdir():
  if not item.name.isdigit():continue
  pid=int(item.name)
  try:
   if os.readlink(item/'ns/pid')!=pidns:continue
   g=cgroup_paths(pid)
   if any(g.get(x)!=groups[x] for x in ('devices','memory')):continue
   args=(item/'cmdline').read_bytes().split(b'\0')
   if not args or b'python' not in Path(os.fsdecode(args[0])).name.encode():continue
   env=(item/'environ').read_bytes().split(b'\0')
   if ('HP_PR310_ID='+tag).encode() not in env:continue
   data=(item/'stat').read_text(); state=data[data.rfind(')')+2:].split()[0]
   if state=='Z':continue
   matches.append({'host_pid':pid,'starttick':starttick(pid)})
  except (FileNotFoundError,ProcessLookupError):continue
 return matches
def identity_alive(entry):
 try:
  p=Path('/proc')/str(entry['host_pid']); raw=(p/'stat').read_text();state=raw[raw.rfind(')')+2:].split()[0]
  return state!='Z' and starttick(entry['host_pid'])==entry['starttick']
 except (FileNotFoundError,ProcessLookupError):return False
ENTRY_CODE=r"""# hp_nsentry_proof_v1
import base64,hashlib,importlib.util,json,os,sys,time
from pathlib import Path
spec=json.loads(base64.b64decode(sys.argv[1]).decode());worker=sys.argv[2:]
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
ns={k:os.readlink('/proc/self/ns/'+k) for k in ('mnt','pid','uts','ipc','net','user')};assert ns==spec['namespaces'],(ns,spec['namespaces'])
groups={}
for line in Path('/proc/self/cgroup').read_text().splitlines():
 _,names,path=line.split(':',2)
 for name in names.split(','):groups[name]=path
assert all(groups.get(k)==spec['cgroups'][k] for k in ('devices','memory')),(groups,spec['cgroups'])
flags={p:bool(os.statvfs(p).f_flag & os.ST_RDONLY) for p in spec['ro']+spec['rw']};assert all(flags[p] for p in spec['ro']) and all(not flags[p] for p in spec['rw']),flags
for path,want in spec['files'].items():assert not Path(path).is_symlink() and sha(path)==want,path
base=Path('/work/envs/hp-pta212/lib/python3.12/site-packages');fields={}
for name in ['torch-2.12.0+cpu.dist-info','torch_npu-2.12.0.post3.dist-info','transformers-5.21.0.dist-info','tokenizers-0.23.3.dist-info','hyper_parallel-1.2.0.dev0.dist-info']:
 p=base/name/'METADATA';ver=next(x.split(':',1)[1].strip() for x in p.read_text().splitlines() if x.startswith('Version:'));fields[name]={'version':ver,'sha256':sha(p)}
for key,path in [('pta_direct_url',base/'torch_npu-2.12.0.post3.dist-info/direct_url.json'),('hp_direct_url',base/'hyper_parallel-1.2.0.dev0.dist-info/direct_url.json'),('hp_pth',base/'__editable__.hyper_parallel-1.2.0.dev0.pth'),('pyvenv',Path('/work/envs/hp-pta212/pyvenv.cfg')),('cann_env',Path('/work/runtime/cann920/cann-9.2.0/set_env.sh')),('python',Path('/usr/local/python3.12.14/bin/python3.12'))]:fields[key]=sha(path)
fp=hashlib.sha256(json.dumps(fields,sort_keys=True,separators=(',',':')).encode()).hexdigest();assert fp==spec['env_fingerprint'],(fp,spec['env_fingerprint'])
mods={}
for name in ('torch','torch_npu','transformers','hyper_parallel'):
 p=importlib.util.find_spec(name);mods[name]=None if p is None else p.origin
assert all((mods[n] or '').startswith('/work/envs/hp-pta212/') for n in ('torch','torch_npu','transformers')),mods
assert (mods['hyper_parallel'] or '').startswith('/work/sources/hyper-parallel/'),mods
status=Path('/proc/self/status').read_text().splitlines();evidence={'namespaces':ns,'cgroups':groups,'mount_flags':flags,'file_hashes':spec['files'],'env_fingerprint':fp,'python':sys.version,'executable':sys.executable,'sys_path':sys.path,'module_origins':mods,'ascend_home':os.getenv('ASCEND_HOME_PATH'),'visible':os.getenv('ASCEND_RT_VISIBLE_DEVICES'),'pid_in_namespace':os.getpid(),'seccomp':next((x for x in status if x.startswith('Seccomp:')),None),'capeff':next((x for x in status if x.startswith('CapEff:')),None),'worker_argv':worker}
path=Path(spec['evidence']);ack=Path(spec['ack'])
with path.open('x') as f:json.dump(evidence,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
for _ in range(600):
 if ack.is_file():
  a=json.loads(ack.read_text());assert a.get('host_pid') and a.get('starttick');break
 time.sleep(0.1)
else:raise RuntimeError('supervisor did not identify entry process')
os.execv(worker[0],worker)
"""
def supervise(cmd,runroot,kind,fd,name,a,m,receipt,groups,pidns,tag,ack_host):
 pid=os.fork()
 if pid:
  print(json.dumps({'supervisor_pid':pid,'starttick':starttick(pid),'status_file':str(runroot/(kind+'.supervisor.json'))}),flush=True)
  try:_,status=os.waitpid(pid,0);return os.waitstatus_to_exitcode(status)
  except KeyboardInterrupt:print('supervisor remains detached and retains lock',file=sys.stderr);return 130
 os.setsid();signal.signal(signal.SIGHUP,signal.SIG_IGN);signal.signal(signal.SIGINT,signal.SIG_IGN);signal.signal(signal.SIGTERM,signal.SIG_IGN)
 child=None; tracked={};status_path=runroot/(kind+'.supervisor.json');status={'supervisor_pid':os.getpid(),'supervisor_starttick':starttick(os.getpid()),'container_id':receipt['id'],'state':'preparing','argv':cmd,'execution_boundary':'host mount namespace; target pid/uts/ipc/net and devices/memory cgroups; other host cgroups retained'}
 try:
  persist(status_path,status);inputs=formal_inputs(a);info=guard(a);obj=inspect(name);verify(obj,m)
  if obj['Id']!=receipt['id'] or obj['State']['Pid']!=receipt['pid'] or info['env_fingerprint']!=receipt['pins']['env_fingerprint'] or inputs!=receipt['pins']['inputs']:fail('target, inputs or environment changed')
  status['container_checks']=actual_checks(obj,m,name);status['state']='starting';persist(status_path,status)
  with (runroot/(kind+'.stdout.log')).open('x') as out,(runroot/(kind+'.stderr.log')).open('x') as err:
   child=subprocess.Popen(cmd,stdout=out,stderr=err,start_new_session=True,preexec_fn=lambda:cg_join(groups))
   status.update(state='running',entry_pid=child.pid,entry_starttick=starttick(child.pid));persist(status_path,status)
   while child.poll() is None:
    found=scan_workers(pidns,groups,tag)
    for ent in found:tracked[(ent['host_pid'],ent['starttick'])]=ent
    if found and not ack_host.exists():
     if len(found)!=1:fail('ambiguous entry processes')
     with ack_host.open('x') as f:json.dump(found[0],f);f.flush();os.fsync(f.fileno())
     status['worker_processes']=list(tracked.values());persist(status_path,status)
    time.sleep(0.2)
   rc=child.wait()
  while True:
   found=scan_workers(pidns,groups,tag)
   for ent in found:tracked[(ent['host_pid'],ent['starttick'])]=ent
   alive=[e for e in tracked.values() if identity_alive(e)]
   proc=run(['npu-smi','info','-t','proc-mem','-i','1','-c','0'])
   if not found and not alive and 'No process in device.' in proc:break
   status.update(state='waiting_for_exact_worker_exit_and_card_idle',worker_processes=list(tracked.values()),entry_exit=rc);persist(status_path,status);time.sleep(5)
  if rc==0 and (not tracked or not ack_host.is_file()):fail('successful entry lacked worker identity handshake')
  status.update(state='complete',exit_code=rc,worker_processes=list(tracked.values()));persist(status_path,status);os._exit(rc if 0<=rc<=255 else 1)
 except BaseException as exc:
  # Observation failures keep the lock until the entry and all identified workers
  # have exited and a fresh scan plus card idle succeeds.
  if child is not None:
   while True:
    try:
     found=scan_workers(pidns,groups,tag)
     for ent in found:tracked[(ent['host_pid'],ent['starttick'])]=ent
     if child.poll() is not None and not found and not any(identity_alive(e) for e in tracked.values()) and 'No process in device.' in run(['npu-smi','info','-t','proc-mem','-i','1','-c','0']):break
    except Exception:pass
    time.sleep(5)
  try:status.update(state='failed_before_or_during_launch',error=str(exc),worker_processes=list(tracked.values()));persist(status_path,status)
  finally:os._exit(1)
def guard(a):
 if os.geteuid()!=0:fail('host root required')
 if NFS.is_symlink() or (NFS.stat().st_dev,NFS.stat().st_ino)!=(1048588,191983715):fail('NFS parent identity differs')
 if run(['findmnt','-n','-o','FSTYPE','--target',str(NFS)]).strip() not in ('nfs','nfs4'):fail('not NFS')
 if run(['findmnt','-n','-o','FSTYPE','--target','/tmp']).strip()!='tmpfs':fail('tmp not tmpfs')
 for p,need in [(NFS,86*GiB),(Path('/tmp'),40*GiB),(Path('/home'),GiB)]:
  v=os.statvfs(p)
  if v.f_bavail*v.f_frsize<need:fail('space guard '+str(p))
 mem={x.split(':')[0]:int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith(('MemAvailable:','MemTotal:'))}
 if mem['MemAvailable']<96*GiB:fail('host memory low')
 cg(os.getpid()); port(a.master_port)
 if 'No process in device.' not in run(['npu-smi','info','-t','proc-mem','-i','1','-c','0']):fail('card1 busy')
 u=run(['npu-smi','info','-t','usages','-i','1','-c','0']); h=re.search(r'HBM Usage Rate\(%\)\s*:\s*(\d+)',u); c=re.search(r'Aicore Usage Rate\(%\)\s*:\s*(\d+)',u)
 if not h or not c or int(h[1])>8 or int(c[1])!=0:fail('card1 usage not idle')
 for x in DEVS:
  if not stat.S_ISCHR(os.stat(x).st_mode):fail('not character device '+x)
 if run(['git','--no-optional-locks','-C',str(SRC),'rev-parse','HEAD']).strip()!=HEAD or run(['git','--no-optional-locks','-C',str(SRC),'status','--porcelain']).strip():fail('source head/clean differs')
 for p,h in PINS.items():pin(SRC/p,h)
 pin(CANN/'cann-9.2.0/set_env.sh','c7a39560b70e554c6d514db176292412dcad242653325d836e2374dc35fb310c')
 pin(PY/'bin/python3.12','397fa1fc22a59692cc873425906b552ff1ced6dcc12da0c77c3e219e8abd92d3')
 pin(VENV/'pyvenv.cfg','88fa8778ba3a165fe21a8f2328e135f3c92e89bee80c5adf0e2f7cfa2a37da74')
 base=VENV/'lib/python3.12/site-packages'
 pin(base/'torch-2.12.0+cpu.dist-info/METADATA','4dea4f71028daacd7f5b67b9992c243a2cefabbadb8566316a8c2e5af0c86228')
 pin(base/'torch_npu-2.12.0.post3.dist-info/METADATA','ebb2e0f775ea9b7730ff34376e85d8a019f10381763ea6832cd40856bca85483')
 pin(base/'transformers-5.21.0.dist-info/METADATA','dc0a9fb777810c1eee9318798f69ea246f2e6de3a81ef09bcf18adf605ffc03c')
 pin(base/'__editable__.hyper_parallel-1.2.0.dev0.pth','1789fa1f2d88f64d1d618aa6a39b62e139c8ee9812c89e00355d176247d3f200')
 if 'include-system-site-packages = true' not in (VENV/'pyvenv.cfg').read_text().lower():fail('candidate venv cfg differs')
 direct=json.loads((base/'torch_npu-2.12.0.post3.dist-info/direct_url.json').read_text())
 if direct.get('archive_info',{}).get('hashes',{}).get('sha256')!='d553e741081ca9cbc672175281b486c9c3b06676928590087f9afa70f0bceb6b':fail('PTA wheel identity differs')
 hp=json.loads((base/'hyper_parallel-1.2.0.dev0.dist-info/direct_url.json').read_text())
 if hp.get('url')!='file:///work/sources/hyper-parallel' or not hp.get('dir_info',{}).get('editable'):fail('HP editable identity differs')
 for ext,(sz,digest) in DATA.items():
  p=Path(D+ext); before=p.stat()
  if p.is_symlink() or before.st_size!=sz or sha(p)!=digest or (lambda t:(t.st_dev,t.st_ino,t.st_size,t.st_mtime_ns,t.st_ctime_ns))(p.stat())!=(before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns):fail('dataset differs '+ext)
 img=json.loads(run(['docker','image','inspect',IMAGE]))[0]
 if img['Id']!=IMAGE or img['Architecture']!='arm64' or img['Config'].get('Volumes'):fail('image differs')
 return {'source_head':HEAD,'card1_hbm':int(h[1]),'card1_aicore':int(c[1]),'host_memory':mem,'launcher_cgroup':cg(os.getpid()),'env_fingerprint':env_fingerprint()}
def lock():
 if TMP.is_symlink() or TMP.stat().st_uid!=0:fail('bad lock parent')
 fd=os.open(LOCK,os.O_RDWR|os.O_NOFOLLOW|os.O_CLOEXEC); st=os.fstat(fd)
 if not stat.S_ISREG(st.st_mode) or st.st_uid!=0 or st.st_nlink!=1 or stat.S_IMODE(st.st_mode)&0o077:fail('bad shared lock')
 fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);return fd
def mounts_base(runroot,work):
 return [(PY,'/usr/local/python3.12.14',1),(VENV,'/work/envs/hp-pta212',1),(SRC,'/work/sources/hyper-parallel',1),(S,'/work/scripts',1),(CANN,'/work/runtime/cann920',1),(DR,'/usr/local/Ascend/driver',1),(Path(D+'.bin'),'/datasets/mmap_deepseekv3_datasets_text_document.bin',1),(Path(D+'.idx'),'/datasets/mmap_deepseekv3_datasets_text_document.idx',1),(runroot,'/work/runs/netbase-hp2500',0),(work,'/work/private',0)]
def mounts(runroot,work,inputs):
 result=mounts_base(runroot,work)+[(Path(inputs['ready_root']),'/work/inputs/mf-ready',1),(Path(inputs['source_root']),'/work/inputs/mf-source',1),(Path(inputs['mf_config']),'/work/inputs/mf-config.yaml',1),(Path(inputs['hp_config']),'/work/inputs/hp-config.yaml',1),(Path(inputs['diagnostic_report']),'/work/inputs/hp5-report.json',1)]
 if 'export_root' in inputs:result += [(Path(inputs['export_root']),'/work/inputs/hp-export',1),(Path(inputs['export_create_path']),'/work/inputs/hp-export-create.json',1),(Path(inputs['export_terminal_path']),'/work/inputs/hp-export-supervisor.json',1)]
 return result
def inspect(name):return json.loads(run(['docker','inspect',name]))[0]
def verify(obj,m):
 if obj['Image']!=IMAGE or not obj['State']['Running'] or (obj['Config'].get('Labels') or {}).get('codex.task')!='hp-mf-pr310-hp2500':fail('target image/state/label differs')
 host=obj['HostConfig']; ds=host['Devices'] or []
 if host['Privileged'] or not host['ReadonlyRootfs'] or host['NetworkMode']!='host' or {(x['PathOnHost'],x['PathInContainer']) for x in ds}!={(x,x) for x in DEVS} or host.get('DeviceCgroupRules'):fail('target isolation differs')
 actual={x['Destination']:x for x in obj['Mounts'] if x['Type']=='bind'}
 if set(actual)!={d for _,d,_ in m}:fail('target mounts differ')
 for s,d,ro in m:
  if os.path.realpath(actual[d]['Source'])!=os.path.realpath(s) or actual[d]['RW']==bool(ro):fail('mount differs '+d)
 cg(obj['State']['Pid'])
def main():
 p=argparse.ArgumentParser(); g=p.add_mutually_exclusive_group(required=True)
 for x in ('check','create','execute'):g.add_argument('--'+x,action='store_true')
 p.add_argument('--receipt-sha256');p.add_argument('--container-id');p.add_argument('--container-pid',type=int);p.add_argument('--run-id',required=True);p.add_argument('--master-port',type=int);p.add_argument('--stage',choices=['export','resume'],required=True);p.add_argument('--worker-sha256',required=True);p.add_argument('--worker-manifest',required=True);p.add_argument('--worker-manifest-sha256',required=True);p.add_argument('--ready-dir',required=True);p.add_argument('--ready-sha256',required=True);p.add_argument('--collector',required=True);p.add_argument('--collector-sha256',required=True);p.add_argument('--mf-config',required=True);p.add_argument('--hp-config',required=True);p.add_argument('--hp-config-sha256',required=True);p.add_argument('--data-bin-sha256',required=True);p.add_argument('--data-idx-sha256',required=True);p.add_argument('--diag-report',required=True);p.add_argument('--diag-report-sha256',required=True);p.add_argument('--diag-create',required=True);p.add_argument('--diag-create-sha256',required=True);p.add_argument('--diag-terminal',required=True);p.add_argument('--diag-terminal-sha256',required=True);p.add_argument('--export-root');p.add_argument('--export-report-sha256');p.add_argument('--export-create');p.add_argument('--export-create-sha256');p.add_argument('--export-terminal');p.add_argument('--export-terminal-sha256');a=p.parse_args();a.mode=a.stage;a.master_port=a.master_port or {'export':56353,'resume':56375}[a.stage]
 if a.stage!='resume':p.error('this independent draft is resume-only')
 if not re.fullmatch('[a-z0-9][a-z0-9-]{0,39}',a.run_id) or not 1024<=a.master_port<=65535:p.error('invalid run-id/port')
 runroot=NFS/('hp2500-formal-'+a.run_id); work=Path('/tmp')/('hp-mf-pr310-hp2500-formal-'+a.run_id); name='hp-mf-pr310-hp2500-formal-'+a.run_id; fd=None
 try:
  if a.create or a.execute:fd=lock()
  inputs=formal_inputs(a);m=mounts(runroot,work,inputs);info=guard(a);info['inputs']=inputs
  if a.check:
   print(json.dumps({'status':'read-only','pins':info,'run_exists':runroot.exists(),'tmp_exists':work.exists(),'container':name},indent=2));return 0
  if a.create:
   if runroot.exists() or runroot.is_symlink() or work.exists() or work.is_symlink():fail('output already exists')
   # Docker name collision checked before mkdir and rechecked by Docker.
   names=run(['docker','ps','-a','--format','{{.Names}}']).splitlines()
   if name in names:fail('container exists')
   runroot.mkdir(mode=0o700);work.mkdir(mode=0o700)
   for sub in ('home','tmp','cache','ascend-log'):(work/sub).mkdir(mode=0o700)
   cmd=['docker','run','-d','--name',name,'--read-only','--network','host','--shm-size','16g','--tmpfs','/tmp:rw,nosuid,nodev,size=1g','--tmpfs','/var/tmp:rw,nosuid,nodev,size=1g','--label','codex.task=hp-mf-pr310-hp2500','--label','codex.mode='+a.mode]
   for d in DEVS:cmd+=['--device',d]
   for s,d,ro in m:cmd+=['--mount',f'type=bind,src={s},dst={d}'+(',readonly' if ro else '')]
   cmd+=['--env','ASCEND_RT_VISIBLE_DEVICES=0','--entrypoint','/bin/sleep',IMAGE,'infinity']; cid=run(cmd).strip();obj=inspect(name);verify(obj,m); actual_checks(obj,m,name)
   receipt={'id':cid,'pid':obj['State']['Pid'],'mode':a.mode,'port':a.master_port,'run_id':a.run_id,'pins':info,'argv':cmd,'run_dev_ino':[runroot.stat().st_dev,runroot.stat().st_ino]}
   with (runroot/'create-receipt.json').open('x') as f:json.dump(receipt,f,indent=2)
   print(json.dumps({'receipt':receipt,'receipt_sha256':sha(runroot/'create-receipt.json')},indent=2));return 0
  receipt_path=runroot/'create-receipt.json'
  if not re.fullmatch('[0-9a-f]{64}',a.receipt_sha256 or '') or not re.fullmatch('[0-9a-f]{64}',a.container_id or '') or not a.container_pid or sha(receipt_path)!=a.receipt_sha256:fail('exact external receipt/container pins required')
  receipt=json.loads(receipt_path.read_text());obj=inspect(name);verify(obj,m)
  if receipt['id']!=a.container_id or receipt['pid']!=a.container_pid or receipt['pins']['env_fingerprint']!=info['env_fingerprint'] or receipt['pins']['inputs']!=inputs:fail('external container, inputs or environment pin differs')
  actual_checks(obj,m,name)
  if receipt['id']!=obj['Id'] or receipt['pid']!=obj['State']['Pid'] or receipt['mode']!=a.mode or receipt['port']!=a.master_port or receipt['run_dev_ino']!=[runroot.stat().st_dev,runroot.stat().st_ino]:fail('creation identity mismatch')
  if {x.name for x in runroot.iterdir()}!={'create-receipt.json'}:fail('output directory contains other files; no reuse')
  marker=runroot/'execute.started'
  with marker.open('x') as f:f.write('one-shot; no retry\n')
  env=['HOME=/work/private/home','TMPDIR=/work/private/tmp','XDG_CACHE_HOME=/work/private/cache','ASCEND_PROCESS_LOG_PATH=/work/private/ascend-log','HYPER_PARALLEL_PLATFORM=torch','PYTHONPATH=/work/scripts','PATH=/work/envs/hp-pta212/bin:/usr/local/python3.12.14/bin:/usr/bin:/bin','PYTHONDONTWRITEBYTECODE=1','ASCEND_RT_VISIBLE_DEVICES=0','DEVICE_ID=0','RANK=0','LOCAL_RANK=0','WORLD_SIZE=1','RANK_SIZE=1','MASTER_ADDR=127.0.0.1','MASTER_PORT='+str(a.master_port),'OMP_NUM_THREADS=8','MKL_NUM_THREADS=8','MAX_JOBS=8']
  worker=['/work/envs/hp-pta212/bin/python','-B','/work/scripts/pr310_formal_candidate_v4/worker.py','--stage',a.stage,'--ready-dir','/work/inputs/mf-ready','--ready-sha256',a.ready_sha256,'--collector-sha256',a.collector_sha256,'--source-root','/work/inputs/mf-source','--mf-config','/work/inputs/mf-config.yaml','--data-bin','/datasets/mmap_deepseekv3_datasets_text_document.bin','--data-bin-sha256',a.data_bin_sha256,'--data-idx','/datasets/mmap_deepseekv3_datasets_text_document.idx','--data-idx-sha256',a.data_idx_sha256,'--hp-config','/work/inputs/hp-config.yaml','--hp-config-sha256',a.hp_config_sha256,'--diagnostic-report','/work/inputs/hp5-report.json','--diagnostic-report-sha256',a.diag_report_sha256,'--output-root','/work/runs/netbase-hp2500','--device-id','0']
  if a.stage=='resume':worker+=['--export-root','/work/inputs/hp-export','--export-report-sha256',a.export_report_sha256,'--export-create-receipt','/work/inputs/hp-export-create.json','--export-create-sha256',a.export_create_sha256,'--export-supervisor','/work/inputs/hp-export-supervisor.json','--export-supervisor-sha256',a.export_terminal_sha256]
  kind='execute';target=obj['State']['Pid'];groups=cgroup_paths(target)
  if receipt['id'] not in groups['devices'] or receipt['id'] not in groups['memory']:fail('target cgroup identity differs')
  target_ns={k:os.readlink('/proc/%s/ns/%s'%(target,k)) for k in ('pid','uts','ipc','net','user')}
  target_ns['mnt']=os.readlink('/proc/self/ns/mnt');pidns=target_ns['pid'];tag=a.run_id+':'+kind
  env.append('HP_PR310_ID='+tag)
  files={'/work/sources/hyper-parallel/'+k:v for k,v in PINS.items()};files.update({'/work/scripts/'+k:v for k,v in inputs['worker_files'].items()});files.update({'/work/inputs/mf-ready/ready-spec.json':a.ready_sha256,'/work/inputs/mf-config.yaml':inputs['mf_config_sha256'],'/work/inputs/hp-config.yaml':a.hp_config_sha256,'/work/inputs/hp5-report.json':a.diag_report_sha256})
  if a.stage=='resume':
   files['/work/inputs/hp-export/report.json']=a.export_report_sha256;files['/work/inputs/hp-export/mapped-state-index.json']=inputs['export_index_sha256'];files['/work/inputs/hp-export-create.json']=a.export_create_sha256;files['/work/inputs/hp-export-supervisor.json']=a.export_terminal_sha256
  spec={'namespaces':target_ns,'cgroups':groups,'ro':[d for _,d,ro in m if ro],'rw':[d for _,d,ro in m if not ro],'files':files,'env_fingerprint':receipt['pins']['env_fingerprint']['sha256'],'evidence':'/work/private/'+kind+'.entry-evidence.json','ack':'/work/private/'+kind+'.entry-ack.json'}
  encoded=base64.b64encode(json.dumps(spec,sort_keys=True).encode()).decode();ack_host=work/(kind+'.entry-ack.json')
  if ack_host.exists() or (work/(kind+'.entry-evidence.json')).exists():fail('entry evidence already exists')
  cmd=['nsenter','-t',str(target),'--root','--pid','--uts','--ipc','--net','--wd','--','/usr/bin/env','-i',*env,'bash','--noprofile','--norc','-c','source /work/runtime/cann920/cann/set_env.sh && cd /work/runs/netbase-hp2500 && exec "$@"','hp2500','/work/envs/hp-pta212/bin/python','-I','-c',ENTRY_CODE,encoded,*worker]
  print(json.dumps({'entry':'host mount ns; target pid/uts/ipc/net, devices/memory cgroups','physical':1,'logical':0,'target_cgroups':groups,'target_memory':cg(target),'entry_argv':cmd}),flush=True)
  return supervise(cmd,runroot,kind,fd,name,a,m,receipt,groups,pidns,tag,ack_host)
 except Exception as e:print(json.dumps({'status':'failed','error':str(e),'note':'no automatic cleanup or retry'}),file=sys.stderr);return 1
 finally:
  if fd is not None:os.close(fd)
if __name__=='__main__':sys.exit(main())
