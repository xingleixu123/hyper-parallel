"""Create a pinned resume spec from a completed producer. Read only; does not launch."""
import argparse,hashlib,json,re
from pathlib import Path

def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def root(raw):
 p=Path(raw)
 if not p.is_absolute() or p.is_symlink() or p.resolve()!=p or not p.is_dir():raise ValueError('path must be a canonical directory')
 return p
p=argparse.ArgumentParser();p.add_argument('--baseline-spec',required=True);p.add_argument('--baseline-output',required=True);p.add_argument('--scripts',required=True);p.add_argument('--output',required=True);p.add_argument('--run-id',required=True);p.add_argument('--master-port',type=int,required=True);p.add_argument('--hccl-start',type=int,required=True);a=p.parse_args()
if not re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,63}',a.run_id):raise ValueError('unsafe run id')
scripts=root(a.scripts);source=root(a.baseline_output);ck=root(str(source/'checkpoints/global_step_5'));out=Path(a.output)
if out.exists():raise ValueError('output already exists')
state=json.loads((source/'controller.json').read_text())
if state.get('status')!='finished_process_exit_0' or state.get('returncode')!=0:raise ValueError('baseline controller not complete')
files={}
for f in ck.rglob('*'):
 if f.is_symlink():raise ValueError('checkpoint symlink')
 if f.is_file():files[str(f.relative_to(ck))]=digest(f)
if '.metadata' not in files or not all('extra_state/extra_state_rank_'+str(i)+'.pt' in files for i in range(4)):raise ValueError('checkpoint incomplete')
refs={str(i):digest(source/('rank-'+str(i))/'checkpoint-5-live-state.json') for i in range(4)}
c=json.loads(Path(a.baseline_spec).read_text());c.update(run_id=a.run_id,mode='resume',stop=105,master_port=a.master_port,hccl_start=a.hccl_start,checkpoint_path=str(ck),reference_path=str(source),checkpoint_sha256=files,reference_sha256=refs,entry_sha256=digest(scripts/'resume-fourcard/entry.py'));c['paths']['scripts']=str(scripts);c['source_manifest']=str(scripts/'source-manifest.json');c['source_manifest_sha256']=digest(scripts/'source-manifest.json')
with out.open('x') as f:json.dump(c,f,indent=2);f.write('\n')
print(json.dumps({'spec':str(out),'sha256':digest(out),'checkpoint_files':len(files)}))
