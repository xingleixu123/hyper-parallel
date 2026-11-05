"""Reviewed manifest gated DCP verifier; no checkpoint writes, one global tensor resident."""
import argparse,hashlib,json,math,tempfile
from pathlib import Path
import torch,torch.distributed as dist
from hyper_parallel.core.distributed_checkpoint.api import load
from hyper_parallel.core.distributed_checkpoint.filesystem_storage import FileSystemReader
from hyper_parallel.core.distributed_checkpoint.metadata import TensorStorageMetadata,BytesStorageMetadata
from hyper_parallel.core.distributed_checkpoint.utils import str_to_dtype
from local_hash import expected_local

def pin(p,sha):
 p=Path(p)
 if p.is_symlink() or not p.is_file():raise ValueError('unsafe input '+str(p))
 a=p.stat();raw=p.read_bytes();z=p.stat()
 if (a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns,a.st_ctime_ns)!=(z.st_dev,z.st_ino,z.st_size,z.st_mtime_ns,z.st_ctime_ns) or hashlib.sha256(raw).hexdigest()!=sha:raise ValueError('identity mismatch '+str(p))
 return json.loads(raw)
def identity(root):
 rows={}
 for p in root.rglob('*'):
  if p.is_symlink():raise ValueError('symlink in checkpoint')
  if p.is_file():
   s=p.stat();rows[str(p.relative_to(root))]=[s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
 return rows

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--spec',required=True);ap.add_argument('--spec-sha256',required=True);ap.add_argument('--max-tensor-bytes',type=int,required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
 spec=pin(a.spec,a.spec_sha256);ck=Path(spec['checkpoint']);out=Path(a.output)
 if out.exists() or a.max_tensor_bytes<=0 or ck.is_symlink():raise ValueError('unsafe output/checkpoint')
 if spec.get('source_step')!=5 or spec.get('world_size')!=4:raise ValueError('expected four-rank step5 checkpoint')
 if not ck.is_absolute() or not ck.is_dir() or ck.resolve()!=ck:raise ValueError('checkpoint path must be canonical')
 before=identity(ck)
 if '.metadata' not in before:raise ValueError('metadata absent')
 expected=spec.get('checkpoint_sha256')
 if not isinstance(expected,dict) or set(expected)!=set(before):raise ValueError('checkpoint file hash coverage incomplete')
 for name,want in expected.items():
  h=hashlib.sha256()
  with (ck/name).open('rb') as f:
   for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
  if h.hexdigest()!=want:raise ValueError('checkpoint hash mismatch '+name)
 if set(spec['target_rank_proofs'])!=set(map(str,range(4))):raise ValueError('four target proofs required')
 proofs={int(r):pin(v['path'],v['sha256']) for r,v in spec['target_rank_proofs'].items()}
 for r,p in proofs.items():
  if not isinstance(p,dict) or not {'model','slots','groups','main_aliases'}.issubset(p):raise ValueError('invalid local state proof')
 mapping=spec['tensors']; keys=[x['dcp_key'] for x in mapping]
 if len(keys)!=len(set(keys)):raise ValueError('each DCP tensor key exactly once; multiple target bindings per row')
 with tempfile.TemporaryDirectory() as tmp:
  dist.init_process_group('gloo',init_method='file://'+tmp+'/pg',rank=0,world_size=1)
  try:
   meta=FileSystemReader(ck).load_metadata().state_dict_metadata
   tensor_keys={k for k,v in meta.items() if isinstance(v,TensorStorageMetadata)}; byte_keys={k for k,v in meta.items() if isinstance(v,BytesStorageMetadata)}
   if tensor_keys!=set(keys):raise ValueError('DCP tensor key coverage incomplete')
   if byte_keys!=set(spec['optimizer_bytes']):raise ValueError('DCP byte key coverage incomplete')
   results=[];covered={r:set() for r in proofs}
   for row in mapping:
    key=row['dcp_key'];m=meta[key];dt=str_to_dtype(m.properties.dtype);n=math.prod(m.size)*torch.empty((),dtype=dt).element_size()
    if n>a.max_tensor_bytes:raise ValueError('single tensor memory budget exceeded: '+key)
    value=torch.empty(m.size,dtype=dt,device='cpu');load({key:value},checkpoint_id=ck,use_collectives=False)
    if key in ('model.model.layers.1.mlp.gate._token_counts', 'model.mtp.layers.0.mtp_block.mlp.gate._token_counts'):
     if value.dtype != torch.float32 or not torch.equal(value, value.round()):raise ValueError('invalid legacy count values '+key)
     converted=value.to(torch.int64)
     if not torch.equal(converted.to(torch.float32),value):raise ValueError('inexact legacy count conversion '+key)
     value=converted
    for binding in row['targets']:
     section=binding['section'];name=binding['name'];slot=binding.get('slot')
     if section not in ('model','slots','main'):raise ValueError('unknown section')
     for rank,p in proofs.items():
      state=p; rec=(state['slots'][name][slot] if section=='slots' else state['main_aliases'][name]['main'] if section=='main' else state['model'][name]); rec=dict(rec,global_shape=rec['shape']); item=(section,name,slot)
      if item in covered[rank]:raise ValueError('duplicate binding')
      covered[rank].add(item);results.append({'key':key,'section':section,'name':name,'slot':slot,**expected_local(value,rec,rank,plain_full=binding.get('plain_full',False))})
    del value
   for rank,p in proofs.items():
    state=p; required={('model',n,None) for n in state['model']} | {('slots',n,s) for n,slots in state['slots'].items() for s in slots}
    for name,alias in state['main_aliases'].items():
     if alias['same_storage']:
      main=alias['main']; model=state['model'][name]
      if any(main[k]!=model[k] for k in ('shape','local_shape','dtype','sha256','placements','mesh')):raise ValueError('main/model alias descriptors or values differ '+name)
     else:required.add(('main',name,None))
    if required!=covered[rank]:raise ValueError('target model/slots/nonalias main coverage mismatch')
   for key,decl in spec['optimizer_bytes'].items():
    state={key:None};load(state,checkpoint_id=ck,use_collectives=False)
    for rank,p in proofs.items():
     group=p['groups'][decl['group_index']];target=group['parameters'] if decl['field']=='parameters' else group['options'][decl['field']]
     if json.loads(json.dumps(state[key]))!=target:raise ValueError('optimizer group mismatch '+key)
  finally:dist.destroy_process_group()
 if identity(ck)!=before:raise ValueError('checkpoint changed while read')
 if pin(a.spec,a.spec_sha256)!=spec:raise ValueError('spec changed')
 out.write_text(json.dumps({'status':'passed','source_step':5,'rows':results,'spec_sha256':a.spec_sha256},indent=2)+'\n')
if __name__=='__main__':main()
