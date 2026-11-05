"""Candidate loader: tied alias and real HP DTensor, rank-one CPU Gloo."""
import hashlib, json, tempfile, os
from pathlib import Path
import torch
import torch.distributed as dist
from torch import nn
from transformers import PreTrainedModel,PretrainedConfig
from safetensors.torch import save_file
from hyper_parallel import DTensor, init_device_mesh
from hyper_parallel.core.dtensor.placement_types import Shard
from hyper_parallel.models._transformers.checkpoint_loader import load_pretrained_weights
from hyper_parallel.models._transformers import checkpoint_loader
print('SOURCE',checkpoint_loader.__file__,hashlib.sha256(Path(checkpoint_loader.__file__).read_bytes()).hexdigest(),flush=True)
assert hashlib.sha256(Path(checkpoint_loader.__file__).read_bytes()).hexdigest()=='8f623e58927b1923fecf92330fc3a0bc414e1a5e8d37d9111051b18a8f463385'
class Tied(PreTrainedModel):
 config_class=PretrainedConfig
 def __init__(self):
  super().__init__(PretrainedConfig());self.src=nn.Linear(2,2,bias=False);self.dst=nn.Linear(2,2,bias=False)
  self.dst.weight=self.src.weight;self.all_tied_weights_keys={'dst.weight':'src.weight'};self.init_calls=0
 def _init_weights(self,module):
  if isinstance(module,nn.Linear):self.init_calls+=1;nn.init.constant_(module.weight,99.)
class DistModel(PreTrainedModel):
 config_class=PretrainedConfig
 _keys_to_ignore_on_load_missing=[r'^ignored$']
 def __init__(self,mesh):
  super().__init__(PretrainedConfig());self.ignored=nn.Parameter(DTensor.from_local(torch.ones(2),mesh,(Shard(0),)))
  self.other=nn.Parameter(torch.zeros(2));self.init_calls=0
 def _init_weights(self,module):
  if module is self:self.init_calls+=1;nn.init.zeros_(module.ignored);nn.init.zeros_(module.other)
print('torch',torch.__version__,'transformers',__import__('transformers').__version__,flush=True)
with tempfile.TemporaryDirectory(prefix='loader-boundary-') as td:
 rendezvous=Path(td)/'gloo_init';dist.init_process_group('gloo',init_method='file://'+str(rendezvous),rank=0,world_size=1)
 try:
  mesh=init_device_mesh('cpu',(1,),mesh_dim_names=('dp',))
  for loader in ('legacy','dcp'):
   os.environ['HYPER_PARALLEL_HF_LOADER']=loader
   path=Path(td)/(loader+'-tied.safetensors');save_file({'src.weight':torch.full((2,2),7.)},str(path),metadata={'format':'pt'})
   model=Tied();report=load_pretrained_weights(model,path,strict=True)
   assert model.dst.weight is model.src.weight and torch.equal(model.src.weight,torch.full((2,2),7.))
   assert model.init_calls==0 and not report.missing_keys,(model.init_calls,report)
   print(loader,'TIED PASS: missing dst alias resolved; shared value=7; init_calls=0; missing=',report.missing_keys,flush=True)
   path=Path(td)/(loader+'-dtensor.safetensors');save_file({'other':torch.full((2,),8.)},str(path),metadata={'format':'pt'})
   model=DistModel(mesh);assert isinstance(model.ignored,DTensor)
   try:load_pretrained_weights(model,path,strict=True)
   except ValueError as exc:
    assert 'DTensor-aware initialization' in str(exc) and 'ignored' in str(exc),str(exc)
    assert model.init_calls==0 and torch.equal(model.other,torch.full((2,),8.))
    print(loader,'DTENSOR PASS: real HP DTensor rejected before initialization;',str(exc),flush=True)
   else:raise AssertionError('ignored DTensor unexpectedly accepted')
 finally:dist.destroy_process_group()
print('PASS both boundaries; CPU Gloo world_size=1',flush=True)
