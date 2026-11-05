"""Real HF model/safetensors probe against independent loader files; CPU only."""
import argparse, hashlib, importlib.util, json, os, sys, tempfile
from pathlib import Path
import torch
from torch import nn
from transformers import PreTrainedModel, PretrainedConfig
from safetensors.torch import save_file
from hyper_parallel.models.materialization import (register_rebuildable_buffer, rebuild_materialized_state,
                                                    MaterializationContext)

parser=argparse.ArgumentParser(); parser.add_argument('--version',choices=('before','after'),required=True); args=parser.parse_args()
base=Path(__file__).resolve().parent
source=base / f'checkpoint_loader-{args.version}.py'
expected=json.loads((base/'loader-candidate-binding.json').read_text())[args.version]
hash_value=hashlib.sha256(source.read_bytes()).hexdigest(); assert hash_value==expected
spec=importlib.util.spec_from_file_location('hf_loader_candidate_'+args.version,source)
mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
class Block(nn.Module):
 def __init__(self):
  super().__init__();self.weight=nn.Parameter(torch.empty(2,2));self.ignored_param=nn.Parameter(torch.empty(2))
  self.register_buffer('loaded_buf',torch.empty(2));self.register_buffer('ignored_buf',torch.empty(2))
  register_rebuildable_buffer(self,'scratch',value=torch.full((2,),17.,device='cpu'))
class Model(PreTrainedModel):
 config_class=PretrainedConfig
 _keys_to_ignore_on_load_missing=[r'^block\.ignored_param$',r'^block\.ignored_buf$']
 def __init__(self):super().__init__(PretrainedConfig());self.block=Block()
 def _init_weights(self,module):
  if isinstance(module,Block):
   nn.init.constant_(module.weight,101.);nn.init.constant_(module.loaded_buf,102.)
   nn.init.constant_(module.ignored_param,3.);nn.init.constant_(module.ignored_buf,5.)

def setup():
 with torch.device('meta'): model=Model()
 assert model.block.weight.is_meta
 model.to_empty(device='cpu')
 with torch.no_grad():
  for parameter in model.parameters(): parameter.fill_(71.)
  for buf in model.buffers(): buf.fill_(71.)
 rebuild_materialized_state(model,MaterializationContext(reason='checkpoint_load',device=torch.device('cpu')))
 assert torch.equal(model.block.scratch,torch.full((2,),17.))
 return model

def run(loader,case):
 os.environ['HYPER_PARALLEL_HF_LOADER']=loader
 m=setup(); state={'block.weight':torch.full((2,2),7.),'block.loaded_buf':torch.full((2,),8.),
                    'block.ignored_param':torch.full((2,),9.),'block.ignored_buf':torch.full((2,),10.)}
 if case=='ignored': state.pop('block.ignored_param');state.pop('block.ignored_buf')
 if case=='strict_other': state.pop('block.weight')
 with tempfile.TemporaryDirectory(prefix='hf-loader-regression-') as td:
  path=Path(td)/'model.safetensors';save_file(state,str(path),metadata={'format':'pt'})
  if case=='strict_other':
   try:mod.load_pretrained_weights(m,path,strict=True)
   except RuntimeError as e:
    assert 'block.weight' in str(e),str(e);print(loader,case,'PASS rejected unresolved missing');return
   raise AssertionError('strict missing unexpectedly succeeded')
  report=mod.load_pretrained_weights(m,path,strict=True)
 assert report.missing_keys==(),report
 assert torch.equal(m.block.weight,torch.full((2,2),7.)),m.block.weight
 assert torch.equal(m.block.loaded_buf,torch.full((2,),8.)),m.block.loaded_buf
 assert torch.equal(m.block.scratch,torch.full((2,),17.)),m.block.scratch
 if case=='all_present':
  assert torch.equal(m.block.ignored_param,torch.full((2,),9.))
  assert torch.equal(m.block.ignored_buf,torch.full((2,),10.))
 else:
  p,b=m.block.ignored_param.detach(),m.block.ignored_buf
  if args.version=='after':
   assert torch.equal(p,torch.full((2,),3.)),p;assert torch.equal(b,torch.full((2,),5.)),b
  else:
   assert torch.equal(p,torch.full((2,),71.)),p;assert torch.equal(b,torch.full((2,),71.)),b
 print(loader,case, 'ignored_param=',m.block.ignored_param.detach().tolist(),'ignored_buf=',m.block.ignored_buf.tolist(),
       'siblings=preserved scratch=rebuild17 report_missing=',report.missing_keys,flush=True)
print('LOADER',args.version,'sha256',hash_value,'torch',torch.__version__,'transformers',__import__('transformers').__version__,flush=True)
for loader in ('legacy','dcp'):
 for case in ('ignored','all_present','strict_other'):run(loader,case)
print('PASS expected '+('RED (ignored state uninitialized)' if args.version=='before' else 'GREEN (ignored state initialized)'),flush=True)
