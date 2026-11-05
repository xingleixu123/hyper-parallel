import importlib.util,sys,hashlib,json
from pathlib import Path
import torch,torch_npu
from torch import nn
from transformers import PreTrainedModel,PretrainedConfig

def test_loader_before_after_actual_npu():
 torch.npu.set_device(0);device=torch.device('npu:0');base=Path('/work/scripts/hp-inventory');binding=json.loads((base/'loader-candidate-binding.json').read_text())
 class Block(nn.Module):
  def __init__(self):
   super().__init__();self.weight=nn.Parameter(torch.empty(2,2,device=device));self.register_buffer('bias',torch.empty(2,device=device));self.register_buffer('count',torch.empty(2,device=device,dtype=torch.int64))
 class Model(PreTrainedModel):
  config_class=PretrainedConfig
  _keys_to_ignore_on_load_missing=[r'^block\.count$']
  def __init__(self):super().__init__(PretrainedConfig());self.block=Block()
  def _init_weights(self,module):
   if isinstance(module,Block):nn.init.constant_(module.weight,101);nn.init.constant_(module.bias,102);nn.init.zeros_(module.count)
 for version in ('before','after'):
  path=base/('checkpoint_loader-'+version+'.py');digest=hashlib.sha256(path.read_bytes()).hexdigest();assert digest==binding[version]
  spec=importlib.util.spec_from_file_location('probe_loader_'+version,path);mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
  model=Model();model.block.weight.data.fill_(7);model.block.bias.fill_(0.50015);model.block.count.fill_(71)
  saved_weight=model.block.weight.detach().clone();saved_bias=model.block.bias.clone();report=mod.LoadReport(loaded_keys=('block.weight','block.bias'),missing_keys=('block.count',),unexpected_keys=())
  final=mod._finalize_model_loading(model,report,strict=True);torch.npu.synchronize()
  assert final.missing_keys==() and torch.equal(model.block.weight,saved_weight) and torch.equal(model.block.bias,saved_bias)
  expected=71 if version=='before' else 0;assert model.block.count.dtype==torch.int64 and torch.equal(model.block.count,torch.full((2,),expected,device=device,dtype=torch.int64))
  print(json.dumps({'version':version,'sha256':digest,'device':str(device),'count':model.block.count.cpu().tolist(),'loaded_weight_bias':'unchanged','report_missing':list(final.missing_keys)}),flush=True)
