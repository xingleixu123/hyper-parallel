"""Run once per source tree; compare CPU-generated fixtures and NPU packed gradients."""
import json,os,torch,torch_npu
from types import SimpleNamespace
from torch import nn
from hyper_parallel.models.xing_4.transformer.modeling_xing4_0 import Xing4_0MLP
from hyper_parallel.models.xing_4.adapter.moe import adapt_mlp
from hyper_parallel.models.xing_4.adapter.conversion.module_replacement import Xing4GroupedExperts

torch.npu.set_device(0)
E,H,I,T,K=4,128,128,32,2
record={}
for name,limit,dtype in [('default',None,None),('clipped_fp32',2.0,'float32')]:
 torch.manual_seed(122)
 c=SimpleNamespace(hidden_size=H,intermediate_size=I,moe_intermediate_size=I,n_routed_experts=E,
   num_local_experts=E,hidden_act='silu',swiglu_limit=limit,swiglu_compute_dtype=dtype,
   use_fused_swiglu=True,initializer_range=.02)
 experts=nn.ModuleList([adapt_mlp(Xing4_0MLP(c,I)) for _ in range(E)]).to(torch.bfloat16)
 for e in experts:
  for p in e.parameters():
   with torch.no_grad():p.copy_(torch.randn(p.shape).mul(.15).to(p.dtype))
 x0=torch.randn(T,H).mul(2).to(torch.bfloat16)
 inds0=torch.randn(T,E).topk(K,dim=-1).indices
 probs0=torch.rand(T,K).softmax(-1).to(torch.bfloat16)
 upstream0=torch.randn(T,H).to(torch.bfloat16)
 for j,e in enumerate(experts):
  for n in ('gate_proj','up_proj','down_proj'):
   record[f'{name}.fixture.expert{j}.{n}']=getattr(e,n).weight.detach().clone()
 for k,v in [('x',x0),('indices',inds0),('probs',probs0),('upstream',upstream0)]:record[f'{name}.fixture.{k}']=v
 experts=experts.npu();packed=Xing4GroupedExperts(module=experts)
 x=x0.npu().detach().requires_grad_();probs=probs0.npu().detach().requires_grad_()
 out=packed(x,inds0.npu(),probs)
 grads=torch.autograd.grad((out.float()*upstream0.npu().float()).sum(),(x,probs,packed.gate_up_proj,packed.down_proj))
 for k,v in zip(('output','input_grad','routing_grad','gate_up_grad','down_grad'),(out,*grads)):
  record[f'{name}.result.{k}']=v.detach().cpu()
 assert all(torch.isfinite(v).all() for k,v in record.items() if k.startswith(name+'.result.'))
torch.save(record,os.environ['OUTPUT_FILE'])
print(json.dumps({'event':'passed','snapshot':os.environ['SNAPSHOT'],'keys':len(record)}),flush=True)
