import json,torch,torch_npu
from torch import nn
from hyper_parallel.components.modules.mhc import MhcPreModule,MhcPostModule
from hyper_parallel.models.xing_4.adapter.training import Xing4DecoderLayer
from hyper_parallel.models.xing_4.transformer.modeling_xing4_0 import Xing4_0HyperConnection
import omni_training_custom_ops
torch.manual_seed(107);N,H,B,S=4,512,1,16;dev='npu:0'
base=[(torch.randn(B,S,N,H)*0.5).to(torch.bfloat16),torch.randn((N+2)*N,N*H)*0.03,torch.tensor([0.7,0.9,1.2]),torch.randn((N+2)*N)*0.6]
u=torch.randn(B,S,N,H,device=dev)
class Src(nn.Module):
 def __init__(self):
  super().__init__();self.num_stream=N;self.mhc_use_gamma=False;self.mhc_recur_norm=20;self.hc_eps=self.norm_eps=1e-6
  self.phi=nn.Linear(N*H,(N+2)*N,bias=False,device=dev,dtype=torch.float32)
  self.branch_alpha=nn.Parameter(base[2].to(dev).clone());self.branch_beta=nn.Parameter(base[3].to(dev).clone())
  with torch.no_grad():self.phi.weight.copy_(base[1].to(dev))
def run(mode):
 x=base[0].to(dev).detach().requires_grad_(True);src=Src();pre=MhcPreModule(module=src);postmod=MhcPostModule(module=src)
 y,post,mix,res=pre(x.flatten(2));branch=torch.tanh(y.float()).to(torch.bfloat16)
 if mode=='public':out=postmod(branch,res,post,mix).unflatten(-1,(N,H))
 elif mode=='public_transposed':out=postmod(branch,res,post,mix.transpose(-1,-2).contiguous()).unflatten(-1,(N,H))
 elif mode=='xing_post':out=Xing4DecoderLayer._post(branch,res.unflatten(-1,(N,H)),post,mix)
 elif mode in ('oracle_srcdest','oracle_destsrc'):
  mat=mix if mode=='oracle_srcdest' else mix.transpose(-1,-2)
  out=(post.unsqueeze(-1)*branch.unsqueeze(-2)+torch.sum(mat.unsqueeze(-1)*res.unflatten(-1,(N,H)).unsqueeze(-2),dim=2)).to(branch.dtype)
 else:raise ValueError(mode)
 loss=(out.float()*u).sum();grads=torch.autograd.grad(loss,(x,src.phi.weight,src.branch_alpha,src.branch_beta))
 return {'out':out.detach().float().cpu(),'mix':mix.detach().float().cpu(),'collapsed':y.detach().float().cpu(),'post':post.detach().float().cpu(),**{k:v.detach().float().cpu() for k,v in zip(('grad_x','grad_phi','grad_alpha','grad_beta'),grads)}}
def metric(a,b):
 d=(a-b).abs();return {'max_abs':d.max().item(),'rmse':d.square().mean().sqrt().item(),'reference_rms':b.square().mean().sqrt().item()}
def official():
 x,p,a,b=[v.to(dev).detach().requires_grad_(True) for v in base]
 class C: pass
 q=C();q.hc_mult=N;q.hc_sinkhorn_iters=20;q.hc_eps=1e-6;q.input_norm=lambda z:z*torch.rsqrt(z.square().mean(-1,keepdim=True)+1e-6)
 q.hc_fn=p;q.hc_base=b;q.hc_scale=a;q.mhc_h_res_clamp_min=-30.;q.mhc_h_res_clamp_max=30.
 post,mix,y=Xing4_0HyperConnection.forward(q,x);branch=torch.tanh(y.float()).to(torch.bfloat16)
 out=post.unsqueeze(-1)*branch.unsqueeze(-2)+torch.matmul(mix,x)
 gs=torch.autograd.grad((out.float()*u).sum(),(x,p,a,b))
 return {'out':out.detach().float().cpu(),'mix':mix.detach().float().cpu(),'collapsed':y.detach().float().cpu(),'post':post.detach().float().cpu(),**{k:v.detach().float().cpu() for k,v in zip(('grad_x','grad_phi','grad_alpha','grad_beta'),gs)}}
r={m:run(m) for m in ('public','public_transposed','xing_post','oracle_srcdest','oracle_destsrc')}
r['official']=official()
keys=('out','grad_x','grad_phi','grad_alpha','grad_beta')
comparisons={}
for a,b in [('public','oracle_srcdest'),('public_transposed','oracle_destsrc'),('xing_post','public_transposed'),('public','public_transposed'),('oracle_srcdest','oracle_destsrc'),('public','official'),('public_transposed','official')]:comparisons[a+'__vs__'+b]={k:metric(r[a][k],r[b][k]) for k in keys}
out={'environment':{'torch':torch.__version__,'torch_npu':torch_npu.__version__,'source':'571d9cb9bfb6c878c59e6d40f3ccc75c1cc359c5','matching_pr_head':'0854496de0004a01c04678d3f49983d2bf14ea27','device':torch.npu.get_device_name(0),'dtype':'bf16 inputs, fp32 phi/alpha/beta','shape':[B,S,N,H],'seed':107,'sinkhorn_iterations':20,'eps':1e-6,'upstream':'seeded nonuniform fp32'},'mix_asymmetry':metric(r['public']['mix'],r['public']['mix'].transpose(-1,-2)),'comparisons':comparisons,'contract':'oracle_srcdest sums axis source of h_res[source,dest]; oracle_destsrc implements h_res[dest,source] @ residual; both use identical public pre output','note':'This tests operator dataflow and both interpretations; public consumer orientation cannot be declared wrong without its source model contract.'}
print(json.dumps(out,indent=2));open('/work/output/result-official.json','w').write(json.dumps(out,indent=2)+'\n');torch.save(r,'/work/output/raw-official.pt')
