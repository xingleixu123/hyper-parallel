import sys,types,os,hashlib,json
from types import SimpleNamespace
# CPU-only import shim: attention replacement is never called; CP collectives are real.
m=types.ModuleType('hyper_parallel.models.xing_4.adapter.conversion.module_replacement')
m.Xing4MLAAttention=type('Xing4MLAAttention',(),{})
m.fusion_attention_with_logit_maxima=lambda *a,**k:None
sys.modules[m.__name__]=m
import torch,torch.distributed as dist
from hyper_parallel.models.xing_4.adapter.distributed.context_parallel import xing4_cp_mean_loss,xing4_cp_shift_tokens
from hyper_parallel.trainer.runtime.loss_normalization import StepLossNormalization
from hyper_parallel.components.functional.aux_loss import aux_loss_auto_scale,aux_loss_scale_context,bind_aux_loss_scale
class Mesh:
 def __init__(self,group,ranks): self.group=group;self.ranks=ranks
 def size(self): return len(self.ranks)
 def get_group(self): return self.group
 def get_local_rank(self): return self.ranks.index(dist.get_rank())
def go():
 dist.init_process_group('gloo'); rank=dist.get_rank(); dp,cp=divmod(rank,2)
 groups=[dist.new_group(ranks=[0,1]),dist.new_group(ranks=[2,3])]
 cm=Mesh(groups[dp],[2*dp,2*dp+1]); world=Mesh(dist.group.WORLD,[0,1,2,3]); mesh=SimpleNamespace(sequence_parallel=False,dp_cp_mesh=world,cp_mesh=cm,dp_size=2,cp_size=2)
 mean=xing4_cp_mean_loss(module=None,method_fqn='',original_method=lambda *a:None,context={'mesh_context':mesh})
 shift=xing4_cp_shift_tokens(module=None,method_fqn='',original_method=lambda *a:None,context={'mesh_context':mesh})
 seqs=[[[1,2,3,-1,-1,-1],[1,2,3,4,5,-1]],[[1,2,3,4,-1,-1],[1,2,3,4,5,6]]]
 w=torch.nn.Parameter(torch.tensor(0.7,dtype=torch.float64)); v=torch.nn.Parameter(torch.tensor(0.4,dtype=torch.float64)); u=torch.nn.Parameter(torch.tensor(0.3,dtype=torch.float64)); norm=StepLossNormalization('token',2,mesh); total=0.; metrics={}
 for step in range(2):
  labels=torch.tensor([seqs[step][dp][cp*3:(cp+1)*3]]); shifted=shift(labels,-1); valid=labels.ne(-1); mv=shifted.ne(-1); x=labels.clamp_min(0).double(); y=shifted.clamp_min(0).double()
  lm=mean((((w*x-0.2)**2)*valid).sum(),valid.sum()); mtp=mean((((v*y-0.1)**2)*mv).sum(),mv.sum())*.1
  aux=u.square()*(step+dp+1)*.2
  with aux_loss_scale_context():
   lm=aux_loss_auto_scale(lm,aux);bind_aux_loss_scale(lm);bind_aux_loss_scale(mtp)
   out=norm.reduce({'foundation_loss':lm,'foundation_loss/mtp':mtp},{'foundation_tokens':valid.sum()}); loss=sum(out.values());loss.backward();total+=loss.item()
 for p in (w,v,u):dist.all_reduce(p.grad);p.grad.div_(4)
 got,met=norm.finish([w,v,u],total,metrics)
 # independent full-sequence reference; aux only affects gradients, never reported forward loss.
 wr=torch.tensor(.7,dtype=torch.float64,requires_grad=True);vr=torch.tensor(.4,dtype=torch.float64,requires_grad=True);ur=torch.tensor(.3,dtype=torch.float64,requires_grad=True);obj=0; vis=0;den=0
 for step in range(2):
  for d in range(2):
   a=torch.tensor(seqs[step][d]); mask=a.ne(-1); shifted=torch.cat((a[1:],torch.tensor([-1])));sm=shifted.ne(-1);n=mask.sum();x=a.clamp_min(0).double(); y=shifted.clamp_min(0).double(); lm=(((wr*x-.2)**2)*mask).sum()/n; mtp=(((vr*y-.1)**2)*sm).sum()/sm.sum(); aux=ur.square()*(step+d+1)*.2; vis=vis+n*(lm+.1*mtp);obj=obj+n*(lm+.1*mtp+aux);den+=n
 (obj/den).backward();torch.testing.assert_close(torch.tensor(got,dtype=torch.float64),vis.detach()/den,rtol=2e-6,atol=2e-6)
 for a,b in zip((w,v,u),(wr,vr,ur)):torch.testing.assert_close(a.grad,b.grad,rtol=2e-6,atol=2e-6)
 print(json.dumps({'rank':rank,'visible_loss':got,'expected_visible':(vis/den).item(),'grad':[p.grad.item() for p in (w,v,u)],'expected_grad':[p.grad.item() for p in (wr,vr,ur)],'total_tokens':int(den)}),flush=True);dist.destroy_process_group()
if __name__=='__main__':go()
