import argparse,copy,hashlib,importlib.util,inspect,sys
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
from hyper_parallel.components.optim.mixed_precision_optimizer import Float16OptimizerWithFloat16Params
from hyper_parallel.components.modules.moe_utils import MoEMonitorCallback

parser=argparse.ArgumentParser(description='Compare the chain post-hook against the existing QKClip step wrapper.')
parser.add_argument('--device',default='cpu',help='cpu or npu device, e.g. npu:0')
parser.add_argument('--base',type=Path,required=True,help='source root containing unmodified Xing4 adapter')
parser.add_argument('--candidate-dir',type=Path,default=Path(__file__).resolve().parent)
args=parser.parse_args()
if args.device.startswith('npu'):
 import torch_npu  # noqa: F401 - registers the NPU backend
DEVICE=torch.device(args.device)
if DEVICE.type=='npu': torch.npu.set_device(DEVICE)
ROOT=args.candidate_dir.resolve()
BASE=args.base.resolve()
print('device:',DEVICE,'torch:',torch.__version__,'base:',BASE,'candidate:',ROOT)
import hyper_parallel.core.optimizer.optimizer as imported_core
expected={
 'base_muon': (BASE/'hyper_parallel/models/xing_4/adapter/optim/muon.py','4a43ee78b91123e820d1fe2cf4192678d5239a20b91f00d844d5df1982f1a2ad'),
 'runtime_core': (Path(imported_core.__file__),'9e24e38e602e929ef8cc3a214710602452aa99abb2f94980a26afa29d7bf63c6'),
 'candidate_core': (ROOT/'candidate_optimizer.py','4b9d5d7ee0a445f65d45b56235ff38d444dbbbe7dedc5331c98c8c6fa0512e2c'),
 'candidate_muon': (ROOT/'candidate_muon.py','26a9713b8f81ad97c5f12c427c80de6b6392d73489fcbcb69e22b7b4ce9cc21d'),
}
for name,(path,want) in expected.items():
 got=hashlib.sha256(path.read_bytes()).hexdigest()
 if got!=want:raise RuntimeError(f'{name} SHA mismatch: {got}; expected {want}')
 print(name,path,got)
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
Candidate=load('hook_core',ROOT/'candidate_optimizer.py').ChainedOptimizer
OrigMuon=load('hook_oldmuon',BASE/'hyper_parallel/models/xing_4/adapter/optim/muon.py')
NewMuon=load('hook_newmuon',ROOT/'candidate_muon.py')
OldChain=__import__('hyper_parallel.core.optimizer.optimizer',fromlist=['ChainedOptimizer']).ChainedOptimizer

# Exact PyTorch hook argument convention and removal semantics, including no closure.
a=nn.Linear(2,1,bias=False).to(DEVICE);chain=Candidate(a,{'a':torch.optim.SGD(a.parameters(),lr=.01)})
assert str(inspect.signature(chain.step)) == '(closure: Any = None) -> None'
events=[]
h=chain.register_step_post_hook(lambda obj,args,kw:events.append((obj,args,kw)))
chain.step(); assert len(events)==1 and events[-1][0] is chain and events[-1][1]==(chain,) and events[-1][2]=={}
calls=[]
def closure():calls.append(1);return torch.tensor(0.,device=DEVICE)
chain.step(closure);assert events[-1][1]==(chain,closure) and events[-1][2]=={} and len(calls)==1
chain.step(closure=closure);assert events[-1][1]==(chain,) and events[-1][2]=={'closure':closure} and len(calls)==2
h.remove();chain.step();assert len(events)==3
# Existing generic consumer expects a removable handle; do not wire it into Trainer as well.
callback=MoEMonitorCallback(a);callbacks=[];callback.on_step_end=lambda:callbacks.append('called');callback.register(chain)
chain.step();assert callbacks==['called'];callback.remove();chain.step();assert callbacks==['called']

# A failed leaf prevents callbacks. A failed callback propagates after leaf updates.
class BrokenSGD(torch.optim.SGD):
 def step(self,closure=None): raise RuntimeError('leaf failed')
f=nn.Linear(2,1,bias=False).to(DEVICE);bad=Candidate(f,{'bad':BrokenSGD(f.parameters(),lr=.01)})
seen=[];bad.register_step_post_hook(lambda *_:seen.append('should not run'))
try:bad.step()
except RuntimeError as e:assert str(e)=='leaf failed'
else:raise AssertionError('leaf failure was swallowed')
assert not seen
b=nn.Linear(2,1,bias=False).to(DEVICE);c=Candidate(b,{'b':torch.optim.SGD(b.parameters(),lr=.1)});b.weight.grad=torch.ones_like(b.weight);before=b.weight.detach().clone()
def fail(*_):raise ValueError('hook failed')
c.register_step_post_hook(fail)
try:c.step()
except ValueError as e:assert str(e)=='hook failed'
else:raise AssertionError('hook failure was swallowed')
assert not torch.equal(before,b.weight)

class Attention(nn.Module):
 def __init__(self):
  super().__init__();self.q_b_proj=nn.Linear(2,6,bias=False,dtype=torch.bfloat16);self.kv_b_proj=nn.Linear(2,6,bias=False,dtype=torch.bfloat16)
  self.qk_nope_head_dim=2;self.qk_rope_head_dim=1;self.v_head_dim=1;self.max_logits=None
class Model(nn.Module):
 def __init__(self):super().__init__();self.attn=Attention()
def run(mod,klass):
 torch.manual_seed(99);model=Model().to(DEVICE)
 class Builder:
  def __init__(self,muon_config,adamw_config,model,no_decay_params):
   del muon_config,adamw_config,no_decay_params
   self.optimizer=klass(model,{'q':torch.optim.SGD([model.attn.q_b_proj.weight],lr=.1),'kv':torch.optim.SGD([model.attn.kv_b_proj.weight],lr=.1)})
  def get_optimizer(self):return self.optimizer
 mod.Muon=Builder
 builder=mod.build_optimizer(model=model,muon_config={},adamw_config={},qk_clip_threshold=100)
 inner=builder.get_optimizer();wrapped=Float16OptimizerWithFloat16Params(inner,model)
 results=[]
 for step in range(3):
  if step==1:
   names=list(inner.optimizers_dict);leaves=[torch.optim.SGD(inner.optimizers_dict[n].param_groups[0]['params'],lr=.1) for n in names]
   inner.chained_optimizers[:]=leaves
   for n,l in zip(names,leaves):inner.optimizers_dict[n]=l
  model.attn.max_logits=torch.tensor([200.,50.],device=DEVICE)
  for p in model.parameters():p.grad=torch.ones_like(p)
  wrapped.step()
  assert all(p.device.type==DEVICE.type and p.main_param.device.type==DEVICE.type for p in model.parameters())
  assert model.attn.max_logits is None
  results.append([(p.detach().clone(),p.main_param.detach().clone()) for p in model.parameters()])
 return results
old=run(OrigMuon,OldChain);new=run(NewMuon,Candidate)
for oldstep,newstep in zip(old,new):
 for oldpair,newpair in zip(oldstep,newstep):
  for a,b in zip(oldpair,newpair):assert torch.equal(a,b)
print('PASS: Torch args/kwargs, removable handle, public MoEMonitor consumer, closure, failures, actual QKClip + FP32 copy-back and leaf replacement; old/new 3 steps bitwise equal')
# Chain hook objects are runtime registrations, absent from DCP optimizer payload.
model=nn.Linear(2,1,bias=False).to(DEVICE); first=Candidate(model,{'sgd':torch.optim.SGD(model.parameters(),lr=.1)})
state_before=first.state_dict();first.register_step_post_hook(lambda *_: None);state_after=first.state_dict()
assert state_before.keys()==state_after.keys() and str(state_before)==str(state_after)
model2=nn.Linear(2,1,bias=False).to(DEVICE);second=Candidate(model2,{'sgd':torch.optim.SGD(model2.parameters(),lr=.1)})
second.load_state_dict(state_after)
assert len(second._optimizer_step_post_hooks)==0
print('PASS: state_dict unchanged by registration; fresh optimizer requires explicit re-registration')
# Compare argument packets against a native torch.optim.SGD hook directly.
t=nn.Linear(2,1,bias=False).to(DEVICE);native=torch.optim.SGD(t.parameters(),lr=.1);native_packets=[];ours=[]
native.register_step_post_hook(lambda opt,args,kwargs:native_packets.append((args[1:],kwargs)))
mirror=Candidate(t,{'native':torch.optim.SGD(t.parameters(),lr=.1)});mirror.register_step_post_hook(lambda opt,args,kwargs:ours.append((args[1:],kwargs)))
for call in (lambda x:x.step(),lambda x:x.step(closure),lambda x:x.step(closure=closure)):
 call(native);call(mirror)
assert native_packets==ours
print('PASS: native torch SGD and candidate emit identical hook argument packets')
# Independent head-row oracle: s=[0.5,1]; Q uses [sqrt(s),sqrt(s),s], K/V [sqrt(s),sqrt(s),1].
torch.manual_seed(99);initial=Model().to(DEVICE);q0=initial.attn.q_b_proj.weight.detach().float();kv0=initial.attn.kv_b_proj.weight.detach().float()
s=torch.tensor([.5,1.],device=DEVICE);qf=torch.stack((s.sqrt(),s.sqrt(),s),dim=1).reshape(-1,1);kvf=torch.stack((s.sqrt(),s.sqrt(),torch.ones_like(s)),dim=1).reshape(-1,1)
for actual,expected in zip(new[0],[(q0-.1)*qf,(kv0-.1)*kvf]):
 torch.testing.assert_close(actual[1],expected,rtol=0,atol=0)
 assert torch.equal(actual[0],expected.to(torch.bfloat16))
print('PASS: independent per-head Q/K/V factor oracle and FP32-before-BF16 copy-back')
# A failing post-hook is raised before mixed-precision copy-back, matching current wrapper timing.
model=nn.Linear(2,1,bias=False,dtype=torch.bfloat16).to(DEVICE);inner=Candidate(model,{'sgd':torch.optim.SGD(model.parameters(),lr=.1)})
wrapped=Float16OptimizerWithFloat16Params(inner,model);model.weight.grad=torch.ones_like(model.weight);before=model.weight.detach().clone();inner.register_step_post_hook(fail)
try:wrapped.step()
except ValueError as e:assert str(e)=='hook failed'
else:raise AssertionError('mixed hook failure swallowed')
assert torch.equal(model.weight,before) and not torch.equal(model.weight.main_param,before.float())
print('PASS: post-hook failure propagates before model-weight copy-back')
