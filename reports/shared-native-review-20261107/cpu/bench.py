import ast, copy, hashlib, json, math, statistics, sys, time, platform
from pathlib import Path
import torch
from hyper_parallel.core.optimizer import Muon, SwapOptimizerConfig
import hyper_parallel.core.optimizer.swap_optimizer_base as mod
from hyper_parallel.core.optimizer.swap_adam import build_adam_swap_adapter
from hyper_parallel.core.optimizer.swap_muon import build_muon_swap_adapter

root=Path(__file__).resolve().parent
sources={k:(root/(k+'.py.txt')).read_text(encoding='utf-8-sig') for k in ('base','head')}
def extract(text):
    tree=ast.parse(text)
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='SwapOptimizer')
    f=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
    return f
nodes={k:extract(v) for k,v in sources.items()}
a=copy.deepcopy(nodes['head'])
assert isinstance(a.body[1],ast.If) # docstring then guard
a.body.pop(1)
assert ast.dump(nodes['base'],include_attributes=False)==ast.dump(a,include_attributes=False)
methods={}
for k,f in nodes.items():
    assert not any(isinstance(n,ast.Name) and n.id in ('super','__class__') for n in ast.walk(f))
    module=ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),f],type_ignores=[]))
    scope=dict(vars(mod)); exec(compile(module,k+'-initializer','exec'),scope); methods[k]=scope['__init__']
config=SwapOptimizerConfig(swap_times=1,min_numel=1024,packed_swap=False)
original=mod.SwapOptimizer.__init__
res={'python':sys.version,'torch':torch.__version__,'platform':platform.platform(),'source_sha256':{k:hashlib.sha256(v.encode()).hexdigest() for k,v in sources.items()},'initializer_ast_sha256':{k:hashlib.sha256(ast.dump(f,include_attributes=False).encode()).hexdigest() for k,f in nodes.items()},'method_only_delta':'head guard, checked via AST','batch_iterations':200,'batches_per_variant':12,'units':'nanoseconds per construction','cases':{}}
def pct(x,p):
    a=sorted(x); return a[math.ceil(len(a)*p)-1]
try:
 for name,opt_type,factory in [('AdamW',torch.optim.AdamW,build_adam_swap_adapter),('Muon',Muon,build_muon_swap_adapter)]:
    p=torch.nn.Parameter(torch.ones(2,2,device='cpu')); before=p.detach().clone(); opt=opt_type([p],lr=.01)
    def construct(which):
        mod.SwapOptimizer.__init__=methods[which]
        return mod.SwapOptimizer(opt,config,factory)
    first={}
    for k in ('base','head'):
        t=time.perf_counter_ns(); w=construct(k); first[k]=time.perf_counter_ns()-t
        assert w.optimizer is opt
    h=opt.register_step_pre_hook(lambda *args:None)
    base_wrapped=construct('base'); assert base_wrapped.optimizer is opt
    try: construct('head'); raise AssertionError('head accepted hook')
    except ValueError as e: assert 'registered step hooks' in str(e)
    h.remove(); assert not opt.state and torch.equal(p,before)
    cold={}
    for k in ('base','head'):
        t=time.perf_counter_ns(); w=construct(k); cold[k]=time.perf_counter_ns()-t
        assert w.optimizer is opt
    for k in ('base','head'):
        for _ in range(100): construct(k)
    samples={'base':[],'head':[]}
    for cycle in range(6):
        order=('base','head','head','base') if cycle%2==0 else ('head','base','base','head')
        for k in order:
            mod.SwapOptimizer.__init__=methods[k]
            t=time.perf_counter_ns()
            for _ in range(200): mod.SwapOptimizer(opt,config,factory)
            samples[k].append((time.perf_counter_ns()-t)/200)
    assert not opt.state and torch.equal(p,before)
    res['cases'][name]={'first_construction_ns_ordered_base_then_head':first,'after_semantics_check_ns':cold,'samples_ns_per_construction':samples,'summary':{k:{'median':statistics.median(v),'p95_nearest_rank':pct(v,.95),'min':min(v),'max':max(v)} for k,v in samples.items()},'semantic_checks':'base accepts pre-hook, head rejects pre-hook, no-hook both succeed; CPU parameter unchanged and state empty'}
finally: mod.SwapOptimizer.__init__=original
(root/'results.json').write_text(json.dumps(res,indent=2)+'\n',encoding='utf-8')
print(json.dumps(res,indent=2))



