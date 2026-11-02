"""Four-rank DP2 x TP2 SwiGLU micro-test. Independent test, not training baseline."""
import argparse,copy,hashlib,json,traceback
from pathlib import Path
import os
import torch_npu
import torch
import torch.distributed as dist
from torch import nn

class _Bootstrap:
    RANK = int(os.environ['RANK'])
    LOCAL = int(os.environ['LOCAL_RANK'])
    OUT = None
    @staticmethod
    def require(ok, message):
        if not ok: raise RuntimeError(message)
    @staticmethod
    def sha(path):
        h = hashlib.sha256()
        with Path(path).open('rb') as f:
            for block in iter(lambda: f.read(4 * 1024 * 1024), b''): h.update(block)
        return h.hexdigest()
    @staticmethod
    def write(path, value):
        with path.open('x', encoding='utf8') as f:
            json.dump(value, f, indent=2, allow_nan=False)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())

b = _Bootstrap()

class Source(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj=nn.Linear(16,32,bias=False)
        self.up_proj=nn.Linear(16,32,bias=False)
        self.down_proj=nn.Linear(32,16,bias=False)

class MoE(nn.Module):
    def __init__(self,source):
        super().__init__()
        from hyper_parallel.components.modules.swiglu_mlp import SwiGLUMLP
        self.shared_experts=SwiGLUMLP(module=source)
    def forward(self,x): return self.shared_experts(x)
class Layer(nn.Module):
    def __init__(self,source): super().__init__(); self.mlp=MoE(source)
    def forward(self,x): return self.mlp(x)
class Core(nn.Module):
    def __init__(self,source): super().__init__(); self.layers=nn.ModuleList([Layer(source)])
    def forward(self,x): return self.layers[0](x)
class Model(nn.Module):
    def __init__(self,source): super().__init__(); self.model=Core(source)
    def forward(self,x): return self.model(x)

def digest(t):
    v=t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(v).hexdigest()
def diff(actual,reference):
    a=actual.detach().float(); r=reference.detach().float()
    return {'max_abs':float((a-r).abs().max().item()),'rtol':0.008,'atol':0.0,
            'close':bool(torch.allclose(a,r,rtol=0.008,atol=0.0))}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--case',required=True,choices=['default','replicate'])
    parser.add_argument('--output',type=Path,default=Path('swiglu-tp-results'))
    args=parser.parse_args()
    b.require(int(os.environ['WORLD_SIZE'])==4 and 0<=b.LOCAL<4,'requires four torchrun ranks')
    b.require(torch.npu.device_count()==4,'requires exactly four visible NPU devices')
    torch.npu.set_device(b.LOCAL)
    b.OUT=args.output / args.case / ('rank-'+str(b.RANK))
    b.OUT.mkdir(parents=True,exist_ok=False)
    case=args.case
    from hyper_parallel.core.dtensor.device_mesh import init_device_mesh
    from hyper_parallel.distributed import ShardingPlanner
    from hyper_parallel.distributed.apply import apply_sharding_plan
    from hyper_parallel.distributed.recipe_spec import ModuleShardingSpec,TP,CP,EP
    from hyper_parallel.core.dtensor.placement_types import Replicate,Shard
    import hyper_parallel.components.modules.swiglu_mlp as swiglu_mod
    report={'kind':'isolated_micro_module_not_training','rank':b.RANK,'case':case,'phase':'initializing',
            'shape':{'H':16,'I':32,'B':2,'S':8},'dtype':'bfloat16','tol':{'rtol':.008,'atol':0.0}}
    profiler=None; original_swiglu=swiglu_mod.functional.swiglu; calls={'count':0}
    def observed_swiglu(*a,**kw):
        calls['count']+=1
        return original_swiglu(*a,**kw)
    try:
        dist.init_process_group(backend='hccl')
        full=init_device_mesh('npu',(2,2),mesh_dim_names=('dp','tp'),rank_list=(0,1,2,3),init_backend=True)
        tp=full['tp'];coord=full.get_coordinate(); group=tp.get_group()
        group_ranks=dist.get_process_group_ranks(group)
        b.require(len(group_ranks)==2 and b.RANK in group_ranks and list(coord)==[b.RANK//2,b.RANK%2], 'unexpected actual mesh')
        report['mesh']={'shape':[2,2],'names':['dp','tp'],'coordinate':list(coord),'tp_group_ranks':group_ranks}
        torch.manual_seed(123)
        device='npu:'+str(b.LOCAL); source=Source().to(device=device,dtype=torch.bfloat16)
        for p in source.parameters(): dist.broadcast(p.data,src=0)
        x=torch.randn(2,8,16,device=device,dtype=torch.bfloat16);dist.broadcast(x,src=0)
        model=Model(source);ref=copy.deepcopy(model)
        report['actual_shared_weight_sha256']={n:digest(p) for n,p in model.named_parameters()}
        report['reference_weight_sha256']={n:digest(p) for n,p in ref.named_parameters()}
        b.require(report['actual_shared_weight_sha256']==report['reference_weight_sha256'],'clone weight mismatch')
        report['input_sha256']=digest(x); xr=x.clone().detach().requires_grad_()
        refy=ref(xr);refy.float().sum().backward()
        swiglu_mod.functional.swiglu=observed_swiglu
        overrides={}
        if case=='replicate':
            act={TP:Replicate(),CP:Shard(1),EP:Replicate()};par={TP:Replicate(),CP:Replicate(),EP:Replicate()}
            for fc in ('linear_fc1','linear_fc2'):
                overrides['model.layers.0.mlp.shared_experts.'+fc]=ModuleShardingSpec(
                    params={'weight':par},in_src={'hidden_states':act},in_dst={'hidden_states':act},
                    out_src={'output':act},out_dst={'output':act})
        plan=ShardingPlanner(plan_overrides=overrides).plan(model,tp,tp_size=2,sequence_parallel=False,loss_parallel=True)
        report['phase']='planned'
        report['placements']={f'{n}.{p}':str(v) for n,s in plan.modules.items() for p,v in (s.params or {}).items()}
        report['contracts']={n:{'in_src':str(s.in_src),'in_dst':str(s.in_dst),'out_src':str(s.out_src),'out_dst':str(s.out_dst)} for n,s in plan.modules.items()}
        model,source_info=apply_sharding_plan(model,plan,tp,validate_mode=False)
        report['phase']='applied';report['source_shards']={k:str(v) for k,v in (source_info or {}).items()}
        report['local_parameters']={k:{'shape':list(v.shape),'dtype':str(v.dtype),'sha256':digest(v)} for k,v in model.named_parameters()}
        xin=x.clone().detach().requires_grad_(); report['actual_input_sha256']=digest(xin)
        profiler=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU])
        with profiler:
            y=model(xin);report['phase']='forward_complete'
            y.float().sum().backward();torch.npu.synchronize();report['phase']='backward_complete'
        report['forward']=diff(y,refy);report['input_gradient']=diff(xin.grad,xr.grad)
        rp=dict(ref.named_parameters())
        report['weight_gradients']={k:diff(v.grad,rp[k].grad) for k,v in model.named_parameters()}
        report['classification']='compared'
    except Exception as e:
        report['classification']='exception';report['exception']=type(e).__name__;report['message']=str(e);report['traceback']=traceback.format_exc()
    finally:
        swiglu_mod.functional.swiglu=original_swiglu;report['fused_swiglu_calls']=calls['count']
        if profiler is not None:
            try:report['cpu_profiler_collectives']={z.key:z.count for z in profiler.key_averages() if any(s in z.key.lower() for s in ('gather','reduce','broadcast','redistribute','c10d','hccl'))}
            except Exception as e:report['profiler_error']=str(e)
        b.write(b.OUT/'micro-result.json',report)
        b.write(b.OUT/'complete.json',{'status':'micro_test_finished','rank':b.RANK,'case':case,'classification':report.get('classification'),'report_sha256':b.sha(b.OUT/'micro-result.json')})
        if dist.is_initialized():dist.destroy_process_group()

if __name__=='__main__':main()
