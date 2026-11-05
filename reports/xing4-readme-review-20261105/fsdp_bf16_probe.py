import os,json,torch,torch_npu
import torch.distributed as dist
from hyper_parallel import fully_shard,init_device_mesh,SkipDTensorDispatch
from hyper_parallel.core.distributed_checkpoint.api import save,load
from hyper_parallel.models._transformers.model_builder import apply_model_init_dtype,validate_model_init_dtype
from hyper_parallel.models.xing_4.adapter.moe import Xing4Router
from hyper_parallel.models.xing_4.transformer.configuration_xing4_0 import Xing4_0Config
from hyper_parallel.models.xing_4.transformer.modeling_xing4_0 import Xing4_0TopkRouter
rank=int(os.environ['LOCAL_RANK']);torch.npu.set_device(rank);dist.init_process_group('hccl');device=torch.device('npu',rank)
c=Xing4_0Config(hidden_size=8,intermediate_size=16,moe_intermediate_size=16,n_routed_experts=4,num_experts_per_tok=1,n_group=1,topk_group=1,moe_aux_loss_coeff=.001,moe_router_enable_expert_bias=True,moe_router_bias_update_rate=1e-4,use_pad_tokens=False,moe_input_jitter_eps=0)
m=Xing4Router(c,Xing4_0TopkRouter(c))
with torch.no_grad():
 m.weight.copy_(torch.arange(32,dtype=torch.float32).reshape(4,8)*0.001)
apply_model_init_dtype(m,'bfloat16');validate_model_init_dtype(m,'bfloat16');m.to(device);assert m.weight.dtype==torch.bfloat16 and m.e_score_correction_bias.dtype==torch.float32 and m._token_counts.dtype==torch.int64
mesh=init_device_mesh('npu',(4,),mesh_dim_names=('dp',));m=fully_shard(m,mesh=mesh)
assert m.e_score_correction_bias.dtype==torch.float32 and m._token_counts.dtype==torch.int64
with torch.no_grad():m.e_score_correction_bias.copy_(torch.tensor([.50015,-.50015,0,0],device=device));m._token_counts.copy_(torch.tensor([256,257,258,259],device=device))
state={'model':m.state_dict()};save(state,checkpoint_id='/work/output/bf16-fsdp-dcp2');dist.barrier()
with torch.no_grad():m.e_score_correction_bias.zero_();m._token_counts.zero_()
state={'model':m.state_dict()};load(state,checkpoint_id='/work/output/bf16-fsdp-dcp2');m.load_state_dict(state['model']);torch.npu.synchronize()
assert m.e_score_correction_bias.dtype==torch.float32 and m._token_counts.dtype==torch.int64
assert torch.equal(m._token_counts.cpu(),torch.tensor([256,257,258,259]));assert torch.equal(m.e_score_correction_bias.cpu(),torch.tensor([.50015,-.50015,0,0]))
x=torch.randn(1,32,8,device=device,dtype=torch.bfloat16,requires_grad=True)
with SkipDTensorDispatch():
 _,weights,_=m(x);weights.sum().backward()
torch.npu.synchronize();assert x.grad is not None and torch.isfinite(x.grad).all();assert m.e_score_correction_bias.dtype==torch.float32 and m._token_counts.dtype==torch.int64
counts=m._token_counts.clone();dist.all_reduce(counts);before=m.e_score_correction_bias.clone();expected=before+(counts.sum()-counts*4).sign().to(torch.float32)*1e-4
optimizer=torch.optim.SGD(m.parameters(),lr=0.01)
with SkipDTensorDispatch(): optimizer.step();optimizer.zero_grad()
from hyper_parallel.components.modules.moe_utils import MoEMonitorCallback
MoEMonitorCallback(m,dp_group=dist.group.WORLD).on_step_end();torch.npu.synchronize()
assert torch.equal(m.e_score_correction_bias,expected) and not m._token_counts.any()
print(json.dumps({'rank':rank,'status':'pass','bias_dtype':str(m.e_score_correction_bias.dtype),'count_dtype':str(m._token_counts.dtype),'checkpoint':'bf16-fsdp-dcp2','global_counts_before_update':counts.cpu().tolist(),'counts_after_monitor':m._token_counts.cpu().tolist(),'bias_after_monitor':m.e_score_correction_bias.cpu().tolist()}),flush=True);dist.barrier();dist.destroy_process_group()
