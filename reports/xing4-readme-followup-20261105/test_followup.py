"""External regression probes; no product-test files are added by this task."""
import os
from types import SimpleNamespace
from functools import partial
from unittest.mock import patch
import pytest
import torch
from torch import nn
import torch.distributed.checkpoint as dcp

from hyper_parallel.core.activation_memory.recompute_state import create_recompute_contexts
from hyper_parallel.core.multicore.modules._precision import preserve_fp32_parameters
from hyper_parallel.models._transformers.model_builder import apply_model_init_dtype, validate_model_init_dtype
from hyper_parallel.models.xing_4.adapter.configuration import create_config
from hyper_parallel.models.xing_4.adapter.moe import Xing4Router, Xing4MLP
from hyper_parallel.models.xing_4.adapter.training import Xing4ForCausalLM
from hyper_parallel.models.xing_4.transformer.configuration_xing4_0 import Xing4_0Config
from hyper_parallel.models.xing_4.transformer.modeling_xing4_0 import Xing4_0TopkRouter, Xing4_0MLP
from hyper_parallel.components.functional.activation_swiglu import swiglu

DEVICE=os.environ.get("TEST_DEVICE","cpu")
if DEVICE.startswith("npu"):
    import torch_npu
    torch.npu.set_device(DEVICE)

def config(**extra):
    values=dict(hidden_size=8,intermediate_size=16,moe_intermediate_size=16,
        n_routed_experts=2,num_experts_per_tok=1,n_group=1,topk_group=1,
        moe_aux_loss_coeff=0.001,moe_router_enable_expert_bias=True,
        moe_router_bias_update_rate=1e-4,use_pad_tokens=False,moe_input_jitter_eps=0,
        mhc_init_gating_factor=0.01,mhc_layernorm_epsilon=1e-6,mtp_loss_factor=.1,
        use_cache=False,num_hidden_layers=2,first_k_dense_replace=1,chunk_loss_size=4)
    values.update(extra)
    return values

def router(**extra):
    c=Xing4_0Config(**config(**extra)); m=Xing4Router(c,Xing4_0TopkRouter(c)).to(DEVICE)
    with torch.no_grad():
        m.weight.zero_();m.weight[0,0]=1;m.weight[1,0]=-1;m.e_score_correction_bias.zero_()
    return m

def inputs(padding=0):
    x=torch.zeros(1,513,8,device=DEVICE,dtype=torch.bfloat16)
    x[:,:256,0]=1;x[:,256:,0]=-1
    if padding:x=torch.cat([x.new_zeros(1,padding,8),x],dim=1)
    return x.requires_grad_()

def test_router_dtype_exact_counts_and_bias():
    m=router();bias=m.e_score_correction_bias
    bias.copy_(torch.tensor([0.50015,-0.50015],device=DEVICE))
    saved=bias.clone()
    apply_model_init_dtype(m,"bfloat16")
    validate_model_init_dtype(m,"bfloat16")
    assert torch.equal(m.e_score_correction_bias,saved)
    assert m.weight.dtype==torch.bfloat16
    assert m._token_counts.dtype==torch.int64
    assert m.e_score_correction_bias.dtype==torch.float32
    assert "e_score_correction_bias" not in dict(m.named_parameters())
    m.e_score_correction_bias.zero_()
    x=inputs()
    _,weights,indices=m(x)
    weights.sum().backward()
    assert torch.equal(m._token_counts.cpu(),torch.tensor([256,257]))
    assert torch.isfinite(m.weight.grad).all()
    before=m.e_score_correction_bias.clone()
    m.update_expert_bias(1e-4)
    assert torch.equal(m.e_score_correction_bias,before+before.new_tensor([1e-4,-1e-4]))
    assert not m._token_counts.any()
    # Integer comparison must also retain a one-token imbalance above FP32's exact range.
    m._token_counts.copy_(torch.tensor([2**24,2**24+1],device=DEVICE))
    before=m.e_score_correction_bias.clone();m.update_expert_bias(1e-4)
    assert torch.equal(m.e_score_correction_bias,before+before.new_tensor([1e-4,-1e-4]))

def test_router_padding_accumulation_recompute_and_disabled():
    m=router(use_pad_tokens=True);apply_model_init_dtype(m,"bfloat16")
    x=inputs(2)
    for _ in range(2):m(x)
    assert torch.equal(m._token_counts.cpu(),torch.tensor([512,514]))
    _,recompute=create_recompute_contexts()
    with recompute:m(x)
    assert torch.equal(m._token_counts.cpu(),torch.tensor([512,514]))
    m.enable_expert_bias=False;m._token_counts.zero_();m(x)
    assert not m._token_counts.any()
    assert m.last_aux_loss is not None
    m.eval();m.enable_expert_bias=True;m(x)
    assert not m._token_counts.any()

def test_router_meta_materialization_and_legacy_dcp(tmp_path):
    c=Xing4_0Config(**config())
    with torch.device("meta"):m=Xing4Router(c,Xing4_0TopkRouter(c))
    apply_model_init_dtype(m,"bfloat16")
    m.to_empty(device=DEVICE)
    assert m.e_score_correction_bias.dtype==torch.float32 and m._token_counts.dtype==torch.int64
    old=router().state_dict()
    old["_token_counts"]=torch.tensor([256,257],dtype=torch.float32,device=DEVICE)
    old["e_score_correction_bias"]=torch.tensor([.50015,-.50015],device=DEVICE)
    checkpoint=tmp_path/"legacy"
    dcp.save({"model":old},checkpoint_id=str(checkpoint))
    target=router();apply_model_init_dtype(target,"bfloat16")
    state={"model":target.state_dict()}
    dcp.load(state,checkpoint_id=str(checkpoint));target.load_state_dict(state["model"],strict=True)
    assert torch.equal(target._token_counts.cpu(),torch.tensor([256,257]))
    assert torch.equal(target.e_score_correction_bias,old["e_score_correction_bias"])
    validate_model_init_dtype(target,"bfloat16")
    assert set(target.state_dict())==set(old)

def test_shared_precision_helper_and_validation():
    class Protected(nn.Module):
        fp32_buffer_names=("bias",)
        def __init__(self):
            super().__init__();self.p=nn.Parameter(torch.ones(2))
            self.register_buffer("bias",torch.tensor([.50015,-.50015]))
            self.register_buffer("normal",torch.ones(2))
        def _apply(self,fn,recurse=True):
            return super()._apply(preserve_fp32_parameters(self,fn,self.fp32_buffer_names),recurse=recurse)
    m=Protected().to(DEVICE);saved=m.bias.clone();apply_model_init_dtype(m,"bfloat16")
    assert torch.equal(m.bias,saved) and m.normal.dtype==torch.bfloat16
    m.bias=m.bias.to(torch.bfloat16)
    with pytest.raises(RuntimeError,match="bias"):validate_model_init_dtype(m,"bfloat16")
    m.bias=m.bias.float();m.normal=m.normal.float()
    with pytest.raises(RuntimeError,match="normal"):validate_model_init_dtype(m,"bfloat16")

@pytest.mark.parametrize("name",["mhc_init_gating_factor","mhc_layernorm_epsilon","mtp_loss_factor",
    "moe_aux_loss_coeff","moe_router_bias_update_rate","moe_router_enable_expert_bias","use_pad_tokens"])
def test_config_missing(name):
    values=config();del values[name]
    with pytest.raises(ValueError,match=name):create_config(values)

def test_config_conditional_and_extensions():
    values=config(num_nextn_predict_layers=0,num_hidden_layers=1,first_k_dense_replace=1,
                  unrelated_hf_extension="kept")
    for key in ["mtp_loss_factor","moe_aux_loss_coeff","moe_router_bias_update_rate",
                "moe_router_enable_expert_bias","use_pad_tokens"]:del values[key]
    assert create_config(values).unrelated_hf_extension=="kept"
    for name,value in [("mhc_layernorm_epsilon",0),("moe_aux_loss_coeff",float("nan")),
                       ("use_pad_tokens",1),("mtp_loss_factor",-1)]:
        with pytest.raises(ValueError,match=name):create_config(config(**{name:value}))

def test_full_head_chunk_size_and_mla_contract():
    pytest.importorskip("torch_npu")
    from hyper_parallel.models.xing_4.adapter.conversion.module_replacement import fusion_attention_with_logit_maxima
    fake=SimpleNamespace(lm_head=nn.Linear(8,16,bias=False),
        config=SimpleNamespace(vocab_size=16))
    with pytest.raises(ValueError,match="chunk_loss_size"):
        Xing4ForCausalLM._cross_entropy(fake,torch.ones(1,2,8),torch.zeros(1,2,dtype=torch.long))
    for kw in [{"indices":torch.ones(1)},{"dropout":.1}]:
        with pytest.raises(ValueError,match="dense attention"):
            fusion_attention_with_logit_maxima(nn.Module(),None,None,None,None,**kw)

@pytest.mark.parametrize("limit,dtype",[(.5,None),(None,"float32"),(.5,"float32")])
@pytest.mark.parametrize("fused",[True,False])
def test_grouped_policy_binding_and_activation(limit,dtype,fused):
    pytest.importorskip("torch_npu")
    from hyper_parallel.models.xing_4.adapter.conversion.module_replacement import Xing4GroupedExperts
    from hyper_parallel.components.modules.grouped_experts import GroupedExperts
    c=Xing4_0Config(**config(swiglu_limit=limit,swiglu_compute_dtype=dtype,use_fused_swiglu=fused))
    source=nn.ModuleList([Xing4MLP(Xing4_0MLP(c,intermediate_size=16)) for _ in range(2)])
    group=Xing4GroupedExperts(module=source).to(DEVICE)
    x=torch.linspace(-3,3,64,device=DEVICE).reshape(2,32).detach().requires_grad_()
    actual=group.activation_func(x)
    gate,value=x.chunk(2,-1)
    ref_gate=gate.clamp(max=limit) if limit is not None else gate
    ref_value=value.clamp(min=-limit,max=limit) if limit is not None else value
    ref=torch.nn.functional.silu(ref_gate)*ref_value
    torch.testing.assert_close(actual,ref)
    torch.testing.assert_close(torch.autograd.grad(actual.sum(),x,retain_graph=True)[0],
                               torch.autograd.grad(ref.sum(),x)[0])
    # A renamed functional alone cannot make an old GroupedExperts consume the policy.
    with patch.object(GroupedExperts,"_activation",staticmethod(lambda *args:partial(swiglu,dim=-1))):
        with pytest.raises(ValueError,match="must support"):Xing4GroupedExperts(module=source)

def test_default_grouped_policy_accepts_legacy_consumer():
    pytest.importorskip("torch_npu")
    from hyper_parallel.models.xing_4.adapter.conversion.module_replacement import Xing4GroupedExperts
    from hyper_parallel.components.modules.grouped_experts import GroupedExperts
    c=Xing4_0Config(**config());source=nn.ModuleList([Xing4_0MLP(c,intermediate_size=16) for _ in range(2)])
    from hyper_parallel.models.replacement import _validate_forward_compatibility
    with patch.object(GroupedExperts,"_activation",staticmethod(lambda *args:partial(swiglu,dim=-1))):
        replacement=Xing4GroupedExperts(module=source)
    _validate_forward_compatibility(source,replacement,"model.layers.1.mlp.experts")
    with patch.object(Xing4GroupedExperts,"forward",GroupedExperts.forward):
        with pytest.raises(ValueError,match="incompatible forward signature"):
            _validate_forward_compatibility(source,replacement,"model.layers.1.mlp.experts")


def test_hf_partial_load_preserves_router_and_initializes_missing(tmp_path):
    from safetensors.torch import save_file
    values=config(vocab_size=32,num_attention_heads=2,num_key_value_heads=2,q_lora_rank=4,kv_lora_rank=4,qk_rope_head_dim=4,qk_nope_head_dim=4,v_head_dim=4,num_nextn_predict_layers=0,max_position_embeddings=64)
    c=create_config(values)
    model=Xing4ForCausalLM(c)
    gate=model.model.layers[1].mlp.gate
    with torch.no_grad():
        gate.weight.fill_(0.03125);gate.e_score_correction_bias.copy_(torch.tensor([0.50015,-0.50015]));gate._token_counts.fill_(17)
    state={k:v.detach().cpu().contiguous().clone() for k,v in model.state_dict().items()}
    count_key='model.layers.1.mlp.gate._token_counts';bias_key='model.layers.1.mlp.gate.e_score_correction_bias';weight_key='model.layers.1.mlp.gate.weight'
    del state[count_key]
    c.save_pretrained(tmp_path);save_file(state,str(tmp_path/'model.safetensors'),metadata={'format':'pt'})
    loaded,info=Xing4ForCausalLM.from_pretrained(tmp_path,config=c,output_loading_info=True)
    assert count_key in info['missing_keys']
    actual=loaded.state_dict()
    assert torch.equal(actual[bias_key],state[bias_key]) and torch.equal(actual[weight_key],state[weight_key])
    assert actual[count_key].dtype==torch.int64 and not actual[count_key].any()
    loaded.to(DEVICE);actual=loaded.state_dict()
    assert torch.equal(actual[bias_key].cpu(),state[bias_key]) and torch.equal(actual[weight_key].cpu(),state[weight_key])
    assert not actual[count_key].any()


def test_random_initialization_equivalent_without_loaded_flags():
    from hyper_parallel.models.xing_4.adapter.training import initialize
    m=router();owner=SimpleNamespace(config=SimpleNamespace(initializer_range=0.02))
    torch.manual_seed(909);initialize(owner,m);actual=m.weight.detach().clone()
    reference=torch.empty_like(actual);torch.manual_seed(909);nn.init.normal_(reference,mean=0.0,std=0.02)
    assert torch.equal(actual,reference) and not m._token_counts.any() and not m.e_score_correction_bias.any()