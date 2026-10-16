import torch,torch_npu
from hyper_parallel.models.jt_deepseek_v3.mf_archive.legacy_gemm import head_weight_gradient
torch.npu.set_device(0)
torch.npu.config.allow_internal_format=False
torch_npu.npu.set_compile_mode(jit_compile=False)
a=torch.arange(32*64,dtype=torch.float32).reshape(32,64).remainder(5).to(device='npu',dtype=torch.bfloat16)
b=torch.ones(32,32,device='npu',dtype=torch.bfloat16)
y=head_weight_gradient(a,b)
expected=(a.cpu().float().T@b.cpu().float()).bfloat16()
torch.testing.assert_close(y.cpu(),expected,rtol=0,atol=0)
assert torch_npu.npu.is_jit_compile_false()
assert torch_npu._C._npu_getOption('ALLOW_INTERNAL_FORMAT')==b'disable'
print('ARCHIVED_HEAD_MICRO_OK',list(y.shape),'compiler and format flags restored')
