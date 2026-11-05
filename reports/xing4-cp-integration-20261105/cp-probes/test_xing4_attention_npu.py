"""External NPU TND Xing4 callback test. Launch with torchrun -m pytest on 2 or 4 NPUs."""
import os
from datetime import timedelta
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch_npu  # noqa: F401

from hyper_parallel.models.xing_4.adapter.conversion.module_replacement import fusion_attention_with_logit_maxima

def test_xing4_head_max_output_gradient():
    rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    assert world in (2, 4)
    torch.npu.set_device(rank)
    dist.init_process_group("hccl", timeout=timedelta(seconds=180))
    try:
        torch.manual_seed(412)
        device = torch.device("npu", rank)
        heads, seq, width, value_width = 8, 32, 192, 128
        per_rank = heads // world
        dtype = torch.bfloat16
        src = [torch.randn(1, heads, seq, d, dtype=dtype).to(device) for d in (width, width, value_width)]
        q, k, v = [t[:, rank * per_rank:(rank + 1) * per_rank].detach().clone().requires_grad_() for t in src]
        ref_q, ref_k, ref_v = [x.detach().float().requires_grad_() for x in (q, k, v)]
        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.max_logits = None
                self.is_causal = True
        module = Module()
        scale = 0.73 / width**0.5
        out, _ = fusion_attention_with_logit_maxima(
            module, q, k, v, None, scaling=scale, is_causal=True,
            mla_head_offset=rank * per_rank, mla_local_heads=heads,
        )
        ref = F.scaled_dot_product_attention(ref_q, ref_k, ref_v, is_causal=True, scale=scale).transpose(1, 2)
        torch.testing.assert_close(out.float(), ref, atol=0.02, rtol=0.08)
        torch.manual_seed(919)
        upstream = torch.randn(ref.shape, dtype=torch.float32).to(device)
        (out.float() * upstream).sum().backward()
        (ref * upstream).sum().backward()
        for actual, expected in zip((q.grad, k.grad, v.grad), (ref_q.grad, ref_k.grad, ref_v.grad)):
            torch.testing.assert_close(actual.float(), expected, atol=0.04, rtol=0.12)
        scores = torch.matmul(ref_q.detach(), ref_k.detach().transpose(-1, -2)) * scale
        causal = torch.ones(seq, seq, device=device, dtype=torch.bool).tril()
        expected_max = scores.masked_fill(~causal, -torch.inf).amax(dim=(0, 2, 3))
        local_max = module.max_logits[rank * per_rank:(rank + 1) * per_rank]
        torch.testing.assert_close(local_max.float(), expected_max, atol=0.04, rtol=0.05)
        inactive = torch.cat((module.max_logits[:rank * per_rank], module.max_logits[(rank + 1) * per_rank:]))
        assert torch.isneginf(inactive).all()
        combined = module.max_logits.clone()
        dist.all_reduce(combined, op=dist.ReduceOp.MAX)
        assert torch.isfinite(combined).all()
        print(f"Xing4 callback rank={rank} world={world} scale={scale} output/grad/max=pass", flush=True)
    finally:
        dist.destroy_process_group()
