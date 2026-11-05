"""External CPU probe for MLA callback integration; run from repository root with pytest."""
from copy import deepcopy
from unittest.mock import patch
import pytest
import torch
import torch.nn.functional as F

from hyper_parallel.distributed.context_parallel import mla_context_parallel as cp
from tests.common.mla_cp_utils import TestCPMesh, install_mla_cp, make_mla

def cpu_attention(module, q, k, v, mask, **kw):
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=kw["is_causal"],
                                          scale=kw["scaling"]).transpose(1, 2), None

def run(module, x, callback=None, rotary=None):
    with patch.object(cp, "_validate_state"), patch.object(cp, "npu_fusion_attention_forward", cpu_attention):
        install_mla_cp(module, TestCPMesh(singleton=True), "expanded_ulysses",
                       attention_interface=callback, rotary_fn=rotary)
        result = module(x)[0]
        result.square().sum().backward()
        return result

def test_default_and_custom_callback_output_grad_scale():
    torch.manual_seed(27)
    first = make_mla()
    second = deepcopy(first)
    first.scaling = second.scaling = 0.1234567
    a = torch.randn(1, 7, 12, dtype=torch.float64, requires_grad=True)
    b = a.detach().clone().requires_grad_()
    calls = []
    def attention(module, q, k, v, mask, **kwargs):
        calls.append(kwargs.copy())
        return cpu_attention(module, q, k, v, mask, **kwargs)
    rotations = []
    def rope(tensor, positions, **kwargs):
        rotations.append((tuple(tensor.shape), kwargs["interleaved"]))
        return cp.apply_mla_rope(tensor, positions, **kwargs)
    default = run(first, a)
    custom = run(second, b, attention, rope)
    torch.testing.assert_close(default, custom)
    torch.testing.assert_close(a.grad, b.grad)
    for p, q in zip(first.parameters(), second.parameters()):
        torch.testing.assert_close(p.grad, q.grad)
    assert len(rotations) == 2
    assert calls[0]["scaling"] == 0.1234567
    assert calls[0]["mla_head_offset"] == 0 and calls[0]["mla_local_heads"] == 4

def test_head_offsets_for_ulysses():
    dims = cp.MLADimensions(8, 5, 6, 3, 2, 4)
    plan = cp.MLACPPlan(dims, 4, tp_degree=2)
    class Mesh:
        def size(self): return 4
        def get_local_rank(self): return 3
    seen = []
    def attention(module, q, k, v, mask, **kwargs):
        seen.append(kwargs)
        return q.new_zeros((1, 4, 1, 4)), None
    runtime = cp.MLAContextParallel(plan, Mesh(), attention_interface=attention)
    module = make_mla(dims)
    module.scaling = 0.2468
    q = torch.randn(1, 1, 5)
    kv = torch.randn(1, 1, 6)
    kr = torch.randn(1, 1, 2)
    # Replace communication with representative rank-local tensor shapes.
    with patch.object(cp, "_validate_state"), patch.object(cp, "_query", return_value=torch.zeros(1,4,1,5)), \
         patch.object(runtime, "_expanded", return_value=(torch.zeros(1,1,4,5), torch.zeros(1,1,4,5),
                                                           torch.zeros(1,1,4,4))), \
         patch.object(cp, "ulysses_head_to_seq", return_value=torch.zeros(1,1,4,4)):
        runtime(module, q, kv, kr)
    assert seen[0]["mla_head_offset"] == 3
    assert seen[0]["mla_local_heads"] == 4
    assert seen[0]["scaling"] == 0.2468

@pytest.mark.parametrize("kwargs", [
    dict(cp_degree=3), dict(cp_degree=4, tp_degree=2, ulysses_degree=1),
    dict(cp_degree=4, kv_exchange="all_to_all_v"), dict(cp_degree=4, overlap="yes"),
])
def test_invalid_plans(kwargs):
    with pytest.raises(ValueError):
        cp.MLACPPlan(cp.MLADimensions(8, 5, 6, 3, 2, 4), **kwargs)

def test_default_extra_kwargs_context_equivalence():
    from hyper_parallel.components.functional.npu_fusion_attention import _prepare_fusion_attention_context
    q = torch.zeros(1, 2, 3, 4)
    base = dict(is_causal=True, actual_seq_qlen=[3], actual_seq_kvlen=[3])
    a = _prepare_fusion_attention_context(torch.nn.Module(), q, q, q, None, base)
    b = _prepare_fusion_attention_context(torch.nn.Module(), q, q, q, None,
                                           {**base, "mla_head_offset": 2, "mla_local_heads": 4})
    for name in ("input_layout", "sparse_mode", "query_lengths", "key_lengths", "pre_tokens", "next_tokens"):
        assert getattr(a, name) == getattr(b, name)
    torch.testing.assert_close(a.attention_mask, b.attention_mask)
