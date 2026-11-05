"""External CPU regression for MLA CP pairwise-mask normalization."""
from copy import deepcopy
from unittest.mock import patch
import pytest
import torch
import torch.nn.functional as F

from hyper_parallel.distributed.context_parallel import mla_context_parallel as cp
from tests.common.mla_cp_utils import TestCPMesh, install_mla_cp, make_mla

def run(module, inputs, mask):
    seen = []
    def attention(_module, query, key, value, received_mask, **kwargs):
        seen.append(received_mask.clone())
        return F.scaled_dot_product_attention(
            query, key, value, attn_mask=received_mask,
            is_causal=kwargs["is_causal"], scale=kwargs["scaling"],
        ).transpose(1, 2), None
    with patch.object(cp, "_validate_state"), patch.object(cp, "npu_fusion_attention_forward", attention):
        install_mla_cp(module, TestCPMesh(singleton=True), "expanded_ulysses")
        output = module(inputs, attention_mask=mask, is_causal=False)[0]
        output.square().sum().backward()
    return output, seen

def test_pairwise_2d_4d_forward_backward_and_attention_mask():
    torch.manual_seed(413)
    a = make_mla()
    b = deepcopy(a)
    x = torch.randn(2, 5, 12, dtype=torch.float64, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    mask = torch.ones(5, 5, dtype=torch.bool).tril()
    actual, captured_a = run(a, x, mask)
    expected, captured_b = run(b, y, mask[None, None])
    assert len(captured_a) == len(captured_b) == 1
    assert captured_a[0].shape == (1, 1, 5, 5)
    torch.testing.assert_close(captured_a[0], captured_b[0])
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(x.grad, y.grad)
    for first, second in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(first.grad, second.grad)

def test_rejected_mask_contracts_unchanged():
    model = make_mla()
    install_mla_cp(model, TestCPMesh(singleton=True), "expanded_ulysses")
    x = torch.randn(1, 5, 12, dtype=torch.float64)
    with pytest.raises(ValueError, match="global Q/K"):
        model(x, attention_mask=torch.ones(5, 4, dtype=torch.bool))
    with pytest.raises(ValueError, match="packed MLA CP"):
        model(x, attention_mask=torch.ones(5, 5, dtype=torch.bool), actual_seq_len=(5,))
