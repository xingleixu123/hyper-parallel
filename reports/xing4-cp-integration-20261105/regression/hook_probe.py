"""External regression for Xing QKClip leaf hook and swap hook guard.

Run with candidate checkout and its declared runtime dependencies on PYTHONPATH:
  python hook_probe.py --repo /path/to/candidate --device cpu
  python hook_probe.py --repo /path/to/candidate --device npu:0
Does not modify source or initialize distributed process groups.
"""
import argparse
import hashlib
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--repo', required=True, type=Path)
parser.add_argument('--device', default='cpu')
a = parser.parse_args()
sys.path.insert(0, str(a.repo.resolve()))
import torch
if a.device.startswith('npu'):
    import torch_npu  # noqa: F401
    torch.npu.set_device(a.device)
from torch import nn
from hyper_parallel.models.xing_4.adapter.optim.muon import build_optimizer
from hyper_parallel.components.optim.mixed_precision_optimizer import Float16OptimizerWithFloat16Params
from hyper_parallel.core.optimizer.muon import Muon
from hyper_parallel.core.optimizer.swap_optimizer import SwapOptimizerConfig, swap_optimizer
from hyper_parallel.core.optimizer.swap_muon import swap_muon
from hyper_parallel.core.optimizer.swap_adam import swap_adam

DEVICE = torch.device(a.device)
for rel in ('hyper_parallel/models/xing_4/adapter/optim/muon.py',
            'hyper_parallel/core/optimizer/swap_optimizer_base.py'):
    f = a.repo / rel
    print('SOURCE', f, hashlib.sha256(f.read_bytes()).hexdigest(), flush=True)
print('ENV', torch.__version__, DEVICE, flush=True)

class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_b_proj = nn.Linear(2, 6, bias=False, dtype=torch.bfloat16)
        self.kv_b_proj = nn.Linear(2, 6, bias=False, dtype=torch.bfloat16)
        self.qk_nope_head_dim, self.qk_rope_head_dim, self.v_head_dim = 2, 1, 1
        self.max_logits = None

class Model(nn.Module):
    def __init__(self, adam_owned=False, empty=False):
        super().__init__()
        if not empty:
            # "embed" is a default AdamW name classifier, giving an ownership counterexample.
            self.add_module('embed_attention' if adam_owned else 'attention', Attention())
        self.other = nn.Linear(2, 2, bias=True, dtype=torch.bfloat16)
    def attn(self):
        return getattr(self, 'attention', getattr(self, 'embed_attention', None))

def build(model, clip):
    return build_optimizer(model=model, muon_config={'muon_weight_decay': 0.0},
                           adamw_config={'adamw_weight_decay': 0.0}, qk_clip_threshold=clip)

def must_reject(fn, label):
    try:
        fn()
    except (ValueError, RuntimeError, NotImplementedError) as error:
        print('PASS', label, type(error).__name__, str(error), flush=True)
    else:
        raise AssertionError(label + ': expected explicit rejection')

# These are real builders and leaves; the hook must remain attached after FP32 rebinding.
torch.manual_seed(99)
base = Model().to(DEVICE)
torch.manual_seed(99)
clipped = Model().to(DEVICE)
base_chain, clip_chain = build(base, 0).get_optimizer(), build(clipped, 100).get_optimizer()
assert list(clip_chain.optimizers_dict)[-1] == 'muon'
leaf = clip_chain.optimizers_dict['muon']
assert isinstance(leaf, Muon) and len(leaf._optimizer_step_post_hooks) >= 1
bopt = Float16OptimizerWithFloat16Params(base_chain, base)
copt = Float16OptimizerWithFloat16Params(clip_chain, clipped)
assert id(clipped.attn().q_b_proj.weight.main_param) in {id(p) for g in leaf.param_groups for p in g['params']}
for step in range(3):
    for model in (base, clipped):
        model.attn().max_logits = torch.tensor([200., 50.], device=DEVICE)
        for param in model.parameters():
            param.grad = torch.ones_like(param)
    bopt.step(); copt.step()
    scale = torch.tensor([0.5, 1.0], device=DEVICE)
    qf = torch.stack((scale.sqrt(), scale.sqrt(), scale), dim=1).reshape(-1, 1)
    kvf = torch.stack((scale.sqrt(), scale.sqrt(), torch.ones_like(scale)), dim=1).reshape(-1, 1)
    # Each run diverges after step one. On first update the independent no-clip
    # twin is the exact pre-hook main-weight value; then check model copyback.
    if step == 0:
        for pname, factors in (('q_b_proj', qf), ('kv_b_proj', kvf)):
            raw = getattr(base.attn(), pname).weight.main_param
            actual = getattr(clipped.attn(), pname).weight.main_param
            torch.testing.assert_close(actual, raw * factors, atol=0, rtol=0)
    for p in clipped.parameters():
        assert torch.equal(p, p.main_param.to(p.dtype))
    assert clipped.attn().max_logits is None
print('PASS native Muon hook, FP32 rebinding, factor oracle and three copybacks', flush=True)

frozen = Model().to(DEVICE)
frozen.attn().q_b_proj.weight.requires_grad_(False)
must_reject(lambda: build(frozen, 100), 'frozen projection')
must_reject(lambda: build(Model(adam_owned=True).to(DEVICE), 100), 'AdamW projection')
must_reject(lambda: build(Model(empty=True).to(DEVICE), 100), 'missing attention')
# QKClip disabled must not impose the clipping-only guard.
build(Model(adam_owned=True).to(DEVICE), 0)
build(Model(empty=True).to(DEVICE), 0)
print('PASS disabled-QKClip ownership and empty paths', flush=True)

config = SwapOptimizerConfig(swap_times=1, min_numel=1024, packed_swap=False)
def fresh_muon():
    return Muon([nn.Parameter(torch.ones(2, 2, device=DEVICE))])
def fresh_adam():
    return torch.optim.AdamW([nn.Parameter(torch.ones(2, 2, device=DEVICE))])
for label, factory, fresh in (('dispatch-muon', swap_optimizer, fresh_muon),
                              ('direct-muon', swap_muon, fresh_muon),
                              ('dispatch-adam', swap_optimizer, fresh_adam),
                              ('direct-adam', swap_adam, fresh_adam)):
    for kind in ('pre', 'post'):
        opt = fresh()
        getattr(opt, 'register_step_' + kind + '_hook')(lambda *args: None)
        must_reject(lambda: factory(opt, config), label + ' existing ' + kind + ' hook')
    # Build and step a hook-free wrapper using real adapters; no model-specific guard.
    opt = fresh()
    wrapper = factory(opt, config)
    for group in wrapper.param_groups:
        for param in group['params']:
            param.grad = torch.ones_like(param)
    wrapper.step()
    print('PASS', label, 'hook-free constructor and step', flush=True)
print('ALL PASS', flush=True)
