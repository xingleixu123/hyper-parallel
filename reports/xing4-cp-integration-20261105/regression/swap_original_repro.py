"""Compare actual old and candidate swap paths with real CPU optimizer adapters."""
import argparse, hashlib, importlib.util, sys
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument('--repo', required=True, type=Path)
a = parser.parse_args()
sys.path.insert(0, str(a.repo.resolve()))
import torch
from hyper_parallel.core.optimizer.muon import Muon
from hyper_parallel.core.optimizer.swap_optimizer import SwapOptimizerConfig
from hyper_parallel.core.optimizer.swap_optimizer_base import SwapOptimizer as Candidate
from hyper_parallel.core.optimizer.swap_muon import build_muon_swap_adapter
from hyper_parallel.core.optimizer.swap_adam import build_adam_swap_adapter
import hyper_parallel.core.optimizer.swap_optimizer_base as candidate_module
old_path = Path(__file__).with_name('swap_optimizer_base_original.py')
spec = importlib.util.spec_from_file_location('old_swap_repro', old_path)
old_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = old_module
spec.loader.exec_module(old_module)
for label, path in [('old', old_path), ('candidate', Path(candidate_module.__file__))]:
    print('SOURCE', label, path, hashlib.sha256(path.read_bytes()).hexdigest(), flush=True)
print('torch', torch.__version__, 'device cpu', flush=True)
config = SwapOptimizerConfig(swap_times=1, min_numel=1024, packed_swap=False)
for family, cls, factory in [('AdamW', torch.optim.AdamW, build_adam_swap_adapter),
                             ('Muon', Muon, build_muon_swap_adapter)]:
    def make():
        p = torch.nn.Parameter(torch.ones(2, 2))
        return p, cls([p])
    def attach(opt, counts):
        opt.register_step_pre_hook(lambda *_: counts.__setitem__('pre', counts['pre'] + 1))
        opt.register_step_post_hook(lambda *_: counts.__setitem__('post', counts['post'] + 1))
    # Verify that the same native hooks are functional on an ordinary leaf step.
    param, leaf = make(); counts = {'pre': 0, 'post': 0}; attach(leaf, counts)
    param.grad = torch.ones_like(param); leaf.step()
    assert counts == {'pre': 1, 'post': 1}
    print(f'{family} native leaf: pre={counts["pre"]} post={counts["post"]}', flush=True)
    param, leaf = make(); counts = {'pre': 0, 'post': 0}; attach(leaf, counts)
    wrapper = old_module.SwapOptimizer(leaf, config, factory)
    before = param.detach().clone(); param.grad = torch.ones_like(param); wrapper.step()
    changed = not torch.equal(before, param)
    print(f'{family} original swap: pre={counts["pre"]} post={counts["post"]} parameter_changed={changed}', flush=True)
    assert counts == {'pre': 0, 'post': 0} and changed
    param, leaf = make(); counts = {'pre': 0, 'post': 0}; attach(leaf, counts)
    try:
        Candidate(leaf, config, factory)
    except ValueError as exc:
        print(f'{family} candidate constructor rejects: {exc}', flush=True)
    else:
        raise AssertionError(f'{family} candidate failed to reject existing hooks')
print('PASS: real CPU adapters reproduce bypass and candidate rejects pre-existing hooks', flush=True)
