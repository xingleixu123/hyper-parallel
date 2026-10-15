"""Explicit optimizer-precision ablations, separate from the original recipe."""

import importlib
import hashlib
import inspect
import textwrap


def install_fp32_newton_schulz() -> None:
    """Retain BF16 optimizer inputs/outputs but evaluate the NS polynomial in FP32."""
    module = importlib.import_module('hyper_parallel.core.optimizer.muon')
    original = module.zeropower_via_newtonschulz5

    def fp32_newton_schulz(inputs, *args, **kwargs):
        result = original(inputs.float(), *args, **kwargs)
        return result.to(inputs.dtype)

    module.zeropower_via_newtonschulz5 = fp32_newton_schulz


def install_fp32_ns_pipeline() -> dict:
    """Diagnostic only: retain FP32 through NS inputs, computation and update relay."""
    module = importlib.import_module('hyper_parallel.core.optimizer.muon')
    evidence = {}
    for name, expected_count in (('_update_muon_momentum', 3), ('_fused_broadcast_and_apply', 2)):
        original = getattr(module.Muon, name)
        source = textwrap.dedent(inspect.getsource(original))
        if source.count('torch.bfloat16') != expected_count:
            raise ValueError(f'The fixed-source dtype sites changed in {name}')
        updated = source.replace('torch.bfloat16', 'torch.float32')
        namespace = dict(module.__dict__)
        exec(compile(updated, f'<diagnostic-fp32-ns:{name}>', 'exec'), namespace)
        replacement = namespace[name]
        setattr(module.Muon, name, replacement)
        evidence[name] = {
            'original_sha256': hashlib.sha256(source.encode()).hexdigest(),
            'modified_sha256': hashlib.sha256(updated.encode()).hexdigest(),
            'dtype_sites': expected_count,
        }
    return evidence
