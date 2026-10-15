"""Bounded diagnostic observations and explicit module-recompute ablations.

These runs are separate from the uninstrumented 200-step acceptance matrix.
Forward allocation deltas include outputs and temporaries; they are not saved
activation sizes. Recompute benefit is measured by independent run-level peaks.
"""

import fnmatch
import json
from collections import defaultdict
from pathlib import Path

import torch

from hyper_parallel.core.activation_memory import checkpoint_wrapper, is_recomputing
from hyper_parallel.models._transformers import model_builder
from hyper_parallel.trainer.callbacks.base import Callback


def install_target_ablation(patterns: list[str], output: Path, rank: int) -> None:
    """Use the normal builder ordering while changing only selected wrap targets."""
    def apply_targets(model, _mode, **_kwargs):
        named = dict(model.named_modules())
        targets = [(name, module) for name, module in named.items()
                   if name and any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)]
        if not targets:
            raise ValueError(f'No recompute modules match {patterns}')
        for name, _ in targets:
            if any(other != name and other.startswith(name + '.') for other, _ in targets):
                raise ValueError(f'Nested recompute targets at {name}')
        for name, module in targets:
            parent, _, key = name.rpartition('.')
            setattr(named[parent], key, checkpoint_wrapper(module))
        (output / f'module-ablation-rank{rank}.json').write_text(json.dumps({
            'patterns': patterns, 'targets': [name for name, _ in targets],
            'kind': 'experiment-only non-reentrant module wrapping before FSDP',
        }, indent=2))
        return model

    model_builder._apply_activation_checkpointing = apply_targets


class ModuleProfileCallback(Callback):
    """Collect events only during a short steady-state diagnostic window."""

    def __init__(self, trainer, output: Path, rank: int):
        super().__init__(trainer)
        self.output = output
        self.rank = rank
        self.enabled = False
        self.pending = defaultdict(list)
        self.records = []
        self.handles = []

    def on_train_begin(self, state, **kwargs):
        suffixes = ('.self_attn', '.mlp', '.shared_experts', '.input_layernorm',
                    '.post_attention_layernorm', '.enorm', '.hnorm', '.eh_proj')
        targets = {}
        for name, module in self.trainer.model.named_modules():
            canonical = name.replace('._wrapped_module', '').replace('._checkpoint_wrapped_module', '')
            if canonical not in ('model.embed_tokens', 'lm_head') and not canonical.endswith(suffixes):
                continue
            # The inner module executes during replay; its wrapper's forward hook does not.
            targets[canonical] = module
        for canonical, module in targets.items():
            self.handles.append(module.register_forward_pre_hook(self._start(canonical)))
            self.handles.append(module.register_forward_hook(self._end(canonical), always_call=True))
        (self.output / f'profile-targets-rank{self.rank}.json').write_text(json.dumps(sorted(targets), indent=2))

    def _start(self, name):
        def hook(_module, _args):
            if not self.enabled:
                return
            event = torch.npu.Event(enable_timing=True)
            event.record()
            self.pending[name].append({
                'name': name, 'phase': 'recompute' if is_recomputing() else 'forward',
                'start': event, 'allocated_before': torch.npu.memory_allocated(),
            })
        return hook

    def _end(self, name):
        def hook(_module, _args, result):
            if not self.enabled:
                return
            record = self.pending[name].pop()
            event = torch.npu.Event(enable_timing=True)
            event.record()
            record.update(end=event, allocated_after=torch.npu.memory_allocated(),
                          completed=result is not None)
            self.records.append(record)
        return hook

    def on_step_begin(self, state, **kwargs):
        self.enabled = 5 <= state.global_step < 10
        self.records.clear()

    def on_step_end(self, state, **kwargs):
        if not self.enabled:
            return
        torch.npu.synchronize()
        with (self.output / f'module-profile-rank{self.rank}.jsonl').open('a') as handle:
            for record in self.records:
                start, end = record.pop('start'), record.pop('end')
                record['elapsed_ms_inclusive'] = start.elapsed_time(end)
                record['allocated_delta_bytes'] = record['allocated_after'] - record['allocated_before']
                record.update(step=state.global_step, rank=self.rank)
                handle.write(json.dumps(record) + '\n')
        self.records.clear()

    def on_train_end(self, state, **kwargs):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
