"""Bounded experiment-only expert gradients, global samples, and routing state."""

import json
from pathlib import Path

import torch

from hyper_parallel.models.validation_spec import ParameterProbeSpec
from hyper_parallel.tools.model_integration.tensor_probes import ParameterProbeRecorder


def install_expert_probes(output: Path, rank: int) -> None:
    """Extend runtime observations without altering training tensors."""
    resolve = ParameterProbeRecorder._resolve_probes
    capture = ParameterProbeRecorder.capture

    def resolve_with_experts(self):
        probes = resolve(self)
        for name in self._named_parameters:
            if '.mlp.' in name:
                probes[name] = ParameterProbeSpec(match=name, stages=(
                    'main_before', 'main_grad_before_clip', 'main_grad_after_clip',
                    'main_after_optimizer', 'optimizer_state'))
        return dict(sorted(probes.items()))

    @torch.no_grad()
    def capture_with_samples(self, stage):
        capture(self, stage)
        if stage not in ('main_before', 'main_grad_before_clip', 'main_after_optimizer'):
            return
        records = []
        for name, parameter in sorted(self._named_parameters.items()):
            if '.mlp.experts.' not in name and not name.endswith('.mlp.gate.weight'):
                continue
            tensor = self._stage_tensor(parameter, stage)
            if tensor is None:
                raise RuntimeError(f'Missing expert tensor: {name}, {stage}')
            full = tensor.full_tensor() if hasattr(tensor, 'full_tensor') else tensor
            # Reconstruct global coordinates before subsampling, independent of EP/FSDP layout.
            selection = (slice(None), slice(None, None, 64), slice(None, None, 64)) if full.ndim == 3 else (slice(None), slice(None, None, 16))
            sample = full[selection].float().cpu().contiguous()
            if rank == 0:
                records.append({'step': self._step, 'stage': stage, 'name': name,
                                'global_shape': list(full.shape), 'sample_shape': list(sample.shape),
                                'strides': [1, 64, 64] if full.ndim == 3 else [1, 16],
                                'values': sample.reshape(-1).tolist()})
            del full, sample
        if rank == 0:
            with (output / 'expert-samples.jsonl').open('a') as handle:
                for row in records:
                    handle.write(json.dumps(row) + '\n')
        routing = []
        for name, module in self.model.named_modules():
            if not hasattr(module, 'tokens_per_expert'):
                continue
            counts = module.tokens_per_expert
            bias = module.gate.e_score_correction_bias
            routing.append({'name': name, 'counts': None if counts is None else counts.detach().cpu().tolist(),
                            'bias': bias.detach().cpu().tolist()})
        with (output / f'routing-rank{rank}.jsonl').open('a') as handle:
            handle.write(json.dumps({'step': self._step, 'stage': stage, 'modules': routing}) + '\n')

    ParameterProbeRecorder._resolve_probes = resolve_with_experts
    ParameterProbeRecorder.capture = capture_with_samples
