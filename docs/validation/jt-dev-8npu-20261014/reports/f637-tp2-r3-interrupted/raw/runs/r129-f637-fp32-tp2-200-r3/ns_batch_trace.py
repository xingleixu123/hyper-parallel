"""Observe Muon's actual matrix batching without retaining or changing tensors."""

from functools import wraps
import json
from pathlib import Path

from hyper_parallel.core.optimizer.muon import Muon


def install_ns_batch_trace(output: Path, rank: int):
    original_updates = Muon._compute_batched_ns_updates
    original_outputs = Muon._compute_batched_ns_outputs_for_tensors
    context = {}

    @wraps(original_updates)
    def observe_updates(self, p_list, ns_inputs, group, no_shard=False):
        identity = id(self)
        if identity in context:
            raise RuntimeError('Unexpected nested optimizer NS batch')
        context[identity] = {
            'step': group.get('step'), 'no_shard': no_shard,
            'parameters': [{'name': getattr(p, 'model_name', None), 'shape': list(p.shape)} for p in p_list],
        }
        try:
            return original_updates(self, p_list, ns_inputs, group, no_shard)
        finally:
            del context[identity]

    @wraps(original_outputs)
    def observe_outputs(self, tensor_list, *args, **kwargs):
        shapes = [list(t.shape) for t in tensor_list]
        count = sum(1 if len(shape) == 2 or (len(shape) == 3 and shape[1] == 1)
                    else shape[0] if len(shape) == 3 else t.numel() // (shape[-2] * shape[-1])
                    for shape, t in zip(shapes, tensor_list))
        row = {**context.get(id(self), {}), 'rank': rank, 'input_shapes': shapes,
               'input_dtypes': [str(t.dtype) for t in tensor_list],
               'merged_matrix_count': count, 'squeezed_to_2d': count == 1,
               'scope': 'Metadata only; no tensor data copied and no arithmetic changed'}
        with (output / f'ns-batches-rank{rank}.jsonl').open('a') as handle:
            handle.write(json.dumps(row) + '\n')
        return original_outputs(self, tensor_list, *args, **kwargs)

    Muon._compute_batched_ns_updates = observe_updates
    Muon._compute_batched_ns_outputs_for_tensors = observe_outputs
