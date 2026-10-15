# Copyright 2026 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Observe native dispatcher operands; retain only212 and prune earlier dumps."""
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import mindspore as ms
from mindspore import ops
from mindformers.parallel_core.training_graph.transformer.moe.token_dispatcher import MoEAlltoAllTokenDispatcher

import mf_router212_scores_tap as scores

ORIGINAL_ATTACH, ORIGINAL_END = scores.attach, scores.observed_end
CONTEXT: dict[str, Any] = {}
SEEN: set[Path] = set()
KINDS = ['tokens', 'unsort', 'outer_unsort', 'probs', 'dy']


class ObservedDispatcher(MoEAlltoAllTokenDispatcher):
    """Keep native unpermutation and expose its actual operands and output derivative."""

    def token_unpermutation(self, tokens: Any, ctx: Any) -> Any:
        ops.TensorDump()(self.weighted_probe_path + '_tokens_value', tokens.astype(ms.float32))
        ops.TensorDump()(self.weighted_probe_path + '_unsort_value', ctx[1])
        ops.TensorDump()(self.weighted_probe_path + '_outer_unsort_value', ctx[2])
        ops.TensorDump()(self.weighted_probe_path + '_probs_value', ctx[0].astype(ms.float32))
        return self.weighted_probe_dy(super().token_unpermutation(tokens, ctx))


def attach(network: Any, directory: Path) -> None:
    """Add one full main-expert operand observer to the audited score observer."""
    ORIGINAL_ATTACH(network, directory)
    module = dict(network.cells_and_names())['model.decoder.layers.1.mlp.experts'].token_dispatcher
    if type(module) is not MoEAlltoAllTokenDispatcher:
        raise ValueError('Unexpected main dispatcher')
    module.__class__ = ObservedDispatcher
    module.weighted_probe_path = str(directory / 'weighted_product')
    module.weighted_probe_dy = ops.InsertGradientOf(scores.base.gradient_sink(
        module.weighted_probe_path + '_dy_value'))
    CONTEXT['directory'] = directory
    CONTEXT['padding'] = module.pad_routing_map.shape[1] if module.use_pad_tokens else 0
    CONTEXT['moe_permute_fusion'] = module.moe_permute_fusion


def observed_end(self: Any, run_context: Any) -> None:
    """Verify dumps completed and retain only the requested native step."""
    ORIGINAL_END(self, run_context)
    step = int(run_context.original_args().cur_step_num) + self.state.get('offset', 0)
    records = []
    for kind in KINDS:
        deadline = time.monotonic() + 45
        while True:
            paths = sorted(set(CONTEXT['directory'].glob('weighted_product_' + kind + '_value_*.npy')) - SEEN)
            try:
                values = [np.load(path, mmap_mode='r') for path in paths]
                ready = paths and all(path.stat().st_size == value.offset + value.nbytes
                                      for path, value in zip(paths, values))
            except (ValueError, EOFError, OSError):
                ready = False
            if ready:
                break
            if time.monotonic() > deadline:
                raise TimeoutError('Weighted boundary: ' + kind)
            time.sleep(.05)
        hashes = {hashlib.sha256(value.tobytes()).hexdigest() for value in values}
        if len(hashes) != 1:
            raise ValueError('Recomputed weighted boundary changed')
        for path, value in zip(paths, values):
            records.append({'name': kind, 'file': path.name, 'shape': list(value.shape),
                            'dtype': str(value.dtype), 'sha256': next(iter(hashes)), 'retained': step == 212})
            SEEN.add(path)
        del values, value
        if step != 212:
            for path in paths:
                if path.is_symlink() or path.resolve().parent != CONTEXT['directory'].resolve():
                    raise ValueError('Unsafe weighted temporary path')
                path.unlink()
    report = {'step': step, 'records': records, 'readonly': True, 'padding': CONTEXT['padding'],
              'moe_permute_fusion': CONTEXT['moe_permute_fusion'],
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (self.output / ('WEIGHTED_PRODUCT-' + str(step) + '.json')).write_text(json.dumps(report, indent=2))


scores.base.capture_mf_muon.attach_taps = attach
scores.base.capture_mf_muon.ObserveMetrics.on_train_step_end = observed_end
