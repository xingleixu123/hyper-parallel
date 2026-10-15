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

"""Observe both branch weight derivatives and the native shared head clipping boundary."""
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import mindspore as ms
from mindspore import ops, Tensor
from mindformers.core.clip_grad import ClipGradNorm
from mindformers.trainer.base_trainer import BaseTrainer
import mindformers.wrapper.wrapper as wrapper_module

import mf_head_selected_operands_tap as base

ORIGINAL_ATTACH = base.attach
ORIGINAL_OPTIMIZER = BaseTrainer.create_optimizer_scheduler
ORIGINAL_END = base.observed_end


def weight_sink(path: str, indices: Tensor) -> Any:
    """Observe the derivative after the native BF16-to-FP32 cast backward."""
    def sink(gradient):
        ops.TensorDump()(path, ops.gather(ops.reshape(gradient, (-1,)), indices, 0).astype(ms.float32))
        return gradient
    return sink


class ObservedHeadWeights(base.ObservedVocabularyHead):
    """Attach an identity observer to the shared FP32 head parameter at each use."""

    def construct(self, input_: Any, weight: Any = None) -> Any:
        if weight is None:
            weight = self.weight
        return super().construct(input_, self.probe_weight(weight))


class ObservedHeadClip(ClipGradNorm):
    """Sample head gradients before and after the native clipping operation."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.probe_index = base.CONTEXT['parameter_index']
        self.probe_indices = base.CONTEXT['selected']
        self.probe_path = str(base.CONTEXT['directory'] / 'shared_head')

    def construct(self, values: Any) -> Any:
        raw = ops.gather(ops.reshape(values[self.probe_index], (-1,)), self.probe_indices, 0)
        ops.TensorDump()(self.probe_path + '_raw', raw)
        clipped, norm = super().construct(values)
        sampled = ops.gather(ops.reshape(clipped[self.probe_index], (-1,)), self.probe_indices, 0)
        ops.TensorDump()(self.probe_path + '_clipped', sampled)
        ops.TensorDump()(self.probe_path + '_norm', norm)
        return clipped, norm


def attach(network: Any, directory: Path) -> None:
    """Reuse selected operand coordinates and instrument both native weight uses."""
    ORIGINAL_ATTACH(network, directory)
    coordinates = json.loads(Path(
        '/alignment/phase12/evidence/hp-denseadj-vocab204-coordinates-coordinates.json').read_text())
    selected = [item['flat_index'] for row in coordinates['records']
                if row['name'] == 'lm_head.weight' for item in row['coordinates']]
    indices = Tensor(np.asarray(selected, dtype=np.int32))
    base.CONTEXT['selected'] = indices
    for name, module in network.cells_and_names():
        if isinstance(module, base.ObservedVocabularyHead):
            module.__class__ = ObservedHeadWeights
            module.probe_weight = ops.InsertGradientOf(weight_sink(module.probe_path + '_dw', indices))
    (directory / 'weight-observer.json').write_text(json.dumps({
        'indices': selected, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2))


def create_optimizer(self: Any, *args: Any, **kwargs: Any) -> Any:
    """Resolve the actual optimizer-owned shared head index."""
    optimizer = ORIGINAL_OPTIMIZER(self, *args, **kwargs)
    matches = [index for index, parameter in enumerate(optimizer.parameters)
               if parameter.name == 'output_layer.weight']
    if len(matches) != 1:
        raise ValueError('Expected unique shared head optimizer ownership')
    base.CONTEXT['parameter_index'] = matches[0]
    return optimizer


def observed_end(self: Any, run_context: Any) -> None:
    """Associate branch and clipping samples with the same native training step."""
    ORIGINAL_END(self, run_context)
    step = int(run_context.original_args().cur_step_num) + self.state.get('offset', 0)
    prefixes = [row['prefix'] + '_dw' for row in base.CONTEXT['modules']]
    prefixes += ['shared_head_raw', 'shared_head_clipped', 'shared_head_norm']
    records = []
    for prefix in prefixes:
        deadline = time.monotonic() + 30
        while True:
            paths = sorted(set(base.CONTEXT['directory'].glob(prefix + '_*.npy')) - base.SEEN)
            try:
                arrays = [np.load(path, mmap_mode='r') for path in paths]
                ready = len(paths) == 1 and paths[0].stat().st_size == arrays[0].offset + arrays[0].nbytes
            except (ValueError, EOFError, OSError):
                ready = False
            if ready:
                break
            if time.monotonic() > deadline:
                raise TimeoutError(prefix)
            time.sleep(.05)
        records.append({'name': prefix, 'file': paths[0].name, 'values': arrays[0].reshape(-1).tolist()})
        base.SEEN.add(paths[0])
    with (self.output / 'HEAD_WEIGHT_BOUNDARIES.jsonl').open('a') as stream:
        stream.write(json.dumps({'step': step, 'records': records}) + '\n')


base.capture_mf_muon.attach_taps = attach
base.capture_mf_muon.ObserveMetrics.on_train_step_end = observed_end
BaseTrainer.create_optimizer_scheduler = create_optimizer
wrapper_module.ClipGradNorm = ObservedHeadClip
