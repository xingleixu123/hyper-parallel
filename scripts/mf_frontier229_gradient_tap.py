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

"""Observe the actual clipped MTP Q-up gradient without changing native Muon arithmetic."""
import json
from pathlib import Path
from mindspore import ops
from mindformers.core.optim.muon import Muon


TARGETS = {'mtp.layers.0.transformer_layer.self_attention.linear_q_up_proj.weight'}


class ObservedMuon(Muon):
    """Write optimizer inputs before invoking unchanged native arithmetic."""
    def construct(self, gradients):
        for index in self.gradient_probe_indices:
            ops.TensorDump()(self.gradient_probe_paths[index], gradients[index])
        return super().construct(gradients)


def attach(optimizer, destination, max_elements=None):
    """Attach a full-state diagnostic to native Muon only."""
    if type(optimizer) is not Muon:
        raise TypeError(f'Unexpected optimizer: {type(optimizer)}')
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    parameters = list(optimizer.parameters)
    if max_elements is not None and max_elements < 1:
        raise ValueError('Gradient observation size limit must be positive')
    selected = [index for index, parameter in enumerate(parameters)
                if parameter.name in TARGETS]
    if len(selected) != 1 or {parameters[index].name for index in selected} != TARGETS:
        raise ValueError('Expected exactly one native gradient target')
    optimizer.__class__ = ObservedMuon
    optimizer.gradient_probe_indices = tuple(selected)
    optimizer.gradient_probe_paths = tuple(str(destination / f'{index:04d}') for index in range(len(parameters)))
    (destination / 'manifest.json').write_text(json.dumps([
        {'name': parameter.name, 'prefix': f'{index:04d}', 'shape': list(parameter.shape)}
        for index, parameter in enumerate(parameters) if index in selected], indent=2))
