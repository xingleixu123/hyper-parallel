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

"""Observe native final RMSNorm operands and derivatives at the226 frontier."""
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import mindspore as ms
from mindspore import ops
from mindformers.parallel_core.training_graph.transformer.norm import FusedRMSNorm
from mindformers.parallel_core.training_graph.transformer.moe.moe_layer import MoELayer

import capture_mf_muon

CONTEXT = {}
SEEN = set()
TARGETS = ['model.decoder.final_layernorm', 'model.mtp.layers.0.final_layernorm', 'model.decoder.layers.1.mlp']


def gradient_sink(path: str) -> Any:
    """Observe a full branch derivative and preserve the original graph value."""
    def sink(gradient):
        ops.TensorDump()(path, gradient.astype(ms.float32))
        return gradient
    return sink


class ObservedFinalNorm(FusedRMSNorm):
    """Preserve the actual native norm primitive, affine parameter and casts."""

    def construct(self, values: Any) -> Any:
        values = self.probe_dx(values)
        inputs = self.cast(values, self.compute_type)
        normalized, inverse_rms = self.norm(inputs, self.weight)
        result = self.cast(normalized, values.dtype)
        ops.TensorDump()(self.probe_path + '_x', inputs.astype(ms.float32))
        ops.TensorDump()(self.probe_path + '_y_fp32', normalized.astype(ms.float32))
        ops.TensorDump()(self.probe_path + '_inverse_rms', inverse_rms.astype(ms.float32))
        ops.TensorDump()(self.probe_path + '_weight', self.weight.astype(ms.float32))
        return self.probe_dy(result)


class ObservedMainMoE(MoELayer):
    """Preserve native main MoE values and expose only branch derivatives."""

    def construct(self, hidden_states, extra_loss=0., seq_chunk=None):
        hidden_states = self.probe_dx(hidden_states)
        ops.TensorDump()(self.probe_path + '_x', hidden_states.astype(ms.float32))
        result, bias, loss = super().construct(hidden_states, extra_loss, seq_chunk)
        ops.TensorDump()(self.probe_path + '_y_fp32', result.astype(ms.float32))
        return self.probe_dy(result), bias, loss


def attach(network: Any, directory: Path) -> None:
    """Observe only the two final norm branches and retain full arrays only at226."""
    directory.mkdir(parents=True, exist_ok=False)
    modules = dict(network.cells_and_names())
    records = []
    for name in TARGETS:
        module = modules[name]
        is_moe = name.endswith('.mlp')
        if (is_moe and type(module) is not MoELayer) or (
                not is_moe and (type(module) is not FusedRMSNorm or module.dim != 5120)):
            raise ValueError('Unexpected native module class')
        prefix = name.replace('.', '_')
        kinds = ['x', 'y_fp32', 'dy', 'dx'] if is_moe else ['x', 'y_fp32', 'inverse_rms', 'weight', 'dy', 'dx']
        records.append({'name': name, 'prefix': prefix, 'kinds': kinds})
        module.__class__ = ObservedMainMoE if is_moe else ObservedFinalNorm
        module.probe_path = str(directory / prefix)
        module.probe_dx = ops.InsertGradientOf(gradient_sink(module.probe_path + '_dx'))
        module.probe_dy = ops.InsertGradientOf(gradient_sink(module.probe_path + '_dy'))
    CONTEXT.update(directory=directory, modules=records)
    (directory / 'manifest.json').write_text(json.dumps({
        'modules': records, 'retained_steps': [226], 'teacher_values_used': False,
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2))


ORIGINAL_END = capture_mf_muon.ObserveMetrics.on_train_step_end


def observed_end(self: Any, run_context: Any) -> None:
    """Hash each executed observation and promptly remove unneeded full arrays."""
    ORIGINAL_END(self, run_context)
    step = int(run_context.original_args().cur_step_num) + self.state.get('offset', 0)
    records = []
    for module in CONTEXT['modules']:
        for kind in module['kinds']:
            deadline = time.monotonic() + 30
            while True:
                paths = sorted(set(CONTEXT['directory'].glob(module['prefix'] + '_' + kind + '_*.npy')) - SEEN)
                try:
                    arrays = [np.load(path, mmap_mode='r') for path in paths]
                    ready = paths and all(path.stat().st_size == array.offset + array.nbytes
                                          for path, array in zip(paths, arrays))
                except (ValueError, EOFError, OSError):
                    ready = False
                if ready:
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError(module['name'] + ':' + kind)
                time.sleep(.05)
            hashes = {hashlib.sha256(array.tobytes()).hexdigest() for array in arrays}
            if len(hashes) != 1:
                raise ValueError('Recomputed norm values differ')
            for path, array in zip(paths, arrays):
                records.append({'name': module['name'], 'kind': kind, 'file': path.name,
                                'shape': list(array.shape), 'sha256': next(iter(hashes)), 'retained': step == 226,
                                'logical_bytes': path.stat().st_size, 'physical_bytes': path.stat().st_blocks * 512})
                SEEN.add(path)
            del arrays, array
            if step != 226:
                for path in paths:
                    if path.is_symlink() or path.resolve().parent != CONTEXT['directory'].resolve():
                        raise ValueError('Unsafe task-owned temporary path')
                    path.unlink()
    with (self.output / 'BACKWARD226_BOUNDARIES.jsonl').open('a') as stream:
        stream.write(json.dumps({'step': step, 'records': records}) + '\n')


capture_mf_muon.attach_taps = attach
capture_mf_muon.ObserveMetrics.on_train_step_end = observed_end
