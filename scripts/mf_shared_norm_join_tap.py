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

"""Observe native final RMSNorm operands and derivatives at the205 frontier."""
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import mindspore as ms
from mindspore import ops
from mindformers.parallel_core.training_graph.transformer.norm import FusedRMSNorm
from mindformers.parallel_core.training_graph.transformer.transformer_block import TransformerBlock

import capture_mf_muon

CONTEXT = {}
SEEN = set()
TARGETS = ['model.decoder.final_layernorm', 'model.mtp.layers.0.hnorm']


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


class ObservedSharedDecoder(TransformerBlock):
    """Observe the actual common output consumed by LM and MTP."""

    def construct(self, hidden_states, attention_mask, rotary_pos_emb=None,
                  prefix_keys_values=None, actual_seq_len=None):
        hidden, extra_loss = super().construct(hidden_states, attention_mask, rotary_pos_emb,
                                               prefix_keys_values, actual_seq_len)
        ops.TensorDump()(self.probe_path + '_y', hidden.astype(ms.float32))
        return self.probe_dy(hidden), extra_loss


def attach(network: Any, directory: Path) -> None:
    """Observe only the two final norm branches and retain full arrays only at205."""
    directory.mkdir(parents=True, exist_ok=False)
    modules = dict(network.cells_and_names())
    records = []
    for name in TARGETS:
        module = modules[name]
        if type(module) is not FusedRMSNorm or module.dim != 5120:
            raise ValueError('Unexpected final norm: ' + repr(type(module)))
        prefix = name.replace('.', '_')
        records.append({'name': name, 'prefix': prefix, 'epsilon': module.eps,
                        'compute_type': str(module.compute_type), 'weight_type': str(module.weight.dtype)})
        module.__class__ = ObservedFinalNorm
        module.probe_path = str(directory / prefix)
        module.probe_dx = ops.InsertGradientOf(gradient_sink(module.probe_path + '_dx'))
        module.probe_dy = ops.InsertGradientOf(gradient_sink(module.probe_path + '_dy'))
    decoder = modules['model.decoder']
    if type(decoder) is not TransformerBlock:
        raise ValueError('Unexpected decoder block')
    decoder.__class__ = ObservedSharedDecoder
    decoder.probe_path = str(directory / 'shared_decoder')
    decoder.probe_dy = ops.InsertGradientOf(gradient_sink(decoder.probe_path + '_dy'))
    records.append({'name': 'model.decoder', 'prefix': 'shared_decoder', 'kinds': ['y', 'dy']})
    CONTEXT.update(directory=directory, modules=records)
    (directory / 'manifest.json').write_text(json.dumps({
        'modules': records, 'retained_steps': [205], 'teacher_values_used': False,
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2))


ORIGINAL_END = capture_mf_muon.ObserveMetrics.on_train_step_end


def observed_end(self: Any, run_context: Any) -> None:
    """Hash each executed observation and promptly remove unneeded full arrays."""
    ORIGINAL_END(self, run_context)
    step = int(run_context.original_args().cur_step_num) + self.state.get('offset', 0)
    records = []
    for module in CONTEXT['modules']:
        for kind in module.get('kinds', ('x', 'y_fp32', 'inverse_rms', 'weight', 'dy', 'dx')):
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
                                'shape': list(array.shape), 'sha256': next(iter(hashes)), 'retained': step == 205,
                                'logical_bytes': path.stat().st_size, 'physical_bytes': path.stat().st_blocks * 512})
                SEEN.add(path)
            del arrays, array
            if step != 205:
                for path in paths:
                    path.unlink()
    with (self.output / 'FINAL_NORM_BOUNDARIES.jsonl').open('a') as stream:
        stream.write(json.dumps({'step': step, 'records': records}) + '\n')


capture_mf_muon.attach_taps = attach
capture_mf_muon.ObserveMetrics.on_train_step_end = observed_end
