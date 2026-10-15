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

"""Observe selected native MF vocabulary-head operands without changing arithmetic."""
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import mindspore as ms
from mindspore import ops, Tensor
from mindformers.parallel_core.training_graph.tensor_parallel.layers import ColumnParallelLinear

import capture_mf_muon

CONTEXT = {}
SEEN = set()


def gradient_sink(path: str, rows: Tensor, compute_dtype: Any) -> Any:
    """Retain both sides of the native output-cast derivative boundary."""
    def sink(gradient):
        sampled = ops.gather(ops.reshape(gradient, (-1, 152064)), rows, 1)
        ops.TensorDump()(path + '_dy_output', sampled.astype(ms.float32))
        ops.TensorDump()(path + '_dy_matmul', sampled.astype(compute_dtype).astype(ms.float32))
        return gradient
    return sink


class ObservedVocabularyHead(ColumnParallelLinear):
    """Keep the native column-linear computation and observe a few operand columns."""

    def construct(self, input_: Any, weight: Any = None) -> Any:
        result, bias = super().construct(input_, weight)
        sampled = ops.gather(ops.reshape(input_, (-1, 5120)), self.probe_hidden, 1)
        ops.TensorDump()(self.probe_path + '_x', sampled.astype(self.compute_dtype).astype(ms.float32))
        return self.probe_dy(result), bias


def attach(network: Any, directory: Path) -> None:
    """Resolve both shared vocabulary-head call sites from the native model."""
    directory.mkdir(parents=True, exist_ok=False)
    coordinates = json.loads(Path(
        '/alignment/phase12/evidence/hp-denseadj-vocab204-coordinates-coordinates.json').read_text())
    selected = [item['flat_index'] for row in coordinates['records']
                if row['name'] == 'lm_head.weight' for item in row['coordinates']]
    rows = sorted({index // 5120 for index in selected})
    columns = sorted({index % 5120 for index in selected})
    matches = [(name, module) for name, module in network.cells_and_names()
               if name.endswith('.output_layer') and isinstance(module, ColumnParallelLinear)]
    if len(matches) != 2:
        raise ValueError('Expected two vocabulary head call sites: ' + repr([name for name, _ in matches]))
    manifest = []
    for name, module in matches:
        if module.input_size != 5120 or not module.transpose_b or module.has_bias:
            raise ValueError('Unexpected head configuration: ' + name)
        prefix = name.replace('.', '_')
        manifest.append({'name': name, 'prefix': prefix, 'compute_dtype': str(module.compute_dtype),
                         'input_size': module.input_size, 'skip_weight_param_allocation': module.skip_weight_param_allocation})
        module.__class__ = ObservedVocabularyHead
        module.probe_path = str(directory / prefix)
        module.probe_hidden = Tensor(np.asarray(columns, dtype=np.int32))
        module.probe_dy = ops.InsertGradientOf(gradient_sink(
            module.probe_path, Tensor(np.asarray(rows, dtype=np.int32)), module.compute_dtype))
    CONTEXT.update(directory=directory, modules=manifest)
    record = {'modules': manifest, 'vocabulary_rows': rows, 'hidden_columns': columns,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'teacher_values_used': False}
    (directory / 'manifest.json').write_text(json.dumps(record, indent=2))


ORIGINAL_END = capture_mf_muon.ObserveMetrics.on_train_step_end


def observed_end(self: Any, run_context: Any) -> None:
    """Associate asynchronous dump files with the exact training step."""
    ORIGINAL_END(self, run_context)
    step = int(run_context.original_args().cur_step_num) + self.state.get('offset', 0)
    records = []
    for module in CONTEXT['modules']:
        for kind in ('x', 'dy_output', 'dy_matmul'):
            deadline = time.monotonic() + 30
            while True:
                paths = sorted(set(CONTEXT['directory'].glob(module['prefix'] + '_' + kind + '_*.npy')) - SEEN)
                try:
                    arrays = [np.load(path, mmap_mode='r') for path in paths]
                    ready = paths and all(path.stat().st_size == arr.offset + arr.nbytes
                                          for path, arr in zip(paths, arrays))
                except (ValueError, EOFError, OSError):
                    ready = False
                if ready:
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError(module['name'] + ':' + kind)
                time.sleep(.05)
            for path, array in zip(paths, arrays):
                records.append({'name': module['name'], 'kind': kind, 'file': path.name,
                                'shape': list(array.shape), 'sha256': hashlib.sha256(array.tobytes()).hexdigest()})
                SEEN.add(path)
    with (self.output / 'HEAD_OPERANDS.jsonl').open('a') as stream:
        stream.write(json.dumps({'step': step, 'records': records}) + '\n')
    print('[P12-HEAD-OPERANDS]', step, len(records), flush=True)


capture_mf_muon.attach_taps = attach
capture_mf_muon.ObserveMetrics.on_train_step_end = observed_end
