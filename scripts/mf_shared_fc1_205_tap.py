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

"""Observe main shared-expert FC1 actual205 operands and native derivatives."""
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import mindspore as ms
from mindspore import ops
from mindformers.parallel_core.training_graph.tensor_parallel.layers import SequenceParallelLinear
import capture_mf_muon

CONTEXT={}
SEEN=set()


def gradient_sink(path):
    def sink(gradient):
        ops.TensorDump()(path,gradient.astype(ms.float32))
        return gradient
    return sink


class ObservedShared(SequenceParallelLinear):
    def construct(self,input_,weight=None):
        actual_weight=self.weight if weight is None else weight
        input_=self.probe_dx(input_)
        result,bias=super().construct(input_,weight)
        ops.TensorDump()(self.probe_path+'_x',input_.astype(self.compute_dtype).astype(ms.float32))
        ops.TensorDump()(self.probe_path+'_weight',actual_weight.astype(self.compute_dtype).astype(ms.float32))
        ops.TensorDump()(self.probe_path+'_y',result.astype(ms.float32))
        return self.probe_dy(result),bias


def attach(network: Any,directory: Path) -> None:
    directory.mkdir(parents=True,exist_ok=False)
    name='model.decoder.layers.1.mlp.shared_experts.linear_fc1'
    module=dict(network.cells_and_names())[name]
    if type(module) is not SequenceParallelLinear:raise ValueError('Unexpected shared FC1')
    module.__class__=ObservedShared
    module.probe_path=str(directory/'shared_fc1')
    module.probe_dx=ops.InsertGradientOf(gradient_sink(module.probe_path+'_dx'))
    module.probe_dy=ops.InsertGradientOf(gradient_sink(module.probe_path+'_dy'))
    records=[{'name':name,'prefix':'shared_fc1','kinds':['x','weight','y','dy','dx']}]
    CONTEXT.update(directory=directory,modules=records)
    (directory/'manifest.json').write_text(json.dumps({'modules':records,'retained_steps':[205],
        'compute_dtype':str(module.compute_dtype),'transpose_b':module.transpose_b,
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2))


ORIGINAL_END = capture_mf_muon.ObserveMetrics.on_train_step_end


def observed_end(self: Any, run_context: Any) -> None:
    """Hash each executed observation and promptly remove unneeded full arrays."""
    ORIGINAL_END(self, run_context)
    step = int(run_context.original_args().cur_step_num) + self.state.get('offset', 0)
    records = []
    for module in CONTEXT['modules']:
        for kind in module.get('kinds', ('x', 'y', 'dy', 'dx')):
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
    with (self.output / 'SHARED_FC1_BOUNDARIES.jsonl').open('a') as stream:
        stream.write(json.dumps({'step': step, 'records': records}) + '\n')


capture_mf_muon.attach_taps = attach
capture_mf_muon.ObserveMetrics.on_train_step_end = observed_end
