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

"""Observe native main MoE205 branch derivatives with native arithmetic unchanged."""
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import mindspore as ms
from mindspore import ops
from mindformers.parallel_core.training_graph.tensor_parallel.layers import SequenceParallelLinear
from mindformers.parallel_core.training_graph.transformer.moe.ffn import FFNGroupedGEMM
from mindformers.parallel_core.training_graph.transformer.moe.moe_layer import MoELayer
from mindformers.parallel_core.training_graph.transformer.moe.router import TopKRouter
import capture_mf_muon

CONTEXT={}
SEEN=set()


def gradient_sink(path):
    def sink(gradient):
        ops.TensorDump()(path,gradient.astype(ms.float32))
        return gradient
    return sink


class ObservedMoE(MoELayer):
    def construct(self,hidden_states,extra_loss=0.,seq_chunk=None):
        hidden_states=self.probe_dx(hidden_states)
        ops.TensorDump()(self.probe_path+'_x',hidden_states.astype(ms.float32))
        result,bias,loss=super().construct(hidden_states,extra_loss,seq_chunk)
        return self.probe_dy(result),bias,loss


class ObservedRouter(TopKRouter):
    def gating(self,inputs):
        weight=self.weight.astype(self.moe_router_dtype)
        inputs=self.probe_dx(inputs.astype(self.moe_router_dtype))
        result=self.linear(inputs,weight)
        ops.TensorDump()(self.probe_path+'_weight',weight.astype(ms.float32))
        return self.probe_dy(result)


class ObservedExperts(FFNGroupedGEMM):
    def construct(self,tokens,probs,routing_map):
        return super().construct(self.probe_dx(tokens),self.probe_dprob(probs),routing_map)


class ObservedShared(SequenceParallelLinear):
    def construct(self,input_,weight=None):
        return super().construct(self.probe_dx(input_),weight)


def attach(network: Any,directory: Path) -> None:
    directory.mkdir(parents=True,exist_ok=False)
    modules=dict(network.cells_and_names())
    prefix='model.decoder.layers.1.mlp'
    targets=[(prefix,MoELayer,ObservedMoE,['x','dy','dx']),
             (prefix+'.router',TopKRouter,ObservedRouter,['weight','dy','dx']),
             (prefix+'.experts',FFNGroupedGEMM,ObservedExperts,['dx','dprob']),
             (prefix+'.shared_experts.linear_fc1',SequenceParallelLinear,ObservedShared,['dx'])]
    records=[]
    for name,original,observed,kinds in targets:
        module=modules[name]
        if type(module) is not original:
            raise ValueError('Unexpected branch class: '+name+':'+repr(type(module)))
        module.__class__=observed
        module.probe_path=str(directory/name.replace('.','_'))
        for kind in ['dx','dy','dprob']:
            if kind in kinds:setattr(module,'probe_'+kind,ops.InsertGradientOf(gradient_sink(module.probe_path+'_'+kind)))
        records.append({'name':name,'prefix':name.replace('.','_'),'kinds':kinds})
    CONTEXT.update(directory=directory,modules=records)
    (directory/'manifest.json').write_text(json.dumps({'modules':records,'retained_steps':[205],
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
    with (self.output / 'MOE_BRANCHES.jsonl').open('a') as stream:
        stream.write(json.dumps({'step': step, 'records': records}) + '\n')


capture_mf_muon.attach_taps = attach
capture_mf_muon.ObserveMetrics.on_train_step_end = observed_end
