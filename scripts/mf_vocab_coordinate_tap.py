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

"""Observe native MF vocabulary gradients and state at preselected coordinates only."""
import json
import time
from pathlib import Path
import numpy as np
from mindspore import ops, Tensor
from mindformers.core.optim.muon import Muon
import mf_gradient_tap
import capture_mf_muon

PAIR = {'embedding.word_embeddings.weight':'model.embed_tokens.weight', 'output_layer.weight':'lm_head.weight'}
COORDS = json.loads(Path('/alignment/phase12/evidence/hp-denseadj-vocab204-coordinates-coordinates.json').read_text())
SELECTED = {r['name']:[c['flat_index'] for c in r['coordinates']] for r in COORDS['records']}
SEEN = set()


class ObservedVocabularyMuon(Muon):
    def construct(self, gradients):
        for index, coordinates, path in self.vocabulary_probes:
            sampled = ops.gather(ops.reshape(gradients[index], (-1,)), coordinates, 0)
            ops.TensorDump()(path, sampled)
        return super().construct(gradients)


def attach(optimizer, destination, max_elements=None):
    del max_elements
    if type(optimizer) is not Muon:
        raise TypeError(type(optimizer))
    destination=Path(destination)
    destination.mkdir(parents=True,exist_ok=False)
    probes=[]
    manifest=[]
    for index,parameter in enumerate(optimizer.parameters):
        if parameter.name not in PAIR:
            continue
        name=PAIR[parameter.name]
        indices=SELECTED[name]
        probes.append((index,Tensor(np.asarray(indices,dtype=np.int32)),str(destination/f'{index:04d}')))
        manifest.append({'name':name,'mf_name':parameter.name,'optimizer_index':index,
                         'prefix':f'{index:04d}','indices':indices,'shape':[len(indices)]})
    if len(probes)!=2:
        raise ValueError('Expected two vocabulary parameters')
    optimizer.__class__=ObservedVocabularyMuon
    optimizer.vocabulary_probes=tuple(probes)
    (destination/'manifest.json').write_text(json.dumps(manifest,indent=2))


ORIGINAL_END=capture_mf_muon.ObserveMetrics.on_train_step_end


def observed_end(self, run_context):
    ORIGINAL_END(self,run_context)
    step=int(run_context.original_args().cur_step_num)+self.state.get('offset',0)
    directory=Path('/state/jt-recipe-20261014/mf-vocab-optimizer-coords-r2/gradients')
    optimizer=self.state['optimizer']
    for row in json.loads((directory/'manifest.json').read_text()):
        deadline=time.monotonic()+60
        while True:
            paths=[f for f in directory.glob(row['prefix']+'_*.npy') if f not in SEEN]
            if len(paths)>1:
                raise ValueError('Ambiguous coordinate gradient dump')
            if paths:
                try:
                    value=np.load(paths[0],mmap_mode='r')
                    if list(value.shape)==row['shape'] and paths[0].stat().st_size==value.offset+value.nbytes:
                        break
                except (ValueError,EOFError,OSError):
                    pass
            if time.monotonic()>deadline:
                raise TimeoutError(row['name'])
            time.sleep(.05)
        SEEN.add(paths[0])
        index=row['optimizer_index']
        coordinates=Tensor(np.asarray(row['indices'],dtype=np.int32))
        def sample(value):
            return ops.gather(ops.reshape(value,(-1,)),coordinates,0).asnumpy().tolist()
        record={'step':step,'name':row['name'],'indices':row['indices'],'gradient':value.tolist(),
                'after':{'weight':sample(optimizer.parameters[index]),
                         'exp_avg':sample(optimizer.moments1[index]),
                         'exp_avg_sq':sample(optimizer.moments2[index])}}
        with (self.output/'VOCAB_OPTIMIZER_TRACE.jsonl').open('a') as stream:
            stream.write(json.dumps(record)+'\n')
    print('[P12-VOCAB-COORDS]',step,'native gradient and AdamW state observed',flush=True)


mf_gradient_tap.attach=attach
capture_mf_muon.ObserveMetrics.on_train_step_end=observed_end

