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

"""Hash archived expert-down state over307–328 without changing native arithmetic."""
import hashlib
import json
from pathlib import Path

import numpy as np
import capture_mf_muon
from model_hash_stream import model_record

TARGET = 'decoder.layers.1.mlp.experts.weight2'
HP_TARGET = 'model.layers.1.mlp.experts.down_proj'


def install() -> None:
    """Observe one selected state after its native update, saving only hashes."""
    original = capture_mf_muon.ObserveMetrics.on_train_step_end

    def end(self, run_context):
        original(self, run_context)
        step = int(run_context.original_args().cur_step_num) + self.state.get('offset', 0)
        if step not in range(307, 329):
            return
        optimizer = self.state['optimizer']
        selected = [(index, parameter) for index, parameter in enumerate(optimizer.parameters)
                    if parameter.name == TARGET]
        if len(selected) != 1:
            raise ValueError('Expected exactly one native target')
        index, parameter = selected[0]
        if not optimizer.use_muon[index]:
            raise ValueError('Expected Muon expert-down state')
        records = []
        for kind, tensor in [('weight', parameter), ('muon_m', optimizer.muon_m[index])]:
            native = tensor.asnumpy()
            if tuple(native.shape) != (12288, 5120):
                raise ValueError('Unexpected native expert-down shape')
            array = np.ascontiguousarray(native.reshape(8, 1536, 5120).transpose(0, 2, 1))
            records.append({'kind': kind, 'all': model_record(HP_TARGET, array),
                            'experts': [model_record(str(index), row) for index, row in enumerate(array)]})
        with (self.output / 'EXPERT_DOWN_TIME.jsonl').open('a') as stream:
            stream.write(json.dumps({'step': step, 'optimizer_step': int(optimizer.global_step.asnumpy().reshape(-1)[0]),
                                    'records': records}) + '\n')
        if step == 307:
            metadata = {'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        'helper_sha256': hashlib.sha256((Path(__file__).parent / 'model_hash_stream.py').read_bytes()).hexdigest(),
                        'target': TARGET, 'raw_arrays_written': False, 'teacher_values_used': False,
                        'layout': 'reshape(8,1536,5120).transpose(0,2,1)',
                        'observation_steps': list(range(307, 329))}
            (self.output / 'EXPERT_DOWN_TIME_OBSERVER.json').write_text(json.dumps(metadata, indent=2))
        print('[P12-EXPERT-DOWN-TIME]', step, flush=True)

    capture_mf_muon.ObserveMetrics.on_train_step_end = end
