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

"""Hash the selected live expert weight and momentum over the narrowed307–328 interval."""
import hashlib
import json
from pathlib import Path

import capture_hp_native
from model_hash_stream import model_record

TARGET = 'model.layers.1.mlp.experts.down_proj'


def install() -> None:
    """Observe only; preserve every update and write no raw arrays."""
    original = capture_hp_native.Observer.on_step_end

    def end(self, state, **kwargs):
        original(self, state, **kwargs)
        step = int(state.global_step)
        if step not in range(307, 329):
            return
        parameter = dict(self.trainer.model.named_parameters())[TARGET]
        states = [leaf.state[parameter] for leaf in capture_hp_native.leaves(self.trainer.optimizer)
                  if parameter in leaf.state]
        if len(states) != 1:
            raise ValueError('Expected exactly one optimizer state')
        records = []
        for kind, tensor in [('weight', parameter), ('muon_m', states[0]['momentum_buffer'])]:
            value = tensor.to_local() if hasattr(tensor, 'to_local') else tensor
            array = value.detach().float().cpu().numpy()
            if tuple(array.shape) != (8, 5120, 1536):
                raise ValueError('Unexpected expert-down shape')
            records.append({'kind': kind, 'all': model_record(TARGET, array),
                            'experts': [model_record(str(index), row) for index, row in enumerate(array)]})
        with (self.output / 'EXPERT_DOWN_TIME.jsonl').open('a') as stream:
            stream.write(json.dumps({'step': step, 'records': records}) + '\n')
        if step == 307:
            metadata = {'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        'helper_sha256': hashlib.sha256((Path(__file__).parent / 'model_hash_stream.py').read_bytes()).hexdigest(),
                        'target': TARGET, 'raw_arrays_written': False, 'teacher_values_used': False,
                        'observation_steps': list(range(307, 329))}
            (self.output / 'EXPERT_DOWN_TIME_OBSERVER.json').write_text(json.dumps(metadata, indent=2))
        print('[P12-EXPERT-DOWN-TIME]', step, flush=True)

    capture_hp_native.Observer.on_step_end = end
