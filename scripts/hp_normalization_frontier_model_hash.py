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

"""Hash actual candidate model states after native optimizer updates without dumps."""
import hashlib
import json
from pathlib import Path

from capture_hp_native import Observer, numpy_value
from model_hash_stream import model_record


def install() -> None:
    """Observe each complete model state with unchanged independent training values."""
    original = Observer.on_step_end

    def end(self, state, **kwargs):
        original(self, state, **kwargs)
        if int(state.global_step) not in range(201, 227):
            return
        actual = self.trainer.model.state_dict()
        if len(actual) != 48:
            raise ValueError('Expected48 actual model states')
        records = [model_record(name, numpy_value(value)) for name, value in sorted(actual.items())]
        with (self.output / 'MODEL_HASHES.jsonl').open('a') as stream:
            stream.write(json.dumps({'step': int(state.global_step), 'records': records}) + '\n')
        if int(state.global_step) == 201:
            metadata = {'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        'helper_sha256': hashlib.sha256((Path(__file__).parent / 'model_hash_stream.py').read_bytes()).hexdigest(),
                        'timing': 'after native optimizer update; no model arrays written'}
            (self.output / 'MODEL_HASH_OBSERVER.json').write_text(json.dumps(metadata, indent=2))
        print('[P12-MODEL-HASH]', state.global_step, len(records), flush=True)

    Observer.on_step_end = end
