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

"""Hash native post-callback model states without writing any model arrays."""
import hashlib
import importlib.util
import json
from pathlib import Path

from mindspore.train.callback import Callback
from mindformers.trainer.base_trainer import BaseTrainer
import capture_mf_muon
from model_hash_stream import model_record

CONTEXT = {}
CONVERTER = Path('/alignment/phase12/hyper-parallel/scripts/checkpoint_conversion/convert_mf_ckpt_to_hp.py')


class ModelHashCallback(Callback):
    """Observe after the unchanged native router-bias callback."""

    def on_train_step_end(self, run_context):
        step = int(run_context.original_args().cur_step_num) + CONTEXT['state'].get('offset', 0)
        if step not in (395, 396):
            return
        source = {parameter.name: parameter.asnumpy() for _, parameter in
                  CONTEXT['state']['network'].parameters_and_names()}
        converted = CONTEXT['conversion'].convert_model(source)
        if len(converted) != 48:
            raise ValueError('Expected48 converted model states')
        records = [model_record(name, value) for name, value in sorted(converted.items())]
        del converted, source
        report = {'step': step, 'records': records}
        with (CONTEXT['output'] / 'MODEL_HASHES.jsonl').open('a') as stream:
            stream.write(json.dumps(report) + '\n')
        if step == 395:
            metadata = {'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        'converter_sha256': hashlib.sha256(CONVERTER.read_bytes()).hexdigest(),
                        'helper_sha256': hashlib.sha256((Path(__file__).parent / 'model_hash_stream.py').read_bytes()).hexdigest(),
                        'timing': 'after native callbacks, including router bias; no model arrays written'}
            (CONTEXT['output'] / 'MODEL_HASH_OBSERVER.json').write_text(json.dumps(metadata, indent=2))
        print('[P12-MODEL-HASH]', step, len(records), flush=True)


def install():
    """Append a read-only callback before capture_mf_muon freezes the native callback builder."""
    original_end = capture_mf_muon.ObserveMetrics.on_train_step_end
    original_sort = BaseTrainer.sort_callbacks
    spec = importlib.util.spec_from_file_location('stream_model_conversion', CONVERTER)
    conversion = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conversion)
    CONTEXT['conversion'] = conversion

    def end(self, run_context):
        original_end(self, run_context)
        CONTEXT.update(state=self.state, output=self.output)

    def sort(callbacks):
        return [*original_sort(callbacks), ModelHashCallback()]

    capture_mf_muon.ObserveMetrics.on_train_step_end = end
    BaseTrainer.sort_callbacks = staticmethod(sort)
