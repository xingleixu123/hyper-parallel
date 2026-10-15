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

"""Read native head207 moment hashes without dumping another pair of large arrays."""
import hashlib
import json
from pathlib import Path

import capture_mf_muon
from head_moments206_observation import digest

ROOT = Path('/alignment/phase12')


def install() -> None:
    """Attach a read-only callback after all native update and bias callbacks."""
    original = capture_mf_muon.ObserveMetrics.on_train_step_end

    def end(self, run_context):
        original(self, run_context)
        step = int(run_context.original_args().cur_step_num) + self.state.get('offset', 0)
        if step != 207:
            return
        optimizer = self.state['optimizer']
        matches = [(index, parameter) for index, parameter in enumerate(optimizer.parameters)
                   if parameter.name == 'output_layer.weight']
        if len(matches) != 1:
            raise ValueError('Head optimizer ownership is not unique')
        index, parameter = matches[0]
        if optimizer.use_muon[index]:
            raise ValueError('Expected AdamW head')
        optimizer_step = int(optimizer.global_step.asnumpy().reshape(-1)[0])
        model_sha = digest(parameter.asnumpy())
        rows = [json.loads(line) for line in (ROOT / 'runs/mf-modelhash226/MODEL_HASHES.jsonl').read_text().splitlines()]
        expected = next(item for row in rows if row['step'] == step for item in row['records']
                        if item['name'] == 'lm_head.weight')
        if optimizer_step != step or model_sha != expected['fp32_sha256']:
            raise ValueError('Native head207 control did not reproduce')
        records = [{'kind': kind, 'sha256': digest(value.asnumpy())}
                   for kind, value in [('exp_avg', optimizer.moments1[index]), ('exp_avg_sq', optimizer.moments2[index])]]
        report = {'step': step, 'optimizer_step': optimizer_step, 'model_fp32_sha256': model_sha,
                  'records': records, 'read_only_observation': True,
                  'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'helper_sha256': hashlib.sha256((Path(__file__).parent / 'head_moments206_observation.py').read_bytes()).hexdigest()}
        (self.output / 'HEAD_MOMENTS207_HASH.json').write_text(json.dumps(report, indent=2))

    capture_mf_muon.ObserveMetrics.on_train_step_end = end
