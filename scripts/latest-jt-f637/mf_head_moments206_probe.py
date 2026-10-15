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

"""Observe native MF output-head moment slots after the206 update."""
import capture_mf_muon
import head_moments206_observation as observation


def install() -> None:
    """Read actual optimizer-owned tensors without changing the MF graph archive."""
    original = capture_mf_muon.ObserveMetrics.on_train_step_end

    def end(self, run_context):
        original(self, run_context)
        step = int(run_context.original_args().cur_step_num) + self.state.get('offset', 0)
        if step != 206:
            return
        optimizer = self.state['optimizer']
        matches = [(index, parameter) for index, parameter in enumerate(optimizer.parameters)
                   if parameter.name == 'output_layer.weight']
        if len(matches) != 1:
            raise ValueError('Expected one optimizer-owned native head')
        index, parameter = matches[0]
        if optimizer.use_muon[index]:
            raise ValueError('Expected AdamW output head')
        records = []
        for kind, value in [('exp_avg', optimizer.moments1[index]), ('exp_avg_sq', optimizer.moments2[index])]:
            records.append(observation.save(self.output, kind, value.asnumpy()))
        optimizer_step = int(optimizer.global_step.asnumpy().reshape(-1)[0])
        observation.finish(self.output, 'mf', parameter.asnumpy(), records, optimizer_step, __file__,
                           parameter_name=parameter.name, optimizer_type=type(optimizer).__name__)

    capture_mf_muon.ObserveMetrics.on_train_step_end = end
