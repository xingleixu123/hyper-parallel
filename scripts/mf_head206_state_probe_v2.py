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

"""Read the native graph archive output-head after the206 optimizer update."""
import capture_mf_muon
from head206_observation_v2 import save_head


def install():
    original = capture_mf_muon.ObserveMetrics.on_train_step_end

    def end(self, run_context):
        original(self, run_context)
        step = int(run_context.original_args().cur_step_num) + self.state.get('offset', 0)
        if step != 206:
            return
        matches = [parameter for _, parameter in self.state['network'].parameters_and_names()
                   if parameter.name == 'output_layer.weight']
        if len(matches) != 1:
            raise ValueError('Expected one native output-head parameter')
        save_head(self.output, matches[0].asnumpy(), __file__)

    capture_mf_muon.ObserveMetrics.on_train_step_end = end
