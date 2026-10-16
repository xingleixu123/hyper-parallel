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

"""Observe the next native gradient/model boundary after global norm alignment."""
import runpy

import hp_global_sqrt235_factor
import hp_ns_epsilon229_factor
import hp_q_up258_factor
import hp_frontier293_state_probe
import hp_tail293_model_hash
import hp_weighted212_model_hash


if __name__ == '__main__':
    hp_global_sqrt235_factor.install()
    original = hp_ns_epsilon229_factor.install

    def install_with_q_up():
        policy = original()
        hp_q_up258_factor.install()
        return policy

    hp_ns_epsilon229_factor.install = install_with_q_up
    hp_frontier293_state_probe.install()
    hp_weighted212_model_hash.install = hp_tail293_model_hash.install
    runpy.run_path('/alignment/phase12/tools-f637/capture_hp_epsilon229.py', run_name='__main__')
