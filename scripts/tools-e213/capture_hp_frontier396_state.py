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

"""Observe395/396 selected shared-expert state without changing the passed training factors."""
import runpy

import hp_frontier396_state_probe
import hp_tail396_model_hash
import hp_weighted212_model_hash


if __name__ == '__main__':
    hp_frontier396_state_probe.install()
    hp_weighted212_model_hash.install = hp_tail396_model_hash.install
    runpy.run_path('/alignment/phase12/tools-e213/capture_hp_expert_norm328.py', run_name='__main__')
