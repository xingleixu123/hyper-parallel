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

"""Run independent training with only the Q-up normalization factor added."""
import runpy

import hp_ns_epsilon229_factor
import hp_q_up258_factor

ORIGINAL_INSTALL = hp_ns_epsilon229_factor.install


def install_with_q_up() -> dict:
    """Preserve the existing epsilon factor before adding the measured reduction."""
    policy = ORIGINAL_INSTALL()
    hp_q_up258_factor.install()
    return policy


if __name__ == '__main__':
    hp_ns_epsilon229_factor.install = install_with_q_up
    runpy.run_path('/alignment/phase12/tools-f637/capture_hp_preclip330.py', run_name='__main__')
