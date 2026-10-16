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

"""Add only the measured shared-expert down reduction after the existing expert factor."""
import os
import runpy

import hp_expert_norm328_factor
import hp_shared_down396_factor

ORIGINAL_EXPERT_INSTALL = hp_expert_norm328_factor.install


def install_expert_and_shared() -> dict:
    """Preserve the existing factor, then apply the same live reduction to matching dense matrices."""
    policy = ORIGINAL_EXPERT_INSTALL()
    hp_shared_down396_factor.install()
    return policy


if __name__ == '__main__':
    hp_expert_norm328_factor.install = install_expert_and_shared
    mode = os.environ.get('P12_EXPERT328_ENTRY', 'train')
    entries = {'train': 'capture_hp_expert_norm328.py', 'observe397': 'capture_hp_expert_preclip397.py'}
    if mode not in entries:
        raise ValueError('Unknown candidate entry mode')
    runpy.run_path('/alignment/phase12/tools-e213/' + entries[mode], run_name='__main__')
