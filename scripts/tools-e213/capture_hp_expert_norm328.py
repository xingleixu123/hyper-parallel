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

"""Add the validated expert-down reduction after the existing expert normalization wrapper."""
import os
import runpy

import hp_expert_norm328_factor
import hp_norm219_factor
import muon_expert_bmm_eager_factor

ORIGINAL_NORMALIZATION_INSTALL = hp_norm219_factor.install


def install_normalization() -> None:
    """Preserve existing installation order, then replace only the measured expert shape."""
    ORIGINAL_NORMALIZATION_INSTALL()
    original_expert_install = muon_expert_bmm_eager_factor.install

    def install_expert():
        result = original_expert_install()
        hp_expert_norm328_factor.install()
        return result

    muon_expert_bmm_eager_factor.install = install_expert


if __name__ == '__main__':
    hp_norm219_factor.install = install_normalization
    mode = os.environ.get('P12_EXPERT328_ENTRY', 'train')
    entries = {'train': 'capture_hp_fc1_norm293.py', 'observe330': 'capture_hp_fc1_preclip330.py'}
    if mode not in entries:
        raise ValueError('Unknown experiment entry mode')
    runpy.run_path('/alignment/phase12/tools-e213/' + entries[mode], run_name='__main__')
