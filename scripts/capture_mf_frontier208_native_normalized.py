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

"""Observe original native norm kernels with unchanged archive functions."""
import runpy

import mf_frontier208_state_tap

if __name__ == '__main__':
    mf_frontier208_state_tap.RUN = 'mf-frontier208-native-normalized'
    runpy.run_path('/alignment/phase12/tools/capture_mf_frontier208_state.py', run_name='__main__')
