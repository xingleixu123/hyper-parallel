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

"""Run a declared head-moment diagnostic and observe complete206/207 model hashes."""
import os
import runpy
import hp_shared_up_fp32_factor
import hp_head206_intervention
import hp_head_moments206_intervention
import hp_frontier206_model_hash
import adamw_archive_scope_factor

if __name__ == '__main__':
    mode = os.environ['P12_HEAD_MOMENT_DIAG_MODE']
    hp_shared_up_fp32_factor.install()
    if mode != 'moments':
        os.environ['P12_MODEL206_INTERVENTION'] = 'head_router' if mode == 'head_router_moments' else 'head'
        hp_head206_intervention.install()
    hp_head_moments206_intervention.install()
    hp_frontier206_model_hash.install()
    adamw_archive_scope_factor.install()
    runpy.run_path('/alignment/phase12/tools-f637/capture_hp_routercontig_denseadj_full.py', run_name='__main__')
