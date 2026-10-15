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

"""Run independent live weighted reductions for complete resumed windows."""
import os
import runpy

import adamw_archive_scope_factor
import hp_packed217_semantics_factor
import hp_weighted_dvm_factor
import hp_head_gemm_legacy_nz_factor
import hp_shared_up_fp32_factor
import muon_q_adjacent_factor
import router_ns_dvm_factor
import router_ns_npu_sqrt_factor

if __name__ == '__main__':
    mode = os.environ['P12_FRONTIER208_NORM_FACTOR']
    if mode not in ('q', 'router', 'q_router'):
        raise ValueError('Unknown independent factor')
    hp_shared_up_fp32_factor.install()
    hp_head_gemm_legacy_nz_factor.install()
    if mode in ('router', 'q_router'):
        router_ns_npu_sqrt_factor.install()
    else:
        router_ns_dvm_factor.install()
    if mode in ('q', 'q_router'):
        muon_q_adjacent_factor.install()
    hp_weighted_dvm_factor.install()
    hp_packed217_semantics_factor.install()
    adamw_archive_scope_factor.install()
    runpy.run_path('/alignment/phase12/tools-f637/capture_hp_routercontig_denseadj_full.py', run_name='__main__')
