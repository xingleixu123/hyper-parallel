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

"""Audit step206 gradient hashes with the independently computed FP32 shared-up factor."""
import runpy
import hp_shared_up_fp32_factor
import hp_gradient206_hash_audit
import adamw_archive_scope_factor

if __name__ == '__main__':
    hp_shared_up_fp32_factor.install()
    hp_gradient206_hash_audit.install()
    adamw_archive_scope_factor.install()
    runpy.run_path('/alignment/phase12/tools-f637/capture_hp_routercontig_denseadj_full.py', run_name='__main__')
