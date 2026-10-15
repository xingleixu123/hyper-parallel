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

"""Observe native dense219 NS and all model hashes without altering the archive source."""
import os
import capture_mf_muon
import mf_frontier219_ns_tap
import mf_model_hash_stream

if __name__ == '__main__':
    if os.environ.get('MS_ROLE') != 'MS_SCHED':
        mf_frontier219_ns_tap.install()
        mf_model_hash_stream.install()
    capture_mf_muon.main()
