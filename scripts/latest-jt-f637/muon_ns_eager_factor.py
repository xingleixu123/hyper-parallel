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

"""One-factor replay of the 2D NS dispatch using the existing arithmetic."""
import functools
import os
import torch_npu
import muon_jit_probe


def install() -> None:
    """Change only NS dispatch, retaining the measured algebra and cast boundaries."""
    original = muon_jit_probe.graph_jit_ns
    scope = os.environ.get('P12_NS_EAGER_SCOPE', '2d')
    if scope not in ('2d', 'all'):
        raise ValueError('Invalid NS dispatch scope')

    @functools.wraps(original)
    def eager(value, steps, normalized=None):
        if scope == '2d' and value.ndim != 2:
            return original(value, steps, normalized)
        torch_npu.npu.set_compile_mode(jit_compile=False)
        try:
            return original(value, steps, normalized)
        finally:
            # The containing GraphJitMuon method enters this function with JIT enabled.
            torch_npu.npu.set_compile_mode(jit_compile=True)

    muon_jit_probe.graph_jit_ns = eager
    print('[P12-NS-EAGER] active target=muon_jit_probe.graph_jit_ns scope='+scope, flush=True)
