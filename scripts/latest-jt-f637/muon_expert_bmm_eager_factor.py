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

"""Isolate expert Gram transpose storage within the existing compatibility path."""
import functools

import torch
import torch_npu

import muon_jit_probe


def install() -> None:
    """Remove only the 3D Gram RHS copy; preserve the prior scalar/cast arithmetic."""
    original = muon_jit_probe.graph_jit_ns

    @functools.wraps(original)
    def expert_view(value, steps, normalized=None):
        if value.ndim != 3:
            return original(value, steps, normalized)
        previous = torch_npu.npu.is_jit_compile_false()
        torch_npu.npu.set_compile_mode(jit_compile=False)
        try:
            transposed = value.shape[-2] > value.shape[-1]
            current = value.mT if transposed else value
            current = ((current / (current.norm(dim=(-2, -1), keepdim=True) + 1e-7)).bfloat16()
                       if normalized is None else normalized)
            for _ in range(steps):
                gram = torch.bmm(current, current.mT)
                square = gram @ gram
                polynomial = (gram.float() * -4.78125 + square.float() * 2.03125).bfloat16()
                product = polynomial @ current
                current = (current.float() * 3.4375 + product.float()).bfloat16()
            return (current.mT if transposed else current).float()
        finally:
            torch_npu.npu.set_compile_mode(jit_compile=not previous)

    muon_jit_probe.graph_jit_ns = expert_view
    print('[P12-EXPERT-BMM-EAGER] 3D explicit bmm Gram with RHS view; square/product unchanged', flush=True)
