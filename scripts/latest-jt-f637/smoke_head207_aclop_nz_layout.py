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

"""Validate physical layout and both ACL descriptor interfaces before actual-data replay."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch_npu

import head_gemm_aclop_nz_smoke_helper as helper
from hyper_parallel.trainer.runtime.random import enable_full_determinism


def decode(words, shape, nz):
    value = (words.astype(np.uint32) << 16).view(np.float32)
    if nz:
        rows, columns = shape
        return value.reshape(columns // 16, rows // 16, 16, 16).transpose(1, 2, 0, 3).reshape(shape)
    return value.reshape(shape)


def main():
    torch.npu.set_device(0)
    enable_full_determinism(42, close_matmul_k_shift=False)
    torch_npu.npu.set_compile_mode(jit_compile=True)
    torch.npu.config.allow_internal_format = True
    left_cpu = ((torch.arange(32 * 64).reshape(32, 64) % 7) - 3).float()
    right_cpu = ((torch.arange(32 * 96).reshape(32, 96) % 5) - 2).float()
    reference = (left_cpu.T @ right_cpu).numpy()
    rows = []
    for mode in ['modern', 'legacy']:
        helper.DESCRIPTOR_MODE = mode
        result, metadata = helper.matmul(left_cpu.npu().bfloat16(), right_cpu.npu().bfloat16())
        actual = result.float().cpu().numpy()
        checks = []
        for words, ref in zip(helper.RAW, [left_cpu.numpy(), right_cpu.numpy(), reference]):
            checks.append({route: {'different': int(np.count_nonzero(decode(words, ref.shape, nz) != ref)),
                                   'first_32': decode(words, ref.shape, nz).reshape(-1)[:32].tolist()}
                           for route, nz in [('raw_ND', False), ('raw_NZ', True)]})
        row = {'descriptor_mode': mode, 'metadata': metadata, 'storage_checks': checks,
               'converted_different': int(np.count_nonzero(actual != reference)),
               'converted_first32': actual.reshape(-1)[:32].tolist(),
               'reference_first32': reference.reshape(-1)[:32].tolist()}
        rows.append(row)
        print(json.dumps(row), flush=True)
    report = {'complete': True, 'records': rows, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'helper_sha256': hashlib.sha256(Path(helper.__file__).read_bytes()).hexdigest()}
    Path('/alignment/phase12/evidence/head207-gemm-aclop-nz-layout-smoke.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
