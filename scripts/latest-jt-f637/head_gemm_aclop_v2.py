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

"""Call installed CANN9.2 MatMulV2 explicitly on Torch-owned contiguous BF16 ND storage."""
import ctypes as ct
from pathlib import Path

import torch
import torch_npu

POINTER = ct.c_void_p
POINTERS = ct.POINTER(POINTER)
ROOT = Path('/usr/local/Ascend/cann-9.2.0-beta.2')
LIBRARY = ct.CDLL(str(ROOT / 'lib64/libacl_op_compiler.so'))


def bind(name, arguments, result=ct.c_int32):
    """Bind the ABI from the installed ACL headers, including void destructors."""
    function = getattr(LIBRARY, name)
    function.argtypes = arguments
    function.restype = result
    return function


create_desc = bind('aclCreateTensorDesc', [ct.c_int, ct.c_int, ct.POINTER(ct.c_int64), ct.c_int], POINTER)
destroy_desc = bind('aclDestroyTensorDesc', [POINTER], None)
create_buffer = bind('aclCreateDataBuffer', [POINTER, ct.c_size_t], POINTER)
destroy_buffer = bind('aclDestroyDataBuffer', [POINTER])
create_attr = bind('aclopCreateAttr', [], POINTER)
destroy_attr = bind('aclopDestroyAttr', [POINTER], None)
set_bool = bind('aclopSetAttrBool', [POINTER, ct.c_char_p, ct.c_uint8])
execute = bind('aclopCompileAndExecuteV2', [ct.c_char_p, ct.c_int, POINTERS, POINTERS, ct.c_int,
                                          POINTERS, POINTERS, POINTER, ct.c_int, ct.c_int,
                                          ct.c_char_p, POINTER])


def check(status, operation):
    """Never accept an unsuccessful launch as numerical output."""
    if status:
        raise RuntimeError(f'{operation}: CANN status {status}')


def matmul(gradient: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """Compute gradient.T@inputs with explicit MatMulV2 transpose attributes."""
    for value in [gradient, inputs]:
        if value.ndim != 2 or value.dtype != torch.bfloat16 or not value.is_contiguous():
            raise ValueError('Expected contiguous2D BF16 operands')
        if value.storage_offset() != 0 or torch_npu.get_npu_format(value) not in (0, 2):
            raise ValueError('Expected zero-offset ND storage')
    if gradient.shape[0] != inputs.shape[0] or gradient.device != inputs.device:
        raise ValueError('Operand reduction dimensions or devices differ')
    torch.npu.synchronize()
    output = torch.empty((gradient.shape[1], inputs.shape[1]), dtype=torch.bfloat16, device=inputs.device)
    descriptors, buffers = [], []
    attribute = create_attr()
    if not attribute:
        raise RuntimeError('aclopCreateAttr returned null')
    try:
        for value in [gradient, inputs, output]:
            dims = (ct.c_int64 * 2)(*value.shape)
            desc = create_desc(27, 2, dims, 2)
            if not desc:
                raise RuntimeError('aclCreateTensorDesc returned null')
            descriptors.append(desc)
            buffer = create_buffer(value.data_ptr(), value.numel() * value.element_size())
            if not buffer:
                raise RuntimeError('aclCreateDataBuffer returned null')
            buffers.append(buffer)
        check(set_bool(attribute, b'transpose_x1', 1), 'transpose_x1')
        check(set_bool(attribute, b'transpose_x2', 0), 'transpose_x2')
        status = execute(b'MatMulV2', 2, (POINTER * 2)(*descriptors[:2]), (POINTER * 2)(*buffers[:2]),
                         1, (POINTER * 1)(descriptors[2]), (POINTER * 1)(buffers[2]),
                         attribute, 0, 0, None, torch.npu.current_stream().npu_stream)
        check(status, 'aclopCompileAndExecuteV2/MatMulV2')
        torch.npu.synchronize()
        return output
    finally:
        for buffer in buffers:
            check(destroy_buffer(buffer), 'aclDestroyDataBuffer')
        for desc in descriptors:
            destroy_desc(desc)
        destroy_attr(attribute)
