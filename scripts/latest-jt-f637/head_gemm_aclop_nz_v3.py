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

"""Describe actual NZ input/output storage for the installed legacy MatMulV2 compiler."""
import ctypes as ct

import torch
import torch_npu

from head_gemm_aclop_v2 import (POINTER, bind, check, create_attr, create_buffer, create_desc,
                               destroy_attr, destroy_buffer, destroy_desc, execute, set_bool)

set_storage_shape = bind('aclSetTensorStorageShape', [POINTER, ct.c_int, ct.POINTER(ct.c_int64)])
set_storage_format = bind('aclSetTensorStorageFormat', [POINTER, ct.c_int])


def matmul(gradient: torch.Tensor, inputs: torch.Tensor):
    """Compute the full live product in NZ storage with explicit original ND shapes."""
    for tensor in [gradient, inputs]:
        if tensor.ndim != 2 or tensor.dtype != torch.bfloat16 or tensor.storage_offset() != 0:
            raise ValueError('Expected zero-offset2D BF16 input')
        if not tensor.is_contiguous() or torch_npu.get_npu_format(tensor) not in (0, 2):
            raise ValueError('Expected contiguous original ND operands')
    if gradient.shape[0] != inputs.shape[0]:
        raise ValueError('Reduction sizes differ')
    left = torch_npu.npu_format_cast(gradient, 29)
    right = torch_npu.npu_format_cast(inputs, 29)
    output = torch_npu.empty_with_format((gradient.shape[1], inputs.shape[1]),
                                          dtype=torch.bfloat16, device=inputs.device, acl_format=29)
    descriptors, buffers, metadata = [], [], []
    attribute = create_attr()
    if not attribute:
        raise RuntimeError('aclopCreateAttr returned null')
    try:
        for tensor in [left, right, output]:
            if any(size % 16 for size in tensor.shape):
                raise ValueError('This diagnostic requires16-aligned actual matrices')
            storage = [tensor.shape[1] // 16, tensor.shape[0] // 16, 16, 16]
            expected_elements = tensor.shape[0] * tensor.shape[1]
            if torch_npu.get_storage_size(tensor) != expected_elements:
                raise ValueError('Actual NZ physical element count differs from the audited2D layout')
            shape = list(tensor.shape)
            actual_format = torch_npu.get_npu_format(tensor)
            if actual_format != 29:
                raise ValueError('Actual input/output is not NZ')
            dimensions = (ct.c_int64 * len(storage))(*storage)
            origin = (ct.c_int64 * len(shape))(*shape)
            descriptor = create_desc(27, len(shape), origin, 2)
            if not descriptor:
                raise RuntimeError('aclCreateTensorDesc returned null')
            descriptors.append(descriptor)
            check(set_storage_shape(descriptor, len(storage), dimensions), 'aclSetTensorStorageShape')
            check(set_storage_format(descriptor, 29), 'aclSetTensorStorageFormat')
            buffer = create_buffer(tensor.data_ptr(), tensor.untyped_storage().nbytes())
            if not buffer:
                raise RuntimeError('aclCreateDataBuffer returned null')
            buffers.append(buffer)
            metadata.append({'logical_shape': shape, 'storage_shape': storage, 'format': actual_format,
                             'buffer_bytes': tensor.untyped_storage().nbytes()})
        check(set_bool(attribute, b'transpose_x1', 1), 'transpose_x1')
        check(set_bool(attribute, b'transpose_x2', 0), 'transpose_x2')
        torch.npu.synchronize()
        check(execute(b'MatMulV2', 2, (POINTER * 2)(*descriptors[:2]), (POINTER * 2)(*buffers[:2]),
                      1, (POINTER * 1)(descriptors[2]), (POINTER * 1)(buffers[2]), attribute,
                      0, 0, None, torch.npu.current_stream().npu_stream), 'MatMulV2/NZ')
        torch.npu.synchronize()
        result = torch_npu.npu_format_cast(output, 2)
        torch.npu.synchronize()
        return result, metadata
    finally:
        for buffer in buffers:
            check(destroy_buffer(buffer), 'aclDestroyDataBuffer')
        for descriptor in descriptors:
            destroy_desc(descriptor)
        destroy_attr(attribute)
