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

"""Execute an explicit installed ReduceSumD on live contiguous ND tensors."""
import ctypes as ct

import torch
import torch_npu

from head_gemm_aclop_v2 import (POINTER, bind, check, create_attr, create_buffer, create_desc,
                                destroy_attr, destroy_buffer, destroy_desc, execute, set_bool)

SET_INTS = bind('aclopSetAttrListInt', [POINTER, ct.c_char_p, ct.c_int, ct.POINTER(ct.c_int64)])


def reduce_last(value: torch.Tensor) -> torch.Tensor:
    """Reduce the final dimension with matching input/output dtype and keep_dims=True."""
    if value.dtype not in (torch.bfloat16, torch.float32) or not value.is_contiguous():
        raise ValueError('Expected contiguous BF16 or FP32')
    if value.storage_offset() or torch_npu.get_npu_format(value) not in (0, 2):
        raise ValueError('Expected ND storage with zero offset')
    torch.npu.synchronize()
    output = torch.empty((*value.shape[:-1], 1), device=value.device, dtype=value.dtype)
    descriptors, buffers = [], []
    attribute = create_attr()
    if not attribute:
        raise RuntimeError('aclopCreateAttr returned null')
    try:
        for tensor in [value, output]:
            dims = (ct.c_int64 * tensor.ndim)(*tensor.shape)
            desc = create_desc(27 if tensor.dtype == torch.bfloat16 else 0, tensor.ndim, dims, 2)
            if not desc:
                raise RuntimeError('aclCreateTensorDesc returned null')
            descriptors.append(desc)
            buffer = create_buffer(tensor.data_ptr(), tensor.numel() * tensor.element_size())
            if not buffer:
                raise RuntimeError('aclCreateDataBuffer returned null')
            buffers.append(buffer)
        check(SET_INTS(attribute, b'axes', 1, (ct.c_int64 * 1)(value.ndim - 1)), 'axes')
        check(set_bool(attribute, b'keep_dims', 1), 'keep_dims')
        check(execute(b'ReduceSumD', 1, (POINTER * 1)(descriptors[0]), (POINTER * 1)(buffers[0]),
                      1, (POINTER * 1)(descriptors[1]), (POINTER * 1)(buffers[1]), attribute, 0, 0, None,
                      torch.npu.current_stream().npu_stream), 'aclopCompileAndExecuteV2/ReduceSumD')
        torch.npu.synchronize()
        return output
    finally:
        for buffer in buffers:
            check(destroy_buffer(buffer), 'aclDestroyDataBuffer')
        for desc in descriptors:
            destroy_desc(desc)
        destroy_attr(attribute)
