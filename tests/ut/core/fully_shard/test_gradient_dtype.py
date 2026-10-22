# Copyright 2026 Huawei Technologies Co., Ltd
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
import unittest
from types import SimpleNamespace

import torch
from torch import nn

from hyper_parallel.core.fully_shard.hsdp_param import HSDPParamV2


class TestGradientDtype(unittest.TestCase):
    def test_fp32_accumulator_stays_on_bf16_leaf(self):
        if not hasattr(torch.Tensor, "grad_dtype"):
            self.skipTest("Requires public Tensor.grad_dtype")
        parameter = object.__new__(HSDPParamV2)
        parameter.unsharded_param_buffers = [torch.ones(4, dtype=torch.bfloat16)]
        parameter.allgather_comm_ctx = SimpleNamespace(allgather_output=None)
        parameter._orig_size = torch.Size((2, 2))
        parameter._contiguous_orig_stride = (2, 1)
        parameter.source_shard_info = None
        parameter.sharded_param = nn.Parameter(torch.ones(2, 2))
        parameter.reduce_dtype = torch.float32
        parameter.unsharded_accumulated_grad = None
        parameter.init_unsharded_param()
        leaf = parameter.unsharded_param
        self.assertEqual(leaf.grad_dtype, torch.float32)
        leaf.float().sum().backward()
        saved_gradient = leaf.grad
        parameter.to_accumulated_grad_if_needed()
        self.assertIs(leaf.grad, saved_gradient)
        self.assertIsNone(parameter.unsharded_accumulated_grad)
        leaf.float().sum().backward()
        torch.testing.assert_close(leaf.grad, torch.full((2, 2), 2.0))

    def test_legacy_accumulator_keeps_existing_behavior(self):
        parameter = object.__new__(HSDPParamV2)
        parameter._unsharded_param = nn.Parameter(torch.ones(2))
        parameter._unsharded_param.grad = torch.ones(2)
        parameter.unsharded_accumulated_grad = None
        parameter.reduce_dtype = None
        parameter.to_accumulated_grad_if_needed()
        self.assertIsNone(parameter.unsharded_param.grad)
        torch.testing.assert_close(parameter.unsharded_accumulated_grad, torch.ones(2))
