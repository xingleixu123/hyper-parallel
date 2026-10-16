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

"""Bounded step counters around the same FP32 dgrad used by the three-step arms."""

from __future__ import annotations

from collections import Counter

# This experiment uses the native Torch implementation, as required by the current code-style rule.
import torch  # pylint: disable=forbidden-backend-import
from torch.nn import functional  # pylint: disable=forbidden-backend-import

from linear_dgrad_trace_experiment import NATIVE_LINEAR, TracedLinearInputGradient


class LongLinearDgrad:
    """Retain the reviewed arithmetic and count actual derivative branches per step."""

    def __init__(self) -> None:
        """Install one owned patch with empty bounded scalar counters."""
        self.original = functional.linear
        if self.original is not NATIVE_LINEAR:
            raise RuntimeError("An unrelated functional Linear patch is already installed")
        self.counts = Counter()
        self.shapes = Counter()
        self.replacement = self._linear
        functional.linear = self.replacement

    def _linear(self, inputs: torch.Tensor, weight: torch.Tensor,
                bias: torch.Tensor | None = None) -> torch.Tensor:
        local_types = (torch.Tensor, torch.nn.Parameter)
        eligible = (type(inputs) in local_types and type(weight) in local_types
                    and (bias is None or type(bias) in local_types)
                    and inputs.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16)
        if not eligible:
            self.counts['bypassed_forward'] += 1
            return self.original(inputs, weight, bias)
        if inputs.device.type == 'npu' and torch.npu.matmul.allow_hf32:
            raise ValueError("The matched FP32 dgrad recipe requires NPU HF32 disabled")
        signature = (tuple(inputs.shape), tuple(weight.shape), bias is not None)
        self.shapes[signature] += 1
        if len(self.shapes) > 128:
            raise RuntimeError("The step exceeded its bounded Linear shape inventory")
        self.counts['eligible_forward'] += 1

        def observe(event: str, metadata: dict) -> None:
            """Count only derivative branches reported after their actual calculation."""
            if event != 'derivative_calculated':
                raise RuntimeError("Unexpected derivative observation event")
            self.counts['actual_backward'] += 1
            for name, calculated in zip(('fp32_dgrad', 'native_wgrad', 'native_bias_grad'),
                                        metadata['computed_gradients'], strict=True):
                self.counts[name] += int(calculated)

        return TracedLinearInputGradient.apply(inputs, weight, bias, observe)

    def finish_step(self) -> dict:
        """Return small scalar counters and release the completed step's metadata."""
        result = {'counts': dict(self.counts), 'shapes': [
            {'input_shape': list(key[0]), 'weight_shape': list(key[1]), 'bias': key[2], 'calls': count}
            for key, count in sorted(self.shapes.items())]}
        self.counts.clear()
        self.shapes.clear()
        return result

    def close(self) -> None:
        """Restore only our own patch; never replace another owner's later modification."""
        if functional.linear is not self.replacement:
            raise RuntimeError("Functional Linear patch ownership was lost")
        functional.linear = self.original
