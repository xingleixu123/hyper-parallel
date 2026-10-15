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

"""Isolate archived MF's last-piece scale on the measured Q-up split layout."""
import math
from typing import Any

import torch

from hyper_parallel.core.optimizer import muon as muon_module
from muon_layout_probe import build_optimizer as layout_builder

NATIVE_SCALE = muon_module.compute_muon_slice_scale
REPORTED = False


def archive_scale(value, matched_adamw_rms, zero_rms_scale_mode="zero"):
    """Use the last 8192-row Q/RoPE piece to scale the first 16384-row piece."""
    global REPORTED
    if tuple(value.shape) == (16384, 1536) and matched_adamw_rms:
        scale = math.sqrt(8192) * matched_adamw_rms
        if not REPORTED:
            print("[P12-LEGACY-SCALE]", {"shape": list(value.shape), "scale": scale,
                  "native_scale": NATIVE_SCALE(value, matched_adamw_rms, zero_rms_scale_mode)}, flush=True)
            REPORTED = True
        return scale
    return NATIVE_SCALE(value, matched_adamw_rms, zero_rms_scale_mode)


def build_optimizer(*, model: torch.nn.Module, **kwargs: Any) -> Any:
    """Change only first-piece Q-up scaling relative to muon-q-up's native scaling."""
    if model.config.num_attention_heads != 128 or model.config.q_lora_rank != 1536:
        raise ValueError("Legacy scale experiment is restricted to the observed source layout")
    optimizer = layout_builder(model=model, layout_factor="q-up", **kwargs)
    muon_module.compute_muon_slice_scale = archive_scale
    return optimizer
