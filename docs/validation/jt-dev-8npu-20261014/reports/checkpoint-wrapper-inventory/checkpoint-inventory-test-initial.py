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
"""CPU regressions for activation-checkpoint inventory after FSDP composition."""

import subprocess
import sys
import unittest
from unittest import mock

from torch import nn

from hyper_parallel.core.activation_memory.api import checkpoint_wrapper
from hyper_parallel.core.fully_shard import api as fully_shard_api
from hyper_parallel.tools.model_integration.integration_validator import collect_activation_checkpoint_wrappers


class TestActivationCheckpointInventory(unittest.TestCase):
    """Inspect actual wrappers without initializing a process group or device."""

    def test_fsdp_composition_preserves_all_regions(self) -> None:
        """Count trunk wrappers extended by FSDP alongside an outer MTP wrapper."""
        model = nn.Module()
        model.layers = nn.ModuleList([checkpoint_wrapper(nn.Linear(3, 3)) for _ in range(2)])
        inner = nn.Linear(3, 3)
        model.mtp = checkpoint_wrapper(inner)
        # Use the real class extension, without constructing HSDP state or groups.
        with mock.patch.dict(fully_shard_api.origin_class_to_extend_class, clear=True):
            for layer in model.layers:
                fully_shard_api._extend_module_with_hsdp_interface(layer)
            fully_shard_api._extend_module_with_hsdp_interface(inner)
            actual = collect_activation_checkpoint_wrappers(model)

        expected = {
            "count": 3,
            "wrappers": [
                {"fqn": "layers.0", "wrapped_type": "torch.nn.modules.linear.Linear"},
                {"fqn": "layers.1", "wrapped_type": "torch.nn.modules.linear.Linear"},
                {"fqn": "mtp", "wrapped_type": f"{type(inner).__module__}.{type(inner).__qualname__}"},
            ],
        }
        self.assertEqual(actual, expected)

    def test_root_wrapper_before_and_after_fsdp_class_extension(self) -> None:
        """Retain the root FQN and wrapped type for both supported compositions."""
        for extend in (False, True):
            with self.subTest(extend=extend):
                model = checkpoint_wrapper(nn.Linear(3, 3))
                with mock.patch.dict(fully_shard_api.origin_class_to_extend_class, clear=True):
                    if extend:
                        fully_shard_api._extend_module_with_hsdp_interface(model)
                    actual = collect_activation_checkpoint_wrappers(model)
                self.assertEqual(actual, {
                    "count": 1,
                    "wrappers": [{"fqn": "<root>", "wrapped_type": "torch.nn.modules.linear.Linear"}],
                })

    def test_plain_and_same_named_modules_are_not_checkpoint_regions(self) -> None:
        """A same-named unrelated class is not an HP activation wrapper."""
        unrelated = type("CheckpointWrapper", (nn.Module,), {"__module__": __name__})()
        unrelated._wrapped_module = nn.Linear(3, 3)
        model = nn.Sequential(nn.Linear(3, 3), unrelated)
        self.assertEqual(collect_activation_checkpoint_wrappers(model), {"count": 0, "wrappers": []})

    def test_inventory_import_remains_backend_free(self) -> None:
        """Keep reporting usable without importing Torch in a fresh process."""
        code = (
            "import sys; "
            "from hyper_parallel.tools.model_integration.integration_validator "
            "import collect_activation_checkpoint_wrappers; "
            "raise SystemExit(int(any(name in sys.modules for name in ('torch', 'torch_npu'))))"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False, timeout=30)
        self.assertEqual(result.returncode, 0, f"returncode={result.returncode}, expected=0, stderr={result.stderr}")
