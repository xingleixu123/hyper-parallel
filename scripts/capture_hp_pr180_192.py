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

"""Guard every new HP run against accidental reuse of the CANN 8.5 runtime."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

import torch
import torch_npu
import hyper_parallel
import capture_hp_native

ORIGINAL_BEGIN = capture_hp_native.Observer.on_train_begin


def observe_begin(self, state, **kwargs):
    """Verify the actual runtime before recording the restored-state audit."""
    version = Path("/usr/local/Ascend/cann/opp/version.info").read_text()
    if "Version=9.2." not in version:
        raise ValueError("HP must use CANN 9.2.X")
    if "/alignment/phase12/hp-jt-pr180-192/" not in hyper_parallel.__file__:
        raise ValueError("HP import does not point to the current JT checkout")
    paths = sorted({line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                    if "/" in line and any(part in line for part in ("Ascend", "libascendcl", "libhccl"))})
    if any("runtime-cann85" in path or "8.5.0" in path for path in paths):
        raise ValueError("An old CANN library is loaded")
    report = {"torch": torch.__version__, "torch_npu": torch_npu.__version__,
              "torch_path": torch.__file__, "torch_npu_path": torch_npu.__file__,
              "hyper_parallel_path": hyper_parallel.__file__, "cann_version_info": version,
              "loaded_cann_libraries": paths, "packages": {
                  name: importlib.metadata.version(name) for name in ("transformers", "datasets", "numpy")}}
    (self.output / "CANN_RUNTIME.json").write_text(json.dumps(report, indent=2))
    expected = "1" if self.trainer.config.training.close_matmul_k_shift else "0"
    if os.environ.get("CLOSE_MATMUL_K_SHIFT") != expected:
        raise ValueError("YAML matmul policy was not propagated")
    (self.output / "ARITHMETIC_MODE.json").write_text(json.dumps({
        "CLOSE_MATMUL_K_SHIFT": expected, "configured_through_trainer": True,
        "seeding_monkeypatch_used": False, "teacher_values_used": False}))
    source = Path('/alignment/phase12/hp-jt-pr180-192')
    manifest = json.loads(Path('/alignment/phase12/evidence/pr180-192-source-manifest.json').read_text())
    for record in manifest['files']:
        if hashlib.sha256((source / record['path']).read_bytes()).hexdigest() != record['sha256']:
            raise ValueError('Integrated PR source differs: ' + record['path'])
    (self.output / 'SOURCE_REVISION.json').write_text(json.dumps(manifest, indent=2))
    ORIGINAL_BEGIN(self, state, **kwargs)


capture_hp_native.Observer.on_train_begin = observe_begin

if __name__ == "__main__":
    capture_hp_native.main()
