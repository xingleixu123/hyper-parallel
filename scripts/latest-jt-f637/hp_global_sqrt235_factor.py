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

"""Move only the global clipping square root from CPU to the active NPU."""
import hashlib
import inspect
import json
from pathlib import Path

import capture_hp_native
import native_clip_reduction_factor


def install() -> None:
    """Retain every live reduction and apply the same square-root rule on every step."""
    original = inspect.getsource(native_clip_reduction_factor.install)
    old = 'norm = torch.tensor(np.sqrt(total),dtype=torch.float32,device=gradients[0].device)'
    logging = "'norm':float(np.sqrt(total))"
    if original.count(old) != 1 or original.count(logging) != 1:
        raise ValueError('Unexpected frozen clipping source')
    source = original.replace('def install():', 'def install_global_npu_sqrt235():')
    source = source.replace(old, 'norm = torch.tensor(total,dtype=torch.float32,device=gradients[0].device).sqrt()')
    source = source.replace(logging, "'norm':float(norm.cpu())")
    exec(compile(source, __file__ + ':global-npu-sqrt', 'exec'), native_clip_reduction_factor.__dict__)
    native_clip_reduction_factor.install = native_clip_reduction_factor.install_global_npu_sqrt235
    original_begin = capture_hp_native.Observer.on_train_begin

    def begin(self, state, **kwargs):
        original_begin(self, state, **kwargs)
        report = {
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'original_function_sha256': hashlib.sha256(original.encode()).hexdigest(),
            'candidate_function_sha256': hashlib.sha256(source.encode()).hexdigest(),
            'teacher_values_used': False, 'coordinate_or_step_special_case': False,
            'scope': 'Global clip norm sqrt on NPU every step; same live FP32 square sums, '
                     'per-parameter reductions, total sum, clipping epsilon and multiplier.',
            'classification': 'Archive compatibility ablation; CPU sqrt is not asserted to be less accurate.',
        }
        (self.output / 'GLOBAL_NPU_SQRT235_FACTOR.json').write_text(json.dumps(report, indent=2))

    capture_hp_native.Observer.on_train_begin = begin
