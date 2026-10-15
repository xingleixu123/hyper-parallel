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

"""Isolate archived FP32 scalar evaluation from learning-rate folding itself."""
import hashlib
import json
from pathlib import Path

from capture_hp_native import Observer
from hyper_parallel.core.optimizer import muon as muon_module
import muon_lr_fp32_scale


def install():
    original = Observer.on_train_begin

    def begin(self, state, **kwargs):
        original(self, state, **kwargs)
        if muon_module.compute_muon_slice_scale.__name__ != 'archive_scale':
            raise ValueError('Expected unchanged archive-scale control')
        muon_module.compute_muon_slice_scale = muon_lr_fp32_scale.fp32_archive_scale
        report = {'single_factor': 'Scalar sqrt(dimension), RMS product and LR product each evaluated in FP32',
                  'teacher_values_used': False, 'learning_rate_folding_unchanged': True,
                  'source_sha256': hashlib.sha256(Path(muon_lr_fp32_scale.__file__).read_bytes()).hexdigest()}
        (self.output / 'MUON_SCALAR_PRECISION.json').write_text(json.dumps(report, indent=2))
        print('[P12-MUON-SCALAR-FP32]', json.dumps(report), flush=True)

    Observer.on_train_begin = begin
