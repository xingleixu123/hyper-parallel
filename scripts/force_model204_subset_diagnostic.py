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

"""Diagnostic-only model intervention after step204; never an acceptance trajectory."""
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import torch

from capture_hp_native import Observer


def install():
    factor = os.environ['P12_MODEL204_FACTOR']
    counts = {'head': 1, 'embedding': 1, 'norms': 3, 'remaining': 5}
    if factor not in counts:
        raise ValueError('Unknown model204 diagnostic subset')
    original_begin = Observer.on_train_begin
    original_end = Observer.on_step_end

    def begin(self, state, **kwargs):
        marker = {'diagnostic_only': True, 'eligible_for_acceptance': False,
                  'intervention': 'After real HP step204, replace selected model tensors with converted MF204; optimizer untouched',
                  'factor': factor,
                  'purpose': 'Determine whether remaining step205 forward/backward discrepancy is caused by post204 parameter differences'}
        (self.output / 'DIAGNOSTIC_ONLY.json').write_text(json.dumps(marker, indent=2))
        original_begin(self, state, **kwargs)

    def end(self, state, **kwargs):
        original_end(self, state, **kwargs)
        if state.global_step in (201, 202, 203, 205, 206):
            return
        if state.global_step != 204:
            raise ValueError('Diagnostic supports only201->206')
        source = Path('/state/jt-recipe-20261014/mf-grad204-state-r2/step-000204/model')
        entries = json.loads((source / 'manifest.json').read_text())
        model = {row['parameter_name']: np.load(source / row['file'], mmap_mode='r') for row in entries}
        path = '/alignment/phase12/hyper-parallel/scripts/checkpoint_conversion/convert_mf_ckpt_to_hp.py'
        spec = importlib.util.spec_from_file_location('teacher204_conversion', path)
        converter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(converter)
        expected = converter.convert_model(model)
        actual = self.trainer.model.state_dict()
        records = []
        with torch.no_grad():
            for name, value in expected.items():
                norm_names = ('model.layers.0.input_layernorm.weight', 'mtp.layers.0.enorm.weight',
                              'mtp.layers.0.transformer_layer.input_layernorm.weight')
                selected = ((factor in ('head', 'remaining') and name == 'lm_head.weight') or
                            (factor in ('embedding', 'remaining') and name == 'model.embed_tokens.weight') or
                            (factor in ('norms', 'remaining') and name in norm_names))
                if not selected:
                    continue
                tensor = actual[name]
                tensor = tensor.to_local() if hasattr(tensor, 'to_local') else tensor
                if tensor.dtype != torch.float32 or tuple(tensor.shape) != value.shape or not tensor.is_contiguous():
                    raise ValueError('Unsupported parameter: ' + name)
                destination = tensor.view(-1)
                source_flat = value.reshape(-1)
                for start in range(0, value.size, 4_000_000):
                    incoming = torch.from_numpy(source_flat[start:start+4_000_000].copy()).to(tensor.device)
                    target = destination[start:start+4_000_000]
                    target.copy_(incoming)
                    if not torch.equal(target, incoming):
                        raise ValueError('Intervention copy failed: ' + name)
                records.append({'name': name, 'shape': list(value.shape), 'all_elements_copied_and_verified': True})
        if len(records) != counts[factor]:
            raise ValueError('Unexpected subset parameter count')
        (self.output / 'MODEL204_INTERVENTION.json').write_text(json.dumps({'complete': True, 'records': records,
            'diagnostic_only': True, 'factor': factor, 'step': 204, 'optimizer_untouched': True}, indent=2))
        print('[P12-DIAGNOSTIC-MODEL204] subset', factor, 'copied and verified', len(records), 'not eligible for acceptance', flush=True)

    Observer.on_train_begin = begin
    Observer.on_step_end = end
