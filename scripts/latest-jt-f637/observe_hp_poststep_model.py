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

"""Compare the actual complete post-update model, storing metrics rather than another checkpoint."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

from capture_hp_native import Observer, numpy_value


def bf16_bits(value):
    bits = np.ascontiguousarray(value, dtype=np.float32).view(np.uint32)
    return ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def compare(name, actual, reference, before):
    if actual.shape != reference.shape or actual.dtype != reference.dtype or actual.dtype != np.float32:
        raise ValueError((name, actual.shape, actual.dtype, reference.shape, reference.dtype))
    error_sq, update_sq, reference_sq, maximum, unequal, bf16_unequal = 0., 0., 0., 0., 0, 0
    actual_flat, reference_flat, before_flat = actual.reshape(-1), reference.reshape(-1), before.reshape(-1)
    digest = hashlib.sha256()
    for start in range(0, actual.size, 1_000_000):
        observed = actual_flat[start:start+1_000_000]
        ref = reference_flat[start:start+1_000_000]
        delta = observed.astype(np.float64) - ref.astype(np.float64)
        update = ref.astype(np.float64) - before_flat[start:start+1_000_000].astype(np.float64)
        error_sq += float(np.dot(delta, delta))
        update_sq += float(np.dot(update, update))
        reference_sq += float(np.dot(ref.astype(np.float64), ref.astype(np.float64)))
        maximum = max(maximum, float(abs(delta).max()))
        unequal += int(np.count_nonzero(delta))
        bf16_unequal += int(np.count_nonzero(bf16_bits(observed) != bf16_bits(ref)))
        digest.update(observed.tobytes())
    return {'name': name, 'shape': list(actual.shape), 'elements': actual.size,
            'unequal': unequal, 'bf16_unequal': bf16_unequal, 'max': maximum,
            'weight_relative_l2': (error_sq / reference_sq) ** .5 if reference_sq else None,
            'update_relative_l2': (error_sq / update_sq) ** .5 if update_sq else None,
            'actual_sha256': digest.hexdigest()}


def install():
    original = Observer.on_step_end

    def observed_end(self, state, **kwargs):
        original(self, state, **kwargs)
        if state.global_step != 201:
            raise ValueError('Post-update observation is a one-step diagnostic only')
        source = Path('/state/jt-recipe-20261014/mf-probe201-r2/step-000201/model')
        manifest = json.loads((source / 'manifest.json').read_text())
        model = {row['parameter_name']: np.load(source / row['file'], mmap_mode='r') for row in manifest}
        path = '/alignment/phase12/hyper-parallel/scripts/checkpoint_conversion/convert_mf_ckpt_to_hp.py'
        spec = importlib.util.spec_from_file_location('poststep_conversion', path)
        converter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(converter)
        expected = converter.convert_model(model)
        actual = self.trainer.model.state_dict()
        records = []
        output = Path('/alignment/phase12/evidence') / (self.output.name + '-poststep.json')
        with np.load(self.offline / 'model.npz', allow_pickle=False) as before:
            if set(expected) != set(before.files):
                raise ValueError('Converted poststep model keys differ from the restored model')
            for name in sorted(expected):
                value = numpy_value(actual[name])
                row = compare(name, value, expected[name], before[name])
                records.append(row)
                print('[P12-ACTUAL-POSTSTEP]', json.dumps(row), flush=True)
                output.write_text(json.dumps({'complete': False, 'records': records}, indent=2))
        output.write_text(json.dumps({'complete': True, 'step': state.global_step, 'records': records,
                                     'scope': 'Actual full HP model after optimizer and native callbacks; no reference values injected or extra model checkpoint saved.'}, indent=2))

    Observer.on_step_end = observed_end
