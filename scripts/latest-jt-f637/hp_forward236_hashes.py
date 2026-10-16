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

"""Observe full forward boundaries only at the first independently differing step."""
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

import capture_hp_native
from mindspore272_stat_hash import native_dump_sha1


def install() -> None:
    """Attach transparent hooks before236 without changing forward or backward values."""
    original = capture_hp_native.Observer.on_micro_step_begin
    attached = False

    def begin(self: Any, state: Any, micro_batch: Any, **kwargs: Any) -> None:
        nonlocal attached
        original(self, state, micro_batch, **kwargs)
        if int(state.global_step) != 235 or attached:
            return
        attached = True
        records, modules, seen = [], [], set()

        def save(name: str, kind: str, value: Any) -> None:
            if isinstance(value, (tuple, list)):
                value = value[0]
            if not isinstance(value, torch.Tensor):
                raise TypeError('Expected tensor boundary: ' + name)
            if hasattr(value, 'to_local'):
                value = value.to_local()
            array = np.ascontiguousarray(value.detach().float().cpu().numpy())
            filename = name.replace('.', '_') + '_' + kind + '.npy'
            native = value.detach().contiguous().cpu()
            native_array = native.view(torch.uint16).numpy() if native.dtype == torch.bfloat16 else native.numpy()
            records.append({'name': name, 'kind': kind, 'file': filename, 'shape': list(array.shape),
                            'sha256': hashlib.sha256(array.tobytes()).hexdigest(), 'dtype': str(value.dtype),
                            'fp32_native_dump_sha1': native_dump_sha1(array),
                            'native_dump_sha1': native_dump_sha1(native_array)})
            (self.output / 'FORWARD236.json').write_text(json.dumps({
                'step': 236, 'records': records, 'readonly': True, 'raw_arrays_written': False,
                'hash_helper_sha256': hashlib.sha256(Path(__file__).with_name('mindspore272_stat_hash.py').read_bytes()).hexdigest(),
                'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2))

        def hook(name: str):
            def observe(module: Any, inputs: Any, result: Any) -> None:
                del module
                if name in seen:
                    return
                seen.add(name)
                save(name, 'input', inputs[0])
                save(name, 'output', result)
            return observe

        for name, module in self.trainer.model.named_modules():
            chosen = name and ('norm' in name or name.endswith('.self_attn.o_proj') or name in (
                'model.layers.1.mlp', 'mtp.layers.0.transformer_layer.mlp'))
            if chosen:
                module.register_forward_hook(hook(name))
                modules.append({'name': name, 'class': type(module).__name__})
        if len(modules) != 21:
            raise ValueError(('Unexpected boundary coverage', modules))
        (self.output / 'FORWARD236_MODULES.json').write_text(json.dumps(modules, indent=2))

    capture_hp_native.Observer.on_micro_step_begin = begin
