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

"""Observe live preclip inputs in MF layouts without consuming reference values."""
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

import capture_hp_native
import hyper_parallel.trainer.base as trainer_base
import native_clip_reduction_factor
from frontier212_array_io import describe
from observe_hp_preclip_norm_adjacent import mf_gradient

CONTEXT: dict[str, Any] = {}


def install() -> None:
    """Wrap native clipping only at218/219 and preserve all training values."""
    original_begin = capture_hp_native.Observer.on_train_begin
    original_install = native_clip_reduction_factor.install

    def begin(self: Any, state: Any, **kwargs: Any) -> None:
        original_begin(self, state, **kwargs)
        root = Path('/state/jt-recipe-20261014') / self.output.name
        root.mkdir(exist_ok=False)
        order = json.loads(Path('/alignment/phase12/configs/mf-gradient-layout-order.json').read_text())
        if len(order) != 46:
            raise ValueError('Expected all46 MF layouts')
        CONTEXT.update(step=int(state.global_step), output=self.output, root=root, order=order)

    def clip_install() -> None:
        original_install()
        original_clip = trainer_base.clip_grad_norm_

        @torch.no_grad()
        def clip(model: Any, *args: Any, **kwargs: Any) -> Any:
            CONTEXT['step'] += 1
            step = CONTEXT['step']
            if step not in (218, 219):
                return original_clip(model, *args, **kwargs)
            parameters, records = dict(model.named_parameters()), []
            for index, row in enumerate(CONTEXT['order']):
                tensor = mf_gradient(parameters, row['name'])
                if list(tensor.shape) != row['shape'] or tensor.dtype != torch.float32:
                    raise ValueError('Unexpected native FP32 gradient layout')
                array = np.ascontiguousarray(tensor.detach().cpu().numpy())
                record = {'name': row['name'], 'retained': False, **describe(array)}
                records.append(record)
                del tensor, array
            norm = original_clip(model, *args, **kwargs)
            report = {'step': step, 'complete': True, 'arrays': 46, 'records': records,
                      'norm': float(norm.cpu()), 'readonly': True, 'teacher_values_used': False,
                      'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      'helper_sha256': hashlib.sha256((Path(__file__).parent / 'frontier212_array_io.py').read_bytes()).hexdigest()}
            (CONTEXT['output'] / ('PRECLIP-' + str(step) + '.json')).write_text(json.dumps(report, indent=2))
            print('[P12-PRECLIP219]', step, '46 live gradients observed', flush=True)
            return norm

        trainer_base.clip_grad_norm_ = clip

    capture_hp_native.Observer.on_train_begin = begin
    native_clip_reduction_factor.install = clip_install
