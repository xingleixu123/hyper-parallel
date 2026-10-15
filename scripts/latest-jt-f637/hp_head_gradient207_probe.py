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

"""Observe only the actual postclip head207 gradient after the declared206 state intervention."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import capture_hp_native
import hyper_parallel.trainer.base as trainer_base
import native_clip_reduction_factor
from head_moments206_observation import digest
from observe_hp_preclip_norm_adjacent import mf_gradient

STATE = Path('/state/jt-recipe-20261014')
CONTEXT = {}


def install() -> None:
    """Wrap the final clipping function without changing the gradient or its update order."""
    original_begin = capture_hp_native.Observer.on_train_begin
    original_install = native_clip_reduction_factor.install

    def begin(self, state, **kwargs):
        original_begin(self, state, **kwargs)
        CONTEXT.update(step=int(state.global_step), output=self.output)

    def clip_install():
        original_install()
        original_clip = trainer_base.clip_grad_norm_

        @torch.no_grad()
        def clip(model, *args, **kwargs):
            norm = original_clip(model, *args, **kwargs)
            CONTEXT['step'] += 1
            if CONTEXT['step'] != 207:
                return norm
            output = CONTEXT['output']
            directory = STATE / output.name
            directory.mkdir(parents=True, exist_ok=True)
            target = directory / 'head-postclip207.npy'
            if target.exists():
                raise FileExistsError(target)
            gradient = mf_gradient(dict(model.named_parameters()), 'output_layer.weight')
            array = capture_hp_native.numpy_value(gradient)
            sha = digest(array)
            np.save(target, array, allow_pickle=False)
            report = {'step': 207, 'name': 'lm_head.weight', 'boundary': 'postclip_before_optimizer',
                      'sha256': sha, 'file': str(target), 'shape': list(array.shape), 'dtype': str(array.dtype),
                      'bytes': target.stat().st_size, 'read_only_observation': True,
                      'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      'helper_sha256': hashlib.sha256(
                          (Path(__file__).parent / 'head_moments206_observation.py').read_bytes()).hexdigest()}
            (output / 'HEAD_GRADIENT207.json').write_text(json.dumps(report, indent=2))
            return norm

        trainer_base.clip_grad_norm_ = clip

    capture_hp_native.Observer.on_train_begin = begin
    native_clip_reduction_factor.install = clip_install
