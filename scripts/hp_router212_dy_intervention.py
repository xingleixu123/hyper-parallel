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

"""One-time causal intervention; never eligible for independent alignment acceptance."""
import hashlib
import json
from pathlib import Path
from types import MethodType
from typing import Any

import numpy as np
import torch

from capture_hp_native import Observer
import muon_dense_adjacent_factor
from router_weight_contiguous_factor import RouterLinear

CONTEXT: dict[str, Any] = {'step': 201, 'active': False, 'applied': 0}


def install() -> None:
    """Replace only two measured212 main-router logit derivatives and verify complete hashes."""
    original_forward, original_backward = RouterLinear.forward, RouterLinear.backward
    original_end = Observer.on_step_end
    original_finalize = muon_dense_adjacent_factor.install

    def forward(ctx: Any, values: Any, weight: Any) -> Any:
        ctx.causal_main = CONTEXT['active']
        return original_forward(ctx, values, weight)

    def backward(ctx: Any, gradient: Any) -> Any:
        if not ctx.causal_main or CONTEXT['step'] != 212:
            return original_backward(ctx, gradient)
        metadata = CONTEXT['metadata']
        if gradient.dtype != torch.float32 or list(gradient.shape) != metadata['shape'] or not gradient.is_contiguous():
            raise ValueError('Unexpected actual router derivative representation')
        before = gradient.detach().cpu().numpy()
        digest = hashlib.sha256(before.tobytes()).hexdigest()
        if digest != metadata['hp_sha256']:
            raise ValueError('Causal intervention input differs from the audited independent trajectory')
        patch = CONTEXT['patch']
        indices = torch.from_numpy(patch['indices']).to(device=gradient.device)
        replacement = gradient.clone()
        replacement.view(-1)[indices] = torch.from_numpy(patch['mf']).to(device=gradient.device)
        after = replacement.detach().cpu().numpy()
        changed = np.flatnonzero(before.view(np.uint32).reshape(-1) != after.view(np.uint32).reshape(-1))
        if not np.array_equal(changed, patch['indices']):
            raise ValueError('Intervention changed unexpected derivative coordinates')
        after_digest = hashlib.sha256(after.tobytes()).hexdigest()
        if after_digest != metadata['mf_sha256']:
            raise ValueError('Intervention does not reproduce the complete measured MF derivative')
        CONTEXT['applied'] += 1
        if CONTEXT['applied'] != 1:
            raise ValueError('Causal intervention unexpectedly repeated')
        report = {'step': 212, 'replaced_elements': 2, 'before_sha256': digest,
                  'after_sha256': after_digest, 'complete_mf_derivative_exact': True,
                  'diagnostic_only': True, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        (CONTEXT['output'] / 'ROUTER212_DY_INTERVENTION.json').write_text(json.dumps(report, indent=2))
        return original_backward(ctx, replacement)

    def begin(self: Any, state: Any, **kwargs: Any) -> None:
        metadata = json.loads(Path('/alignment/phase12/evidence/router212-dy-patch.json').read_text())
        path = Path('/state/jt-recipe-20261014/router212-dy-causal-patch/patch.npz')
        if hashlib.sha256(path.read_bytes()).hexdigest() != metadata['patch_file_sha256']:
            raise ValueError('Private causal patch changed')
        with np.load(path) as patch:
            values = {name: patch[name].copy() for name in patch.files}
        if metadata['different'] != 2 or len(values['indices']) != 2:
            raise ValueError('Expected two measured coordinates')
        CONTEXT.update(step=int(state.global_step) + 1, output=self.output, metadata=metadata, patch=values)
        module = dict(self.trainer.model.named_modules())['model.layers.1.mlp']
        original = module.forward

        def main_forward(self: Any, value: Any) -> Any:
            del self
            CONTEXT['active'] = True
            try:
                return original(value)
            finally:
                CONTEXT['active'] = False

        module.forward = MethodType(main_forward, module)
        (self.output / 'DIAGNOSTIC_ONLY.json').write_text(json.dumps({
            'eligible_for_alignment_acceptance': False, 'teacher_values_used': True,
            'scope': 'Only two main-router logit derivatives at212; model and optimizer state not replaced.',
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2))

    def end(self: Any, state: Any, **kwargs: Any) -> None:
        original_end(self, state, **kwargs)
        if int(state.global_step) >= 212 and CONTEXT['applied'] != 1:
            raise ValueError('Required causal intervention was not applied exactly once')
        CONTEXT['step'] = int(state.global_step) + 1

    def finalize() -> None:
        original_finalize()
        complete_begin = Observer.on_train_begin

        def after_factors(self: Any, state: Any, **kwargs: Any) -> None:
            complete_begin(self, state, **kwargs)
            begin(self, state, **kwargs)

        Observer.on_train_begin = after_factors

    RouterLinear.forward, RouterLinear.backward = staticmethod(forward), staticmethod(backward)
    Observer.on_step_end = end
    muon_dense_adjacent_factor.install = finalize
