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

"""Diagnostic-only one-time Q/router weight interventions after update208."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

import capture_hp_native
from hp_head206_intervention import apply_patch, digest
import hp_head206_intervention

ROOT = Path('/alignment/phase12')
STATE = Path('/state/jt-recipe-20261014')
TARGETS = {'q': ('qkv', 'model.layers.0.self_attn.linear_qkv.weight', 4477438),
           'router': ('router', 'model.layers.1.mlp.gate.weight', 15833)}
CONTEXT = {}


def install():
    """Patch only measured weight coordinates; verify the actual momentum is unchanged."""
    original_begin = capture_hp_native.Observer.on_train_begin
    original_end = capture_hp_native.Observer.on_step_end
    mode = os.environ['P12_MODEL208_INTERVENTION']
    if mode not in ('q', 'router', 'q_router'):
        raise ValueError('Unexpected declared intervention')

    def begin(self, state, **kwargs):
        original_begin(self, state, **kwargs)
        comparison = ROOT / 'evidence/frontier208-gradient-state-comparison.json'
        report = json.loads(comparison.read_text())
        selected = ('q', 'router') if mode == 'q_router' else (mode,)
        for label in selected:
            key, name, count = TARGETS[label]
            hp = np.load(STATE / 'hp-jtf637-headrouter-state208/state-208' / (key + '-weight.npy'))
            mf = np.load(STATE / 'mf-frontier208-state/state-208' / (label + '-weight.npy'))
            if label == 'q':
                kv = np.load(STATE / 'mf-frontier208-state/state-208/kv-weight.npy')
                mf = np.concatenate([mf, kv], axis=0)
                if not np.array_equal(hp[1536:], kv):
                    raise ValueError('KV must remain exact')
            expected = report['states']['208'][key]
            before, after = expected['weight']['hp_sha256'], expected['weight']['mf_sha256']
            if digest(hp) != before or digest(mf) != after or not expected['muon_m']['exact']:
                raise ValueError('Reference model or equal momentum evidence changed')
            indices = np.flatnonzero(hp.reshape(-1) != mf.reshape(-1))
            if indices.size != count or (label == 'q' and np.any(indices >= 1536 * 5120)):
                raise ValueError('Unexpected difference count or scope')
            CONTEXT[name] = (indices, mf.reshape(-1)[indices].copy(), before, after,
                             expected['muon_m']['hp_sha256'])
        marker = {'diagnostic_only': True, 'eligible_for_acceptance': False, 'step': 208, 'mode': mode,
                  'optimizer_state_modified': False, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'patch_helper_sha256': hashlib.sha256(Path(hp_head206_intervention.__file__).read_bytes()).hexdigest(),
                  'comparison_sha256': hashlib.sha256(comparison.read_bytes()).hexdigest()}
        (self.output / 'DIAGNOSTIC_ONLY.json').write_text(json.dumps(marker, indent=2))

    @torch.no_grad()
    def end(self, state, **kwargs):
        original_end(self, state, **kwargs)
        if int(state.global_step) != 208:
            return
        parameters = dict(self.trainer.model.named_parameters())
        changed = {}
        for name, values in CONTEXT.items():
            parameter = parameters[name]
            matches = [leaf.state[parameter] for leaf in capture_hp_native.leaves(self.trainer.optimizer)
                       if parameter in leaf.state]
            if len(matches) != 1:
                raise ValueError('Expected one actual optimizer state')
            momentum = matches[0]['momentum_buffer']
            before_m = digest(capture_hp_native.numpy_value(momentum))
            if before_m != values[4]:
                raise ValueError('Momentum does not reproduce the independent control')
            record = apply_patch(parameter, *values[:4])
            after_m = digest(capture_hp_native.numpy_value(momentum))
            if before_m != after_m:
                raise ValueError('Momentum modified during weight intervention')
            record['momentum_sha256_before_after'] = after_m
            changed[name] = record
        report = {'step': 208, 'mode': mode, 'changed': changed, 'optimizer_state_modified': False}
        (self.output / 'MODEL208_INTERVENTION.json').write_text(json.dumps(report, indent=2))

    capture_hp_native.Observer.on_train_begin = begin
    capture_hp_native.Observer.on_step_end = end
