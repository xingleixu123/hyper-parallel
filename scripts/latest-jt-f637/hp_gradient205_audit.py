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

"""Audit step205 live gradients using read-only native MF references, without large dumps."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import capture_hp_native
import hyper_parallel.trainer.base as trainer_base
import native_clip_reduction_factor
from observe_hp_preclip_norm_adjacent import mf_gradient

ROOT = Path('/alignment/phase12')
STATE = Path('/state/jt-recipe-20261014')
CONTEXT = {}


def compare(actual, reference, coefficient=1.0):
    """Compare chunkwise; the coefficient is a CPU diagnostic, never a training input."""
    if actual.shape != reference.shape or actual.dtype != np.float32 or reference.dtype != np.float32:
        raise ValueError('Unexpected native gradient layout or dtype')
    a, b = actual.reshape(-1), reference.reshape(-1)
    error_sq = reference_sq = maximum = 0.0
    different = 0
    examples = []
    digest = hashlib.sha256()
    for start in range(0, a.size, 1_000_000):
        source = a[start:start+1_000_000]
        digest.update(source.tobytes())
        value = source * np.float32(coefficient)
        ref = b[start:start+1_000_000]
        error = value.astype(np.float64) - ref.astype(np.float64)
        error_sq += float(np.dot(error, error))
        reference_sq += float(np.dot(ref.astype(np.float64), ref.astype(np.float64)))
        maximum = max(maximum, float(abs(error).max()))
        different += int(np.count_nonzero(error))
        if len(examples) < 8:
            for index in np.flatnonzero(error)[:8-len(examples)]:
                examples.append({'flat_index': start+int(index), 'hp': float(value[index]), 'mf': float(ref[index])})
    return {'size': a.size, 'different': different, 'max_abs': maximum,
            'relative_l2': (error_sq/max(reference_sq, 1e-300))**.5,
            'actual_unscaled_sha256': digest.hexdigest(), 'examples': examples}


def install():
    """Observe both sides of the installed native clipping function at exactly205."""
    original_begin = capture_hp_native.Observer.on_train_begin
    original_install = native_clip_reduction_factor.install

    def begin(self, state, **kwargs):
        original_begin(self, state, **kwargs)
        rows = json.loads((ROOT/'runs/mf-grad205-state/GRADIENTS-205.json').read_text())
        native_metrics = [json.loads(line) for line in (ROOT/'runs/mf-resume/metrics.jsonl').read_text().splitlines()]
        native = next(row for row in native_metrics if row['step'] == 205)
        norm = np.float32(native['outputs'][4][0])
        coefficient = np.float32(1) / (norm + np.float32(1e-6))
        if len(rows) != 44:
            raise ValueError('Expected44 native parameters, excluding two vocabulary matrices')
        CONTEXT.update(step=int(state.global_step), output=self.output, rows=rows, coefficient=coefficient,
                       mf_norm=float(norm))

    def clip_install():
        original_install()
        original_clip = trainer_base.clip_grad_norm_

        @torch.no_grad()
        def clip(model, *args, **kwargs):
            CONTEXT['step'] += 1
            if CONTEXT['step'] != 205:
                return original_clip(model, *args, **kwargs)
            parameters = dict(model.named_parameters())
            records = []
            for row in CONTEXT['rows']:
                reference = np.load(STATE/'mf-grad205-state/gradients'/row['file'], mmap_mode='r')
                if hashlib.sha256(reference.tobytes()).hexdigest() != row['sha256']:
                    raise ValueError('MF reference hash changed: ' + row['name'])
                actual = mf_gradient(parameters, row['name']).float().cpu().numpy()
                records.append({'name': row['name'], 'shape': row['shape'],
                                'preclip_replay_mf_coefficient': compare(actual, reference, CONTEXT['coefficient'])})
                del actual, reference
            norm = original_clip(model, *args, **kwargs)
            for row, record in zip(CONTEXT['rows'], records):
                reference = np.load(STATE/'mf-grad205-state/gradients'/row['file'], mmap_mode='r')
                actual = mf_gradient(parameters, row['name']).float().cpu().numpy()
                record['actual_postclip'] = compare(actual, reference)
                del actual, reference
            report = {'complete': True, 'step': 205, 'records': records, 'mf_norm': CONTEXT['mf_norm'],
                      'hp_norm': float(norm.cpu()), 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      'scope': '44 complete parameter gradients excluding embedding/head. MF coefficient is used in read-only CPU replay; native training values unchanged. No extra gradient arrays saved.'}
            (ROOT/'evidence'/(CONTEXT['output'].name+'-grad205.json')).write_text(json.dumps(report, indent=2))
            print('[P12-FULL-GRAD205-AUDIT]', json.dumps({'count': len(records),
                  'replay_exact': sum(r['preclip_replay_mf_coefficient']['different']==0 for r in records),
                  'postclip_exact': sum(r['actual_postclip']['different']==0 for r in records)}), flush=True)
            return norm

        trainer_base.clip_grad_norm_ = clip

    capture_hp_native.Observer.on_train_begin = begin
    native_clip_reduction_factor.install = clip_install
