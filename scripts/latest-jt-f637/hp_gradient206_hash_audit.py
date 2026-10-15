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

"""Compare complete live step206 gradient hashes against retained native MF hashes."""
import hashlib
import json
from pathlib import Path

import torch
import capture_hp_native
import hyper_parallel.trainer.base as trainer_base
import native_clip_reduction_factor
from observe_hp_preclip_norm_adjacent import mf_gradient

ROOT = Path('/alignment/phase12')
CONTEXT = {}


def install():
    """Observe native clipping without saving another full gradient snapshot."""
    original_begin = capture_hp_native.Observer.on_train_begin
    original_install = native_clip_reduction_factor.install

    def begin(self, state, **kwargs):
        original_begin(self, state, **kwargs)
        path = ROOT / 'runs/mf-grad205-state/GRADIENTS-206.json'
        rows = json.loads(path.read_text())
        if len(rows) != 44:
            raise ValueError('Expected44 complete native gradient hashes')
        CONTEXT.update(step=int(state.global_step), output=self.output, rows=rows,
                       reference_sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    def clip_install():
        original_install()
        original_clip = trainer_base.clip_grad_norm_

        @torch.no_grad()
        def clip(model, *args, **kwargs):
            CONTEXT['step'] += 1
            norm = original_clip(model, *args, **kwargs)
            if CONTEXT['step'] != 206:
                return norm
            parameters = dict(model.named_parameters())
            records = []
            for row in CONTEXT['rows']:
                actual = mf_gradient(parameters, row['name']).float().cpu().numpy()
                if list(actual.shape) != row['shape']:
                    raise ValueError('Reference gradient shape changed')
                digest = hashlib.sha256(actual.tobytes()).hexdigest()
                records.append({'name': row['name'], 'shape': row['shape'], 'actual_sha256': digest,
                                'native_mf_sha256': row['sha256'], 'exact': digest == row['sha256']})
                del actual
            report = {'complete': True, 'step': 206, 'records': records, 'hp_norm': float(norm.cpu()),
                      'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      'reference_manifest_sha256': CONTEXT['reference_sha256'],
                      'scope': '44 complete postclip gradients excluding two vocabulary matrices. Hashes only; no teacher input, no new tensor dumps.'}
            (ROOT / 'evidence' / (CONTEXT['output'].name + '-grad206.json')).write_text(json.dumps(report, indent=2))
            print('[P12-GRAD206-HASH]', sum(row['exact'] for row in records), '/44 exact', flush=True)
            return norm

        trainer_base.clip_grad_norm_ = clip

    capture_hp_native.Observer.on_train_begin = begin
    native_clip_reduction_factor.install = clip_install
