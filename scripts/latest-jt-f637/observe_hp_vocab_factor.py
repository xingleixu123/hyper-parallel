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

"""Capture two live vocabulary gradients and small DVM partials before unchanged clipping."""
import json
import os
from pathlib import Path
import numpy as np
import torch
import hyper_parallel.trainer.base as trainer_base
from observe_hp_preclip_norm import mf_gradient
from muon_q_norm_probe import half_tree


def partials(value):
    count = value.numel()
    tile, width = ((count, 64) if count <= 5120 else ((1024, 64) if count == 40960 else (8144, 16)))
    flat = value.reshape(-1)
    results = []
    for start in range(0, count, tile * 512):
        piece = flat[start:start + tile * 512]
        tiles = (piece.numel() + tile - 1) // tile
        padded = piece.new_zeros(tiles * tile)
        padded[:piece.numel()] = piece * piece
        tiled = padded.reshape(tiles, tile // width, width).transpose(1, 2)
        results.append(half_tree(half_tree(tiled)).cpu().numpy())
    return np.concatenate(results)


def install(output):
    original = trainer_base.clip_grad_norm_
    order = json.loads(Path('/alignment/phase12/configs/mf-gradient-layout-order.json').read_text())
    scope = os.environ['P12_VOCABULARY_FACTOR']
    root = Path('/state/jt-recipe-20261014/hp-vocab-' + scope + '201')
    root.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def observed(model, max_norm, *args, **kwargs):
        parameters = dict(model.named_parameters())
        rows = []
        for index, entry in enumerate(order):
            value = mf_gradient(parameters, entry['name'])
            if list(value.shape) != entry['shape']:
                raise ValueError('Wrong layout: ' + entry['name'])
            path = root / ('partials-%02d.npy' % index)
            np.save(path, partials(value))
            row = dict(entry, partials=str(path))
            if index == (29 if scope.startswith('head-') else 0):
                path = root / ('gradient-%02d.npy' % index)
                array = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=tuple(entry['shape']))
                for start in range(0, value.shape[0], 1024):
                    array[start:start + 1024] = value[start:start + 1024].cpu().numpy()
                array.flush()
                del array
                row['gradient'] = str(path)
            rows.append(row)
            print('[P12-PRECLIP-VOCAB]', index, entry['name'], flush=True)
            del value
        actual = original(model, max_norm, *args, **kwargs)
        report = {'complete': True, 'reference_values_used': False, 'clip_behavior_changed': False,
                  'native_norm': float(actual.cpu()), 'records': rows}
        output.write_text(json.dumps(report, indent=2))
        return actual

    trainer_base.clip_grad_norm_ = observed
