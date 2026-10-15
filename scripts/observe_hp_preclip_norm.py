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

"""Observe live pre-clip gradients with MF logical layouts and native DVM reduction tiling."""
import json
from pathlib import Path

import numpy as np
import torch

import hyper_parallel.trainer.base as trainer_base
from hyper_parallel.core.utils.clip_grad import _get_local_grad
from audit_native_q_reduction_schedules import half_tree as numpy_half_tree
from muon_q_norm_probe import half_tree


def local_gradient(parameters, name):
    value = _get_local_grad(parameters[name])
    if value is None:
        raise ValueError('Missing preclip gradient: ' + name)
    if hasattr(value, 'to_local'):
        value = value.to_local()
    if value.dtype != torch.float32:
        raise ValueError('Expected FP32 gradient: ' + name)
    return value.detach()


def mf_gradient(parameters, name):
    direct = {'embedding.word_embeddings.weight': 'model.embed_tokens.weight',
              'output_layer.weight': 'lm_head.weight', 'decoder.final_layernorm.weight': 'model.norm.weight'}
    if name in direct:
        return local_gradient(parameters, direct[name])
    target = name.replace('decoder.layers.', 'model.layers.')
    target = target.replace('.self_attention.', '.self_attn.').replace('.pre_mlp_layernorm.', '.post_attention_layernorm.')
    for old, new in [('q_layernorm', 'q_a_layernorm'), ('kv_layernorm', 'kv_a_layernorm'),
                     ('linear_q_up_proj', 'q_b_proj'), ('linear_kv_up_proj', 'kv_b_proj'),
                     ('linear_proj', 'o_proj')]:
        target = target.replace('.' + old + '.', '.' + new + '.')
    for suffix, selection in [('linear_q_down_proj.weight', slice(0, 1536)),
                              ('linear_kv_down_proj.weight', slice(1536, 2112))]:
        if target.endswith(suffix):
            return local_gradient(parameters, target[:-len(suffix)] + 'linear_qkv.weight')[selection]
    if target.endswith('.linear_fc1.weight'):
        prefix = target[:-len('linear_fc1.weight')]
        gate = local_gradient(parameters, prefix + 'gate_proj.weight')
        up = local_gradient(parameters, prefix + 'up_proj.weight')
        return torch.stack((gate, up), dim=1).flatten(0, 1)
    if target.endswith('.experts.weight1'):
        return local_gradient(parameters, target[:-len('weight1')] + 'gate_up_proj').transpose(1, 2).contiguous().flatten(0, 1)
    if target.endswith('.experts.weight2'):
        return local_gradient(parameters, target[:-len('weight2')] + 'down_proj').transpose(1, 2).contiguous().flatten(0, 1)
    target = target.replace('.linear_fc2.weight', '.down_proj.weight').replace('.mlp.router.weight', '.mlp.gate.weight')
    return local_gradient(parameters, target)


def sequential(values):
    total = np.float32(0)
    for value in values:
        total = np.float32(total + value)
    return total


def dvm_square_sum(value):
    count = value.numel()
    tile, width = ((count, 64) if count <= 5120 else ((1024, 64) if count == 40960 else (8144, 16)))
    flat = value.reshape(-1)
    partials = []
    chunk_size = tile * 512
    for start in range(0, count, chunk_size):
        piece = flat[start:start + chunk_size]
        tile_count = (piece.numel() + tile - 1) // tile
        padded = piece.new_zeros(tile_count * tile)
        padded[:piece.numel()] = piece * piece
        tiles = padded.reshape(tile_count, tile // width, width).transpose(1, 2)
        partials.append(half_tree(half_tree(tiles)).cpu().numpy())
    partials = np.concatenate(partials)
    if len(partials) == 1:
        return partials[0]
    accumulators = np.zeros(40, dtype=np.float32)
    for i, partial in enumerate(partials):
        accumulators[i % 40] = np.float32(accumulators[i % 40] + partial)
    return sequential(accumulators)


def install(output):
    original = trainer_base.clip_grad_norm_
    order = json.loads(Path('/alignment/phase12/configs/mf-gradient-layout-order.json').read_text())

    @torch.no_grad()
    def observed(model, max_norm, *args, **kwargs):
        parameters = dict(model.named_parameters())
        rows = []
        for entry in order:
            value = mf_gradient(parameters, entry['name'])
            if list(value.shape) != entry['shape']:
                raise ValueError('Wrong MF layout: ' + entry['name'])
            native_sum = float((value * value).sum().cpu())
            dvm_sum = dvm_square_sum(value)
            rows.append({'name': entry['name'], 'shape': entry['shape'],
                         'torch_square_sum': native_sum, 'dvm_square_sum': float(dvm_sum)})
            print('[P12-PRECLIP-NORM]', json.dumps(rows[-1]), flush=True)
            del value
        actual = original(model, max_norm, *args, **kwargs)
        report = {'complete': True, 'reference_values_used': False, 'clip_behavior_changed': False,
                  'native_norm': float(actual.cpu()), 'records': rows, 'candidates': {}}
        for key in ['torch_square_sum', 'dvm_square_sum']:
            values = np.array([row[key] for row in rows], dtype=np.float32)
            for join, total in [('sequential', sequential(values)), ('numpy_sum', values.sum()),
                                ('half_tree', numpy_half_tree(values))]:
                report['candidates'][key + '_' + join] = {'sum': float(total), 'norm': float(np.sqrt(total))}
        output.write_text(json.dumps(report, indent=2))
        return actual

    trainer_base.clip_grad_norm_ = observed
