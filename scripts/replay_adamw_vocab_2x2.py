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

"""Replay identical real coordinates to isolate AdamW arithmetic from gradient differences."""
import json
from pathlib import Path
import numpy as np
import torch
import torch_npu
from hyper_parallel.core.optimizer.adamw import AdamW
from hyper_parallel.trainer.runtime.random import enable_full_determinism
from compare_vocab_optimizer_trace import stats

ROOT = Path('/alignment/phase12')
VARIANTS = ['fused', 'archive_eager', 'archive_with_lerp_m', 'archive_with_native_denom', 'native_order_eager']


def main():
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    enable_full_determinism(42)
    hp = {(r['step'], r['name']): r for r in map(json.loads, (ROOT/'runs/hp-denseadj-vocab-optimizer-coords/VOCAB_OPTIMIZER_TRACE.jsonl').read_text().splitlines())}
    mf = {(r['step'], r['name']): r for r in map(json.loads, (ROOT/'runs/mf-vocab-optimizer-coords-r2/VOCAB_OPTIMIZER_TRACE.jsonl').read_text().splitlines())}
    anchors = {(r['step'], r['name']): r['state'] for r in json.loads((ROOT/'evidence/vocab-native-coordinate-anchors.json').read_text())}
    def tensor(value):
        return torch.tensor(value, dtype=torch.float32, device='npu')
    def compare(value, expected):
        return stats(value.detach().cpu().numpy(), expected)
    results = []
    for name in ['model.embed_tokens.weight', 'lm_head.weight']:
        for source in ['hp_actual', 'mf_same_inputs']:
            for variant in VARIANTS:
                for step in range(201, 205):
                    actual = hp[(step, name)]
                    before = actual['before'] if source == 'hp_actual' else anchors[(step-1, name)]
                    gradient = actual['gradient'] if source == 'hp_actual' else mf[(step, name)]['gradient']
                    expected = actual['after'] if source == 'hp_actual' else anchors[(step, name)]
                    param = torch.nn.Parameter(tensor(before['weight']))
                    mom, var, grad = (tensor(before['exp_avg']), tensor(before['exp_avg_sq']), tensor(gradient))
                    betas = tuple(float(np.float32(x)) for x in actual['betas']) if variant == 'fused_fp32_betas' else tuple(actual['betas'])
                    optimizer = AdamW([param], lr=actual['lr'], weight_decay=actual['weight_decay'], betas=betas, eps=actual['eps'])
                    optimizer.param_groups[0]['step'] = step-1
                    optimizer.state[param].update(exp_avg=mom, exp_avg_sq=var)
                    param.grad = grad
                    with torch.no_grad():
                        if variant not in ('fused', 'fused_fp32_betas'):
                            beta1, beta2 = tensor([actual['betas'][0]]), tensor([actual['betas'][1]])
                            lr, decay, eps = tensor([actual['lr']]), tensor([actual['weight_decay']]), tensor([actual['eps']])
                            next_param = param * (1 - lr * decay)
                            mom.copy_(mom + (grad - mom) * (1 - beta1) if variant in ('archive_with_lerp_m', 'native_order_eager') else mom * beta1 + grad * (1 - beta1))
                            var.copy_(torch.addcmul(var * beta2, grad, grad, value=float(np.float32(1) - np.float32(actual['betas'][1]))))
                            bc2 = 1 - beta2.pow(step)
                            if variant in ('archive_with_native_denom', 'native_order_eager'):
                                inverse = torch.tensor([float(np.float32(1) / np.sqrt(np.float32(bc2.item())))], device='npu')
                                denom = var.sqrt() * inverse + eps
                            else:
                                denom = (var / bc2).sqrt() + eps
                            param.copy_(next_param - (mom / denom) * (lr / (1 - beta1.pow(step))))
                        else:
                            optimizer.step()
                    results.append({'step': step, 'name': name, 'source': source, 'variant': variant,
                                    'weight': compare(param, expected['weight']),
                                    'exp_avg': compare(mom, expected['exp_avg']),
                                    'exp_avg_sq': compare(var, expected['exp_avg_sq'])})
    report = {'complete': True, 'scope': 'Moment formula x denominator order2x2; explicit native-order candidate must be checked against fused before interpreting.', 'records': results}
    (ROOT/'evidence/adamw-vocab-moment-denom-2x2.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({source: {variant: {key: sum(row[key]['size'] - row[key]['equal_count'] for row in results if row['source'] == source and row['variant'] == variant) for key in ['weight', 'exp_avg', 'exp_avg_sq']} for variant in VARIANTS} for source in ['hp_actual', 'mf_same_inputs']}, indent=2))


if __name__ == '__main__':
    main()
