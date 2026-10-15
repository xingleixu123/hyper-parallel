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
from decimal import Decimal, localcontext
from pathlib import Path
import numpy as np
import torch
import torch_npu
from hyper_parallel.core.optimizer.adamw import AdamW
from hyper_parallel.trainer.runtime.random import enable_full_determinism
from compare_vocab_optimizer_trace import stats

ROOT = Path('/alignment/phase12')
VARIANTS = ['fused', 'archive_eager']


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
        for source in ['mf_same_inputs']:
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
                        if variant == 'archive_eager':
                            beta1, beta2 = tensor([actual['betas'][0]]), tensor([actual['betas'][1]])
                            lr, decay, eps = tensor([actual['lr']]), tensor([actual['weight_decay']]), tensor([actual['eps']])
                            next_param = param * (1 - lr * decay)
                            mom.copy_(mom * beta1 + grad * (1 - beta1))
                            var.copy_(torch.addcmul(var * beta2, grad, grad, value=float(np.float32(1) - np.float32(actual['betas'][1]))))
                            denom = (var / (1 - beta2.pow(step))).sqrt() + eps
                            param.copy_(next_param - (mom / denom) * (lr / (1 - beta1.pow(step))))
                        else:
                            optimizer.step()
                    results.append({'step': step, 'name': name, 'source': source, 'variant': variant,
                                    'weight': compare(param, expected['weight']),
                                    'exp_avg': compare(mom, expected['exp_avg']),
                                    'exp_avg_sq': compare(var, expected['exp_avg_sq']),
                                    'values': {'weight': param.detach().cpu().tolist(),
                                               'exp_avg': mom.cpu().tolist(), 'exp_avg_sq': var.cpu().tolist()}})
    comparisons = []
    with localcontext() as context:
        context.prec = 80
        for name in ['model.embed_tokens.weight', 'lm_head.weight']:
            for coefficients in ['exact_fp32_hyperparameters', 'nominal_decimal_hyperparameters']:
                totals = {field: {'fused_closer': 0, 'archive_closer': 0, 'tie': 0,
                                  'fused_sum_abs': Decimal(0), 'archive_sum_abs': Decimal(0)}
                          for field in ['weight', 'exp_avg', 'exp_avg_sq']}
                examples = []
                for step in range(201, 205):
                    record = hp[(step, name)]
                    def scalar(value):
                        return Decimal.from_float(float(np.float32(value))) if coefficients.startswith('exact') else Decimal(str(value))
                    b1, b2, lr, decay, eps = [scalar(value) for value in [*record['betas'], record['lr'], record['weight_decay'], record['eps']]]
                    previous, gradient = anchors[(step-1, name)], mf[(step, name)]['gradient']
                    actual = {row['variant']: row['values'] for row in results if row['name'] == name and row['step'] == step}
                    for index in range(len(gradient)):
                        weight, moment, variance, grad = [Decimal.from_float(float(value)) for value in
                            [previous['weight'][index], previous['exp_avg'][index], previous['exp_avg_sq'][index], gradient[index]]]
                        next_m = b1 * moment + (1-b1) * grad
                        next_v = b2 * variance + (1-b2) * grad * grad
                        next_w = weight * (1-lr*decay) - lr * (next_m/(1-b1**step)) / ((next_v/(1-b2**step)).sqrt()+eps)
                        reference = {'weight': next_w, 'exp_avg': next_m, 'exp_avg_sq': next_v}
                        for field, expected in reference.items():
                            fused, archive = [Decimal.from_float(float(actual[variant][field][index])) for variant in VARIANTS]
                            errors = abs(fused-expected), abs(archive-expected)
                            outcome = 'fused_closer' if errors[0] < errors[1] else 'archive_closer' if errors[1] < errors[0] else 'tie'
                            totals[field][outcome] += 1
                            totals[field]['fused_sum_abs'] += errors[0]
                            totals[field]['archive_sum_abs'] += errors[1]
                            if field == 'weight' and outcome != 'tie' and len(examples) < 10:
                                examples.append({'step': step, 'coordinate_slot': index, 'outcome': outcome,
                                                 'reference80': str(expected), 'fused': float(fused), 'archive': float(archive)})
                for values in totals.values():
                    for key in ['fused_sum_abs', 'archive_sum_abs']:
                        values[key] = str(values[key])
                comparisons.append({'name': name, 'coefficients': coefficients, 'metrics': totals, 'examples': examples})
    for row in results:
        row.pop('values')
    report = {'complete': True, 'reference_decimal_precision': 80,
              'scope': 'Selected real coordinates201-204, same native states/gradients; two explicit coefficient interpretations, not universal accuracy claim.',
              'same_input_replay': results, 'accuracy': comparisons}
    (ROOT/'evidence/adamw-vocab-high-precision-accuracy.json').write_text(json.dumps(report, indent=2))
    print(json.dumps([{'name': row['name'], 'coefficients': row['coefficients'], 'metrics': row['metrics']} for row in comparisons], indent=2))



if __name__ == '__main__':
    main()
