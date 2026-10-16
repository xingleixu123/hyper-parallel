# Copyright 2026 Huawei Technologies Co., Ltd
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

"""Render a verified rejected prefix while keeping the missing candidate steps explicit."""

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position

from compare_runs import read_rows  # pylint: disable=wrong-import-position


ROOT = Path(__file__).resolve().parents[1]
METRIC = 'training/foundation_loss/'


def main() -> None:
    """Plot only paired observed LM curves and all-rank maximum scalar errors."""
    source = ROOT / 'evidence/f637-dense-wgrad-cp2-v1-prefix-analysis.json'
    result = json.loads(source.read_bytes())
    if result['comparison_steps'] != [1, 159] or result['ranks'] != 8 or result['acceptance_200_steps'] is not False:
        raise ValueError('Expected the verified 159-step prefix without final acceptance')
    comparison = result['strict_all_component_comparison']
    native = result['native_recipe_comparison']
    if native['findings'] or result['candidate_audit']['status'] != 'VERIFIED_INTERRUPTED_PREFIX':
        raise ValueError('Candidate prefix or native control has not been verified')
    coefficients = comparison['loss_coefficients']
    if coefficients != native['loss_coefficients']:
        raise ValueError('Cannot compare differently weighted loss diagnostics')
    plt.rcParams.update({'font.family': 'Microsoft YaHei', 'font.size': 10, 'svg.fonttype': 'none',
                         'axes.spines.top': False, 'axes.spines.right': False})
    figure, axes = plt.subplots(2, 2, figsize=(16, 10.5))
    raw_hashes = {}
    for layout, label, color in [('fsdp8', '同配方 FSDP8', '#147A74'), ('cp2', '同配方 CP2', '#B55337')]:
        directory = ROOT / 'runs' / f'r129-f637-dense-wgrad-{layout}-200-v1'
        raw = directory / 'metrics-rank0.jsonl'
        manifest = json.loads((ROOT / f'evidence/f637-dense-wgrad-{layout}-200-v1-evidence-manifest.json').read_bytes())
        expected = next(item['sha256'] for item in manifest['files']
                        if item['path'] == raw.relative_to(ROOT).as_posix())
        digest = hashlib.sha256(raw.read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError('A raw training trace changed after collection')
        raw_hashes[raw.relative_to(ROOT).as_posix()] = digest
        rows = read_rows(directory, 0)
        if set(rows) != set(range(1, 201 if layout == 'fsdp8' else 160)):
            raise ValueError('Raw LM trace has an unexpected complete or interrupted window')
        axes[0, 0].plot(range(1, 160), [rows[step][METRIC + 'lm'] for step in range(1, 160)],
                        label=label, linewidth=1.2, color=color, alpha=0.9)
    axes[0, 0].set_title('候选配方：主 LM loss 原始曲线（rank 0）', loc='left', weight='bold', pad=12)
    axes[0, 0].set_ylabel('主 LM loss')
    panels = [(axes[0, 1], 'lm', 1.0, '主 LM loss：当前验收项'),
              (axes[1, 0], 'mtp', coefficients['mtp_loss_factor'], 'MTP loss：按系数还原的观测'),
              (axes[1, 1], 'aux', coefficients['moe_aux_loss_coeff'], 'Aux loss：按系数还原的观测')]
    for axis, key, divisor, title in panels:
        for report, label, color in [(native, '原数值配方', '#78938F'),
                                      (comparison, 'Linear dW 直接 FP32 累加', '#B55337')]:
            values = [value / divisor for value in report['step_errors'][METRIC + key][:159]]
            if len(values) != 159:
                raise ValueError('Missing per-step errors')
            axis.plot(range(1, 160), values, linewidth=1.3, color=color,
                      label=f'{label} · 最大差 {max(values):.7f}')
        if key == 'lm':
            axis.axhline(0.005, color='#52525B', linestyle='--', linewidth=1.1, label='主 loss 阈值 0.005')
        axis.set_title(title, loc='left', weight='bold', pad=12)
        axis.set_ylabel('全部 8 rank 的逐步最大绝对差')
        axis.set_ylim(bottom=0)
    for axis in axes.flat:
        axis.set_xlim(1, 159)
        axis.set_xlabel('共有训练步（1–159）')
        axis.grid(alpha=0.16)
        axis.legend(frameon=False, fontsize=9)
    maximum = result['main_lm_max_absolute_error']
    figure.text(0.06, 0.955, f'FP32 Linear 权重梯度：159 步前缀主 loss 最大差 {maximum:.7f}',
                fontsize=20, weight='bold')
    figure.text(0.06, 0.917,
                '129 · 共有 159 步 × 8 rank · CANN 9.2.0-beta.2 · 源码 18602c4 · 固定数据、初值、LR 和 loss 系数',
                fontsize=10.5, color='#475569')
    figure.text(0.06, 0.085,
                'FSDP8 完成 200 步，CP2 在 159 步后遇其他任务让卡；仅比较共有前缀，不作完整 200 步验收。\n'
                '第 36 步首次超 0.005，已有证据否决该候选。误差曲线分别比较各自数值配方的 CP2 与 FSDP8。\n'
                'MTP / Aux 保留作观测；此图不代表 TP、EP、续训或重算已通过。',
                fontsize=10, color='#475569', linespacing=1.6)
    figure.subplots_adjust(left=0.075, right=0.97, top=0.84, bottom=0.20, hspace=0.48, wspace=0.22)
    outputs = []
    for extension in ('png', 'svg'):
        path = ROOT / 'figures' / f'f637-dense-wgrad-prefix-zh.{extension}'
        if path.exists():
            raise FileExistsError('Refusing to replace an existing precision figure')
        figure.savefig(path, dpi=180, facecolor='white')
        outputs.append({'path': path.relative_to(ROOT).as_posix(),
                        'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    plt.close(figure)
    receipt = {'analysis_sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'files': outputs,
               'raw_trace_sha256': raw_hashes, 'comparison_steps': [1, 159],
               'acceptance_200_steps': False, 'status': result['status']}
    with (ROOT / 'evidence/f637-dense-wgrad-prefix-figure-manifest.json').open('x', encoding='utf-8') as handle:
        json.dump(receipt, handle, ensure_ascii=False, indent=2)
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == '__main__':
    main()
