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

"""Plot a verified diagnostic prefix without presenting it as full-run acceptance."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Show all-rank scalar differences and label the missing reference window explicitly."""
    source = ROOT / 'evidence/f637-dgrad-long-cp2-v2-prefix-analysis.json'
    result = json.loads(source.read_bytes())
    if result['observed_comparison_steps'] != [1, 163] or result['acceptance_200_steps'] is not False:
        raise ValueError('Unexpected comparison scope')
    reports = [(result['native_control_200_step_result'], '原生输入梯度', '#147A74'),
               (result['strict_comparator_result'], 'FP32 输入梯度后转 BF16', '#B55337')]
    plt.rcParams.update({'font.family': 'Microsoft YaHei', 'font.size': 11, 'svg.fonttype': 'none',
                         'axes.spines.top': False, 'axes.spines.right': False})
    figure, axes = plt.subplots(1, 3, figsize=(16.5, 7.5))
    labels = [('lm', 1.0, '主 LM loss（当前目标）'), ('mtp', 0.3, 'MTP loss（按系数还原）'),
              ('aux', 0.0001, 'Aux loss（按系数还原）')]
    for axis, (key, divisor, title) in zip(axes, labels, strict=True):
        for report, label, color in reports:
            values = [value / divisor for value in report['step_errors']['training/foundation_loss/' + key][:163]]
            if len(values) != 163:
                raise ValueError('Missing per-step plotted values')
            axis.plot(range(1, 164), values, linewidth=1.6, color=color,
                      label=f'{label}\n区间最大差 {max(values):.7f}')
        axis.axhline(0.005, color='#52525B', linestyle='--', linewidth=1.2, label='0.005 阈值')
        axis.set_title(title, loc='left', weight='bold', pad=15)
        axis.set_xlim(1, 163)
        axis.set_ylim(bottom=0)
        axis.set_xlabel('训练步（仅共有 1–163 步）')
        axis.grid(alpha=0.17)
        axis.legend(frameon=False, loc='upper left', fontsize=9)
    axes[0].set_ylabel('CP2 与同配方 FSDP8：逐步全部 8 rank 最大绝对差')
    verdict = ('共有 163 步已超 0.005' if result['main_lm_first_step_above_0_005'] is not None
               else '共有 163 步内未超阈值，仍待完整验证')
    figure.text(0.045, 0.945, 'FP32 Linear 输入梯度：' + verdict, fontsize=21, weight='bold')
    figure.text(0.045, 0.895, '129 · CANN 9.2.0-beta.2 · 源码 18602c4 · 原输入顺序、LR 与 loss 系数保持固定',
                fontsize=11, color='#475569')
    figure.text(0.045, 0.205,
                'FP32 配方的 FSDP8 参考因设备冲突停在 163 步；CP2 v2 完整运行 200 步。\n'
                '图中每条曲线均比较相同数值配方下的 CP2 与 FSDP8；164–200 步缺少配对参考，未画入。',
                fontsize=11, linespacing=1.65)
    figure.text(0.045, 0.085,
                '本图不能证明 200 步验收通过。MTP / Aux 为保留的观测，不作为当前主 loss 候选的淘汰依据。\n'
                '诊断包含额外的权重梯度重放开销，不据此声称生产性能收益。',
                fontsize=10, color='#475569', linespacing=1.65)
    figure.subplots_adjust(left=0.075, right=0.98, top=0.765, bottom=0.35, wspace=0.29)
    outputs = []
    for extension in ('png', 'svg'):
        path = ROOT / 'figures' / f'f637-dgrad-long-prefix-zh.{extension}'
        if path.exists():
            raise FileExistsError('Refusing to replace an existing diagnostic figure')
        figure.savefig(path, dpi=180, facecolor='white')
        outputs.append({'path': path.relative_to(ROOT).as_posix(),
                        'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    plt.close(figure)
    receipt = {'analysis_sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'files': outputs,
               'comparison_steps': [1, 163], 'acceptance_200_steps': False}
    (ROOT / 'evidence/f637-dgrad-long-prefix-figure-manifest.json').write_text(
        json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
