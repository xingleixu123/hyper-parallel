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

"""Render complete verified200-step absolute errors as a standalone Chinese figure."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

ROOT = Path(__file__).resolve().parent


def main():
    report = json.loads((ROOT / 'evidence/frontier328-expert-independent200-audit.json').read_text(encoding='utf-8'))
    if not report['complete'] or any(len(rows) != 200 for rows in report['rows'].values()) or not all(report['fidelity'].values()):
        raise ValueError('Plot requires complete audited200-step reports')
    font = FontProperties(fname=r'C:\Windows\Fonts\msyh.ttc')
    plt.rcParams.update({'font.family': font.get_name(), 'axes.unicode_minus': False, 'svg.fonttype': 'path',
                         'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 10})
    figure, axes = plt.subplots(2, 3, figsize=(15.6, 8.5))
    names = [('total_loss', '总 loss'), ('lm_loss', 'LM loss'), ('mtp_raw', '原始 MTP loss'),
             ('mtp_weighted', '加权 MTP loss（×0.3）'), ('aux_raw', '原始 aux loss'),
             ('aux_weighted', '加权 aux loss（×0.0001）')]
    variants = [('control', 'fc1＋Q-up RoPE背景对照', '#6B7280'),
                ('expert_down', '仅增加专家down归约兼容', '#0072B2')]
    for axis, (key, title) in zip(axes.flat, names):
        for mode, label, color in variants:
            axis.plot([row['step'] for row in report['rows'][mode]],
                      [max(row['errors'][key], 1e-10) for row in report['rows'][mode]],
                      color=color, label=label, linewidth=1.25, alpha=.9)
        if key == 'total_loss':
            axis.axhline(.005, color='#A21CAF', linestyle='--', linewidth=1.1, label='总loss逐步阈值0.005')
        axis.set_yscale('log')
        axis.set_xlim(201, 400)
        axis.set_title(title, loc='left', fontsize=12, fontweight='bold')
        axis.set_xlabel('全局训练步')
        axis.set_ylabel('相对MF的绝对误差（对数轴）')
        axis.grid(axis='y', which='major', alpha=.22)
        axis.grid(axis='x', alpha=.12)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .935), ncol=3, frameon=False)
    figure.suptitle('专家down归约：总loss全部200步达标，超限3→0步', fontsize=18, x=.06, ha='left', y=.99)
    figure.text(.06, .022, '125单卡910B3｜MF CANN8.5 / HP CANN9.2｜完整恢复模型与优化器｜当前仅验收总loss，其余五项诊断；完整区间201–400；零值绘至1e-10。', fontsize=10)
    figure.subplots_adjust(left=.065, right=.985, top=.83, bottom=.11, wspace=.29, hspace=.39)
    for suffix in ['png', 'svg']:
        figure.savefig(ROOT / 'evidence' / ('frontier328-expert-independent200-loss-errors.' + suffix), dpi=140, facecolor='white')
    svg = ROOT / 'evidence/frontier328-expert-independent200-loss-errors.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()).rstrip() + '\n', encoding='utf-8')
    plt.close(figure)


if __name__ == '__main__':
    main()
