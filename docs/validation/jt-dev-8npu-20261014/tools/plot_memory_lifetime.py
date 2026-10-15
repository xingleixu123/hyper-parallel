"""Plot measured before/after memory and timing without attributing unrelated fixes."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--chinese', action='store_true')
chinese = parser.parse_args().chinese
names = [('EP2', 'pr180-ep2-off', 'reviewed-ep2-off-200'),
         ('CP2', 'fixed-tp1cp2ep1-off-200', 'reviewed-cp2-off-200')]
before, after = [], []
for label, old, new in names:
    old_report = json.loads((root / f'evidence/{old}-comparison.json').read_text())
    new_report = json.loads((root / f'evidence/{new}-comparison.json').read_text())
    assert new_report['status'] == 'PASS'
    assert all(entry['max_abs'] == 0 for entry in new_report['errors'].values())
    before.append(old_report['performance'])
    after.append(new_report['performance'])

plt.rcParams.update({'font.family': 'Microsoft YaHei' if chinese else 'DejaVu Sans', 'font.size': 11,
                     'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none'})
figure, axes = plt.subplots(1, 3, figsize=(14, 5.3))
positions = np.arange(2)
for axis, key, title, unit in zip(axes,
        ('allocated_gib', 'reserved_gib', 'median_step_seconds'),
        (('已分配显存峰值', '预留显存峰值', '优化器步耗时中位数') if chinese
         else ('Peak allocated memory', 'Peak reserved memory', 'Median optimizer step')),
        (('GiB / 卡', 'GiB / 卡', '秒') if chinese else ('GiB / device', 'GiB / device', 'seconds'))):
    for offset, records, label, color in ((-0.18, before, '修复前' if chinese else 'Before', '#A6B3C4'),
                                          (0.18, after, '修复后' if chinese else 'After', '#15755D')):
        values = [row[key] for row in records]
        bars = axis.bar(positions + offset, values, width=0.34, color=color, label=label)
        axis.bar_label(bars, labels=[f'{value:.3f}' for value in values], padding=4, fontsize=10)
    axis.set_xticks(positions, [row[0] for row in names])
    axis.set_title(title, loc='left', weight='bold', fontsize=12)
    axis.set_ylabel(unit)
    axis.set_ylim(0, max(row[key] for row in before + after) * 1.23)
    axis.grid(axis='y', alpha=0.2)
    axis.set_axisbelow(True)
handles, labels = axes[0].get_legend_handles_labels()
figure.legend(handles, labels, frameon=False, loc='upper right', bbox_to_anchor=(0.98, 0.86), ncol=2)
figure.text(0.06, 0.93, '释放完整源参数：EP2 / CP2 显存收益实测' if chinese
            else 'Release initialization parameters: measured EP2 / CP2 memory savings',
             fontsize=17, weight='bold')
figure.text(0.06, 0.87, '历史轮次 ec2e16b | 8 × 910B3 | CANN 9.2.0-beta.2 | 各 200 步、同拓扑回归' if chinese
            else '8 x Ascend 910B3 | CANN 9.2.0-beta.2 | 200 steps per case | Same-topology regression',
             color='#526276', fontsize=10)
figure.text(0.06, 0.035, ('同拓扑修复前后，全部 LM / MTP / aux / 总损失和梯度范数均精确相同。\n'
            '耗时排除前 20 步；每种配置仅运行一次。跨拓扑精度另行验收；不代表最新 f637ef7 基点结果。') if chinese else
            'All LM/MTP/aux/total losses and gradient norms remain exactly equal to the respective pre-fix topology.\n'
            'Timing excludes the first 20 steps; one run per configuration. Cross-topology accuracy is evaluated separately.',
             color='#526276', fontsize=9.5)
figure.subplots_adjust(left=0.06, right=0.98, top=0.77, bottom=0.2, wspace=0.32)
for extension in ('png', 'svg'):
    suffix = '-zh' if chinese else ''
    figure.savefig(root / 'figures' / f'source-parameter-lifetime{suffix}.{extension}', dpi=180)
plt.close(figure)
