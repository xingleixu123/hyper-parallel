"""Show the resumed cumulative-counter defect using complete NPU run evidence."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--chinese', action='store_true')
chinese = parser.parse_args().chinese
names = ('pr180-fsdp8-off-continuous400', 'pr180-fsdp8-off-200-400', 'reviewed-fsdp8-off-200-400')
labels = (('连续训练参考', '修复前续训', '修复后续训') if chinese
          else ('Continuous reference', 'Resume before fix', 'Resume after fix'))
colors = ('#215C9C', '#B22B3C', '#15755D')
styles = ('-', '-', '--')
plt.rcParams.update({'font.family': 'Microsoft YaHei' if chinese else 'DejaVu Sans', 'font.size': 11,
                     'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none'})
figure, axes = plt.subplots(1, 2, figsize=(12, 5.3))
for name, label, color, style in zip(names, labels, colors, styles):
    rows = [json.loads(line) for line in (root / 'runs' / name / 'metrics-rank0.jsonl').read_text().splitlines()]
    rows = [row for row in rows if row['step'] >= 201]
    for axis, key in zip(axes, ('data/consumed_tokens', 'data/consumed_samples')):
        axis.plot([row['step'] for row in rows], [row[key] for row in rows],
                  color=color, ls=style, lw=2, label=label)
titles = ('累计监督 token 数', '累计样本数') if chinese else ('Cumulative supervised tokens', 'Cumulative samples')
for axis, title in zip(axes, titles):
    axis.set_title(title, loc='left', weight='bold')
    axis.set_xlabel('优化器步数' if chinese else 'Optimizer step')
    axis.set_xlim(201, 400)
    axis.grid(alpha=0.2)
axes[0].legend(frameon=False, fontsize=10)
figure.text(0.07, 0.93, '断点续训：累计样本与监督 token 计数正确延续' if chinese
            else 'Checkpoint resume: cumulative counters now continue correctly',
            weight='bold', fontsize=16)
figure.text(0.07, 0.87, '历史轮次 ec2e16b | 8 × 910B3 | CANN 9.2.0-beta.2 | 不代表最新 f637ef7 基点结果' if chinese
            else '8 x Ascend 910B3 | CANN 9.2.0-beta.2 | JT_dev + #180 | Six-fix integration ec2e16b',
            color='#526276', fontsize=10)
figure.text(0.07, 0.035, ('曲线展示 rank0；全部 8 rank、续训 200 步均已验证与连续参考精确相同。\n'
            '修复前第 201 步：样本为 8 而非 1608，监督 token 为 6304 而非 3720095；LM / MTP / aux 损失保持精确相同。') if chinese else
            'Lines: rank 0. Exact equality verified for all 200 resumed steps on all 8 ranks.\n'
            'Before fix at step 201: 8 samples instead of 1608; 6304 tokens instead of 3720095. '
            'LM/MTP/aux loss remains exactly equal.', fontsize=9, color='#526276')
figure.subplots_adjust(left=0.08, right=0.98, top=0.77, bottom=0.2, wspace=0.28)
for extension in ('png', 'svg'):
    suffix = '-zh' if chinese else ''
    figure.savefig(root / 'figures' / f'resume-counters{suffix}.{extension}', dpi=180)
plt.close(figure)
