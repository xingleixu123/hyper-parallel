"""Plot sampled first-forward evidence, without treating it as a training gate."""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
reports = [
    ('BF16 local math + reduction', 'first-step-tp-projection-activation-comparison.json', '#bd4c34'),
    ('BF16 local math + FP32 reduction', 'rowreduce32-tp-vs-native-activation.json', '#d39837'),
    ('FP32 local math + reduction', 'row32-tp-vs-row32-fsdp-activation.json', '#147e76'),
]
keys = [('model.layers.0.self_attn', 'output'),
        ('model.layers.0.post_attention_layernorm', 'output'),
        ('model.layers.0.mlp', 'output')]
labels = ['Attention output', 'Post-attention norm', 'Dense MLP output']
figure, axes = plt.subplots(1, 2, figsize=(12.5, 6), facecolor='#f7f9fc')
for axis in axes:
    axis.set_facecolor('#f7f9fc')
    axis.spines[['top', 'right']].set_visible(False)
    axis.grid(axis='y', alpha=.2)
    axis.set_axisbelow(True)
width = .24
for index, (label, filename, color) in enumerate(reports):
    rows = json.loads((ROOT / 'reports' / filename).read_text())['records']
    selected = {(row['module'], row['stage']): row for row in rows if row['call'] == 0}
    for axis, metric in zip(axes, ['relative_l2', 'max_abs']):
        values = [selected[key][metric] for key in keys]
        bars = axis.bar(np.arange(3) + (index-1)*width, values, width, label=label, color=color)
        for bar, value in zip(bars, values):
            if index == 1:
                continue
            axis.text(bar.get_x()+bar.get_width()/2, value*1.12, f'{value:.3g}', ha='center',
                      va='bottom', fontsize=8, rotation=0)
for axis in axes:
    axis.set_xticks(range(3), labels, fontsize=9)
    axis.set_yscale('log')
axes[0].set_ylabel('Relative L2 error (log scale)')
axes[1].set_ylabel('Maximum absolute error (log scale)')
axes[0].set_ylim(2e-5, .08)
axes[1].set_ylim(4e-4, .2)
figure.suptitle('TP first-forward rounding: local projection precision matters', x=.07, ha='left',
                fontsize=16, fontweight='bold', y=.96)
figure.text(.07, .88, 'First layer; 10,240 sampled coordinates per boundary across 8 matched samples.', fontsize=10)
figure.legend(*axes[0].get_legend_handles_labels(), loc='lower left', bbox_to_anchor=(.06, .12),
              frameon=False, ncol=1, fontsize=9)
figure.text(.07, .065,
            'FP32 math uses a matching FP32 FSDP reference. Residual errors still grow in later modules.\n'
            'One forward only; native FP32 autograd also changes backward. No 200-step or performance claim.',
            fontsize=9, color='#42536a')
figure.subplots_adjust(left=.08, right=.97, top=.80, bottom=.34, wspace=.25)
for extension in ['png', 'svg']:
    figure.savefig(ROOT / f'figures/tp-row-precision-ablation.{extension}', dpi=180)
plt.close(figure)
