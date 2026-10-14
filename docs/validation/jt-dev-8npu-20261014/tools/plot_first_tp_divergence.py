"""Plot exact projection input evidence separately from sampled output errors."""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
report = json.loads((ROOT / 'reports/first-step-tp-projection-activation-comparison.json').read_text())
digests = json.loads((ROOT / 'reports/first-step-tp-projection-digests.json').read_text())
records = {(row['module'], row['stage']): row for row in report['records'] if row['call'] == 0}
labels = ['Q projection output', 'KV projection output', 'Attention context / o_proj input',
          'Attention output after o_proj', 'Post-attention norm output', 'Dense MLP output']
keys = [('model.layers.0.self_attn.q_b_proj', 'output'),
        ('model.layers.0.self_attn.kv_b_proj', 'output'),
        ('model.layers.0.self_attn.o_proj', 'input'),
        ('model.layers.0.self_attn', 'output'),
        ('model.layers.0.post_attention_layernorm', 'output'),
        ('model.layers.0.mlp', 'output')]
values = [records[key]['max_abs'] for key in keys]
figure = plt.figure(figsize=(12, 6), facecolor='#f7f9fc')
axis = figure.add_axes([.37, .25, .56, .55], facecolor='#f7f9fc')
colors = ['#16856c'] * 3 + ['#d15c31', '#8091a8', '#8091a8']
axis.barh(range(6), values, color=colors, height=.55)
axis.set_yticks(range(6), labels, fontsize=10)
axis.invert_yaxis()
axis.set_xlim(0, .08)
axis.set_xlabel('Maximum absolute difference across matched sampled coordinates', fontsize=10)
axis.spines[['top', 'right', 'left']].set_visible(False)
axis.grid(axis='x', alpha=.2)
axis.set_axisbelow(True)
for index, value in enumerate(values):
    text = f'{value:.7f}'.rstrip('0').rstrip('.') if value else '0 — full tensor hashes also match'
    axis.text(value + .0012, index, text, va='center', fontsize=9, color=colors[index])
figure.text(.06, .91, 'First TP divergence: attention output projection', fontsize=19, weight='bold', color='#19334d')
figure.text(.06, .855, 'FSDP8 vs TP2 × EP2 · step 1 before optimizer update · first transformer layer', fontsize=11)
counts = [digests['modules'][f'{module}:{stage}']['blocks'] for module, stage in keys[:3]]
assert all(digests['modules'][f'{module}:{stage}']['different_blocks'] == 0 for module, stage in keys[:3])
figure.text(.06, .12, f'Exact evidence: all {sum(counts):,} input/projection blocks match across 8 samples, '
            'all sequence positions and features.', fontsize=10, color='#16856c')
figure.text(.06, .075, 'Nonzero bars are sampled maxima (10,240 coordinates each), not full-tensor error bounds. '
            'No 200-step acceptance claim.', fontsize=9, color='#536174')
for extension in ('png', 'svg'):
    figure.savefig(ROOT / 'figures' / f'tp-first-divergence.{extension}', dpi=160, facecolor=figure.get_facecolor())
plt.close(figure)
