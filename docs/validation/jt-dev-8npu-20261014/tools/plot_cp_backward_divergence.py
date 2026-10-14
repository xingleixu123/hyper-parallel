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
"""Plot sampled first-step CP gradient boundaries, with explicit coverage limits."""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
data = json.loads((ROOT / 'reports/cp-first-backward-comparison.json').read_text())
selected = [
    ('mtp.layers.0.transformer_layer', 'MTP decoder output'),
    ('mtp.layers.0.transformer_layer.post_attention_layernorm', 'MTP post-attention norm'),
    ('mtp.layers.0.transformer_layer.self_attn.kv_a_layernorm', 'MTP KV latent norm'),
    ('model.norm', 'Trunk final norm'),
    ('model.layers.1.post_attention_layernorm', 'Layer 1 post-attention norm'),
    ('model.layers.1.input_layernorm', 'Layer 1 input norm'),
    ('model.layers.0.post_attention_layernorm', 'Layer 0 post-attention norm'),
    ('model.layers.0.input_layernorm', 'Layer 0 input norm'),
    ('model.embed_tokens', 'Token embedding output'),
]
index = {(r['module'], r['call'], r['stage']): r for r in data['records']}
rows = [index[(name, 0, 'output_gradient')] for name, _ in selected]
assert all(r['missing_coordinates'] == 0 for r in data['records'])
assert all(r['max_abs'] == 0 for r in data['records'] if r['stage'] in ('input', 'output'))
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'svg.fonttype': 'none'})
fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.7), sharey=True, gridspec_kw={'width_ratios': [1.35, 1]})
y = np.arange(len(rows))
axes[0].barh(y, [r['relative_l2'] for r in rows], color='#196B8A', height=.62)
axes[0].set_xscale('log')
axes[0].set_xlim(4e-8, 2e-2)
axes[0].set_yticks(y, [label for _, label in selected])
axes[0].invert_yaxis()
axes[0].set_xlabel('Sampled gradient relative L2 error (log scale)')
axes[0].set_title('Gradient difference by boundary', loc='left', fontsize=12)
axes[1].barh(y, [100*r['different_fraction'] for r in rows], color='#D77A31', height=.62)
axes[1].set_xlim(0, 86)
axes[1].set_xlabel('Sampled coordinates with nonzero error (%)')
axes[1].set_title('Breadth of the sampled difference', loc='left', fontsize=12)
for i, row in enumerate(rows):
    axes[1].text(100*row['different_fraction']+1.2, i, f"{100*row['different_fraction']:.3g}%", va='center')
for axis in axes:
    axis.grid(axis='x', alpha=.2)
    axis.set_axisbelow(True)
    axis.spines[['top', 'right']].set_visible(False)
fig.suptitle('CP2: sampled forward boundaries match, backward gradients diverge',
             x=.06, ha='left', y=.98, fontsize=15, weight='bold')
fig.text(.06, .922, 'Original BF16 recipe • FSDP8 vs CP2 • expanded Ulysses • first step, all 8 samples', fontsize=10.5)
fig.text(.06, .055, 'Non-mutating tensor hooks; sequence/feature stride 128. No missing sampled coordinates.\n'
         'This is a boundary sample, not a full-tensor proof or a 200-step acceptance result. Branches are not a single chain.',
         fontsize=9.5, color='#444444')
fig.subplots_adjust(left=.275, right=.97, top=.83, bottom=.17, wspace=.13)
for extension in ('png', 'svg'):
    fig.savefig(ROOT / f'figures/cp-first-backward-divergence.{extension}', dpi=170)
plt.close(fig)
print('Rendered CP backward PNG/SVG from verified sampled-coordinate evidence')
