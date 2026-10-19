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

"""Redraw the final expert G0 chart using only the32 public matrix statistics."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib
from matplotlib import font_manager
from matplotlib import pyplot as plt


ALIASES = ('decoder1-down', 'decoder1-gate_up', 'mtp0-down', 'mtp0-gate_up')
DEFAULT_DATA = Path(__file__).resolve().parents[1] / 'data/matrices.json'


def require(condition: bool, message: str) -> None:
    """Reject changed input scope or an existing output instead of overwriting it."""
    if not condition:
        raise ValueError(message)


def main() -> None:
    """Draw only recorded first-step errors; no tensor, checkpoint, framework or network access."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--font', type=Path, default=Path('C:/Windows/Fonts/msyh.ttc'))
    args = parser.parse_args()
    require(args.data.is_file() and not args.data.is_symlink() and args.data.stat().st_size <= 1024 ** 2,
            'Bounded public statistics JSON required')
    raw = args.data.read_bytes()
    data = json.loads(raw)
    require(data['schema'] == 'public-native-expert-g0-matrix-statistics-v1', 'Unexpected data schema')
    members = {}
    for row in data['matrices']:
        key = row['global_expert'], row['alias']
        require(key not in members and row['stage'] == 'G0' and row['step'] == 1, 'Duplicate or foreign matrix')
        statistics = row['statistics']
        require(statistics['numeric_unequal_count'] > 0 and statistics['signed_zero_only_count'] == 0
                and statistics['numeric_threshold_applied'] is False, 'Numeric observation scope changed')
        require(all(math.isfinite(statistics[field]) and statistics[field] > 0
                    for field in ('max_abs', 'RMS', 'relative_L2_to_F')), 'Nonpositive log-axis value')
        members[key] = statistics
    require(set(members) == {(expert, alias) for expert in range(8) for alias in ALIASES},
            'All32 expert/matrix coordinates required')
    require(args.font.is_file(), 'Supply --font with a Chinese-capable font; the original used Microsoft YaHei')
    matplotlib.use('Agg')
    font_manager.fontManager.addfont(str(args.font))
    font_name = font_manager.FontProperties(fname=str(args.font)).get_name()
    plt.rcParams.update({'font.family': font_name, 'font.size': 10, 'axes.unicode_minus': False,
                         'axes.spines.top': False, 'axes.spines.right': False, 'savefig.facecolor': 'white'})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    png = args.output_dir / 'native-expert-g0-statistics.png'
    svg = args.output_dir / 'native-expert-g0-statistics.svg'
    provenance = args.output_dir / 'plot-reproduction.json'
    require(not any(path.exists() for path in (png, svg, provenance)), 'Plot exports must be fresh')
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    colors = ('#2563eb', '#7c3aed', '#dc2626', '#059669')
    labels = ('Decoder 1 / down', 'Decoder 1 / gate_up', 'MTP 0 / down', 'MTP 0 / gate_up')
    for axis, field, title in zip(axes, ('max_abs', 'relative_L2_to_F'),
                                  ('逐矩阵最大绝对差', '逐矩阵相对 L2 差（参考 FSDP）')):
        for alias, color, label in zip(ALIASES, colors, labels):
            values = [members[expert, alias][field] for expert in range(8)]
            axis.plot(range(8), values, marker='o', markersize=4, linewidth=1.3, color=color, label=label)
        axis.set_yscale('log')
        axis.set_xticks(range(8))
        axis.set_xlabel('全局 expert 编号')
        axis.set_title(title, loc='left', fontsize=12, pad=10)
        axis.grid(axis='y', alpha=0.2, which='both')
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, legend_labels, fontsize=9, frameon=False, ncol=4,
                  loc='upper left', bbox_to_anchor=(0.06, 0.86))
    figure.suptitle('首步 expert G0：EP4 对纯 FSDP8（32 个矩阵）', fontsize=16, x=0.06, ha='left', y=0.98)
    figure.text(0.06, 0.89, '同源 59b25cdd；FP32 NS、关闭跨参数拼批；RS 后 / 归一化前 / 裁剪前',
                color='#475569', fontsize=10)
    figure.text(0.06, 0.04, '真实 3 步诊断，梯度未改写。这里只量化梯度差，不能据此归因 RS/NS，不能替代 200 步 loss 验收。',
                color='#475569', fontsize=10)
    figure.subplots_adjust(left=0.075, right=0.98, bottom=0.18, top=0.72, wspace=0.23)
    figure.savefig(png, dpi=180)
    figure.savefig(svg)
    plt.close(figure)
    exports = []
    for path in (png, svg):
        value = path.read_bytes()
        exports.append({'path': path.name, 'bytes': len(value), 'sha256': hashlib.sha256(value).hexdigest()})
    record = {'schema': 'public-native-expert-g0-plot-reproduction-v1',
              'data_sha256': hashlib.sha256(raw).hexdigest(), 'exports': exports,
              'matrices': 32, 'gradient_threshold_invented': False, 'full200_loss_PASS_claimed': False,
              'scope': 'Redraw public recorded statistics only; no original arrays or training re-execution.'}
    with provenance.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(record, ensure_ascii=False))


if __name__ == '__main__':
    main()
