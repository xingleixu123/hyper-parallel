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

"""Render one portable scientific figure from the200-row real-data CSV."""

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
from matplotlib import font_manager, pyplot as plt  # noqa: E402


REPORT = Path(__file__).resolve().parents[1]
BLUE = '#235c88'
ORANGE = '#c8792b'
RED = '#b8443c'
INK = '#1e2b38'
GRAY = '#657583'


def load_data(data: Path, summary: Path) -> list[dict]:
    """Verify the exact CSV checksum and its complete finite200-step shape."""
    metadata = json.loads(summary.read_bytes())
    raw = data.read_bytes()
    if hashlib.sha256(raw).hexdigest() != metadata['csv_sha256']:
        raise ValueError('CSV checksum differs from the source summary')
    rows = list(csv.DictReader(io.StringIO(raw.decode('utf-8'))))
    if len(rows) != 200 or [int(row['step']) for row in rows] != list(range(1, 201)):
        raise ValueError('Exactly200 consecutive data rows are required')
    expected = {'step', 'lm_129_rank0', 'lm_125_rank0', 'lm_8rank_max_abs', 'tolerance_absolute'}
    if any(set(row) != expected for row in rows):
        raise ValueError('CSV field contract differs')
    result = [{key: float(value) for key, value in row.items()} for row in rows]
    if any(not math.isfinite(value) for row in result for value in row.values()):
        raise ValueError('Nonfinite figure input')
    if any(row['tolerance_absolute'] != .005 or row['lm_8rank_max_abs'] <
           abs(row['lm_125_rank0'] - row['lm_129_rank0']) for row in result):
        raise ValueError('Invalid threshold or all-rank error bound')
    return result


def style() -> None:
    """Prefer available Chinese typography and embed SVG glyph outlines."""
    windows_font = Path('C:/Windows/Fonts/msyh.ttc')
    if windows_font.is_file():
        font_manager.fontManager.addfont(str(windows_font))
    installed = {item.name for item in font_manager.fontManager.ttflist}
    candidates = ('Microsoft YaHei', 'Noto Sans CJK SC', 'Source Han Sans SC', 'SimHei')
    font = next((name for name in candidates if name in installed), None)
    if font is None:
        raise ValueError('A Chinese font is required: Microsoft YaHei / Noto Sans CJK SC / Source Han Sans SC')
    matplotlib.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': [font, 'DejaVu Sans'], 'font.size': 11,
        'axes.unicode_minus': False, 'text.color': INK, 'axes.labelcolor': INK,
        'axes.edgecolor': '#b8c5d0', 'xtick.color': INK, 'ytick.color': GRAY,
        'axes.spines.right': False, 'axes.spines.top': False, 'figure.facecolor': 'white',
        'svg.fonttype': 'path', 'svg.hashsalt': 'a1e6-125-129-control200-v1',
    })


def plot(rows: list[dict], output: Path) -> None:
    """Show coincident raw trajectories and actual zero errors on a linear axis."""
    style()
    figure, (top, bottom) = plt.subplots(2, 1, figsize=(12, 8.4),
                                        gridspec_kw={'height_ratios': [1.3, 1]})
    figure.subplots_adjust(left=.105, right=.97, bottom=.18, top=.825, hspace=.48)
    figure.text(.07, .947, '125 与129：普通 FSDP8 控制的200步主 LM 逐值一致', fontsize=19, weight='bold')
    figure.text(.07, .897, '544526f7 · CANN 9.2b2 · 每机1次完整200步 × 8 ranks', fontsize=11, color=GRAY)
    steps = [row['step'] for row in rows]
    top.plot(steps, [row['lm_129_rank0'] for row in rows], color=BLUE, linewidth=2,
             label='129 既有控制（实线）')
    top.plot(steps, [row['lm_125_rank0'] for row in rows], color=ORANGE, linewidth=1.5,
             linestyle=(0, (5, 3)), label='125 新控制（虚线；逐点重合）')
    top.set_title('(a) 两机 rank0 主 LM 原始轨迹；未平滑', loc='left', fontsize=11, pad=12)
    top.set_ylabel('主 LM loss（未加权）', labelpad=12)
    top.legend(frameon=False, loc='upper right', fontsize=10)
    bottom.plot(steps, [row['lm_8rank_max_abs'] for row in rows], color=BLUE, linewidth=1.8,
                marker='o', markevery=20, markersize=4, label='每步8rank最大绝对差')
    bottom.axhline(.005, color=RED, linewidth=1.6, linestyle='--', label='绝对阈值0.005')
    bottom.set_title('(b) all-rank 最大绝对差；200步全部为0', loc='left', fontsize=11, pad=12)
    bottom.set_ylabel('最大 |LM125 − LM129|', labelpad=12)
    bottom.set_xlabel('训练步')
    bottom.set_ylim(-.0004, .0058)
    bottom.set_yticks([0, .0025, .005], ['0', '0.0025', '0.005'])
    bottom.legend(frameon=False, loc='center right', fontsize=10)
    bottom.text(.03, .28, 'max = 0；超阈值 0 / 200 步', transform=bottom.transAxes,
                color=BLUE, fontsize=11, weight='bold')
    for axis in (top, bottom):
        axis.set_xlim(1, 200)
        axis.set_xticks([1, 25, 50, 75, 100, 125, 150, 175, 200])
        axis.set_axisbelow(True)
        axis.yaxis.grid(True, color='#e5ebf0', linewidth=.8)
    figure.text(.07, .092, '8rank日志为归约后的同值全局标量，不是8次独立重复。下图为线性坐标，0未加epsilon。',
                fontsize=10.2, color=GRAY)
    figure.text(.07, .047, '仅报告这两次同源跨机控制；不作性能收益、其他并行矩阵通过或产品修复结论。',
                fontsize=10.2, color=GRAY)
    output.mkdir(parents=True, exist_ok=True)
    png, svg = output / 'control200-main-lm.png', output / 'control200-main-lm.svg'
    if png.exists() or svg.exists():
        plt.close(figure)
        raise ValueError('Existing figure files are preserved')
    figure.savefig(png, dpi=180, metadata={'Software': '125-129 control200 real-data report v1'})
    figure.savefig(svg, metadata={'Date': None, 'Creator': '125-129 control200 real-data report v1'})
    svg.write_text('\n'.join(line.rstrip(' \t') for line in svg.read_text(encoding='utf-8').splitlines()) + '\n',
                   encoding='utf-8', newline='\n')
    plt.close(figure)


def main() -> None:
    """Replot from portable data without reading private raw evidence."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=REPORT / 'data/control200.csv')
    parser.add_argument('--summary', type=Path, default=REPORT / 'data/summary.json')
    parser.add_argument('--output-dir', type=Path, default=REPORT / 'figures')
    args = parser.parse_args()
    rows = load_data(args.data, args.summary)
    plot(rows, args.output_dir)
    print(json.dumps({'status': 'PLOTTED_FROM_PORTABLE_REAL_DATA', 'data_rows': len(rows), 'figures': 1}))


if __name__ == '__main__':
    main()
