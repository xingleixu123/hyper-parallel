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

"""Render two portable Chinese precision figures from checksum-bound CSV; no private evidence is needed."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
from matplotlib import font_manager, pyplot as plt  # noqa: E402


REPORT = Path(__file__).resolve().parents[1]
BLUE, ORANGE, RED, INK, GRAY = '#235c88', '#bf762b', '#b8443c', '#1e2b38', '#657583'


def style() -> None:
    """Use an available Chinese font and embed deterministic SVG glyph outlines."""
    font_path = Path('C:/Windows/Fonts/msyh.ttc')
    if font_path.is_file():
        font_manager.fontManager.addfont(str(font_path))
    installed = {item.name for item in font_manager.fontManager.ttflist}
    font = next((name for name in ('Microsoft YaHei', 'Noto Sans CJK SC', 'Source Han Sans SC', 'SimHei')
                 if name in installed), None)
    if font is None:
        raise ValueError('需要可用中文字体：微软雅黑或 Noto Sans CJK SC')
    matplotlib.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': [font, 'DejaVu Sans'],
                               'font.size': 11, 'text.color': INK, 'axes.labelcolor': INK,
                               'axes.edgecolor': '#b8c5d0', 'xtick.color': INK, 'ytick.color': GRAY,
                               'axes.spines.right': False, 'axes.spines.top': False, 'axes.unicode_minus': False,
                               'figure.facecolor': 'white', 'svg.fonttype': 'path',
                               'svg.hashsalt': 'muon-no-cross-param-mainlm200-v1'})


def read(data: Path, summary: Path) -> tuple[list[dict], dict]:
    """Validate portable data integrity and the exact full200step finite axes."""
    metadata = json.loads(summary.read_bytes())
    raw = data.read_bytes()
    if hashlib.sha256(raw).hexdigest() != metadata['csv']['curves.csv']['sha256']:
        raise ValueError('CSV SHA不匹配')
    rows = [{name: float(value) for name, value in row.items()}
            for row in csv.DictReader(raw.decode('utf-8').splitlines())]
    if len(rows) != 200 or [row['step'] for row in rows] != list(range(1, 201)):
        raise ValueError('必须有完整200步')
    if any(not math.isfinite(value) for row in rows for value in row.values()):
        raise ValueError('图表数值必须有限')
    for label in ('old_pair', 'new_pair', 'cross_source_f'):
        values = [row[label] for row in rows]
        if min(values) < 0 or max(values) != metadata['relations'][label]['max_abs']:
            raise ValueError('误差数组与摘要不匹配')
    if any(row['tolerance_absolute'] != .005 or row['new_pair'] < abs(row['new_f'] - row['new_ep']) for row in rows):
        raise ValueError('阈值或all-rank差值不匹配')
    return rows, metadata


def axes_common(axis) -> None:
    """Keep all panels at the actual unfiltered200step window."""
    axis.set_xlim(1, 200)
    axis.set_xticks([1, 25, 50, 75, 100, 125, 150, 175, 200])
    axis.set_axisbelow(True)
    axis.yaxis.grid(True, color='#e5ebf0', linewidth=.8)


def save(figure, directory: Path, name: str) -> None:
    """Create new PNG/SVG outputs without replacing old evidence."""
    directory.mkdir(parents=True, exist_ok=True)
    png, svg = directory / (name + '.png'), directory / (name + '.svg')
    if png.exists() or svg.exists():
        plt.close(figure)
        raise ValueError('已有图像不覆盖')
    figure.savefig(png, dpi=180, metadata={'Software': 'Muon main-LM200 evidence v1'})
    figure.savefig(svg, metadata={'Date': None, 'Creator': 'Muon main-LM200 evidence v1'})
    svg.write_text('\n'.join(line.rstrip(' \t') for line in svg.read_text(encoding='utf-8').splitlines()) + '\n',
                   encoding='utf-8', newline='\n')
    plt.close(figure)


def plot(rows: list[dict], metadata: dict, output: Path) -> None:
    """Separate the new same-source gate from historical and cross-source observations."""
    style()
    steps = [row['step'] for row in rows]
    relation = metadata['relations']['new_pair']
    figure, (top, bottom) = plt.subplots(2, 1, figsize=(12, 8.6), gridspec_kw={'height_ratios': [1.2, 1]})
    figure.subplots_adjust(left=.10, right=.97, top=.83, bottom=.18, hspace=.45)
    figure.text(.065, .95, '新同源200步：功能完成，主 LM 仍超过0.005', fontsize=19, weight='bold')
    figure.text(.065, .9, '59b25cdd · 显式 ns_batch_across_params=False · 两臂各1次200步 × 8 ranks',
                fontsize=11, color=GRAY)
    top.plot(steps, [row['new_f'] for row in rows], color=BLUE, linewidth=1.9, label='纯 FSDP8 控制')
    top.plot(steps, [row['new_ep'] for row in rows], color=ORANGE, linewidth=1.2,
             linestyle=(0, (5, 3)), label='EP4')
    top.set_title('(a) rank0 主 LM 原始轨迹；未平滑', loc='left', fontsize=11, pad=12)
    top.set_ylabel('主 LM loss（未加权）')
    top.legend(frameon=False, loc='upper right')
    values = [row['new_pair'] for row in rows]
    bottom.plot(steps, values, color=BLUE, linewidth=1.6, label='每步8rank最大绝对差')
    exceeded = relation['exceeded_steps']
    bottom.scatter(exceeded, [values[step - 1] for step in exceeded], s=24, color=RED, zorder=4,
                   label='超过阈值的8步')
    bottom.axhline(.005, color=RED, linestyle='--', linewidth=1.4, label='主 LM 绝对阈值0.005')
    bottom.set_title('(b) 完整1600对记录；首超step97，最大差0.0103092194（step146）',
                     loc='left', fontsize=11, pad=12)
    bottom.set_xlabel('训练步')
    bottom.set_ylabel('每步最大 |LM_EP − LM_F|')
    bottom.set_ylim(-.00035, .0122)
    bottom.set_yticks([0, .005, .01], ['0', '0.005', '0.010'])
    bottom.legend(frameon=False, loc='upper left', fontsize=9.5, ncol=2)
    for axis in (top, bottom):
        axes_common(axis)
    figure.text(.065, .09, '仅主LM应用0.005阈值；其他loss与grad norm只观察。0值按真实线性坐标显示。',
                fontsize=10.2, color=GRAY)
    figure.text(.065, .045, '8rank是归约后同值全局标量，不是8次独立重复；不作性能、唯一根因或完整矩阵通过结论。',
                fontsize=10.2, color=GRAY)
    save(figure, output, 'new59b-main-lm200')

    figure, axes = plt.subplots(3, 1, figsize=(12, 10.4))
    figure.subplots_adjust(left=.10, right=.97, top=.84, bottom=.14, hspace=.63)
    figure.text(.065, .954, '三个关系分开观察：同源验收与跨源码差异不能混为一项', fontsize=18, weight='bold')
    figure.text(.065, .911, '每条曲线都由200步 × 8rank原始主LM计算；没有性能或归因百分比', fontsize=11, color=GRAY)
    panels = [('old_pair', '(a) 旧544526f7：既有 FSDP8 / EP4（历史主LM观察）', GRAY),
              ('new_pair', '(b) 新59b25cdd：FSDP8 / EP4（本次同源主LM验收）', BLUE),
              ('cross_source_f', '(c) 旧544 FSDP8 / 新59b FSDP8（跨源码与政策观察）', ORANGE)]
    for axis, (label, title, color) in zip(axes, panels):
        values = [row[label] for row in rows]
        record = metadata['relations'][label]
        axis.plot(steps, values, color=color, linewidth=1.5)
        axis.axhline(.005, color=RED, linestyle='--', linewidth=1.2)
        axis.set_title(title, loc='left', fontsize=11, pad=10)
        axis.set_ylabel('8rank最大绝对差')
        axis.set_ylim(-max(values) * .04, max(max(values) * 1.35, .007))
        axis.text(.015, .87, 'max %.10f @%d；超0.005：%d/200步' %
                  (record['max_abs'], record['max_step'], record['exceeded_step_count']),
                  transform=axis.transAxes, fontsize=10, color=color)
        axes_common(axis)
    axes[-1].set_xlabel('训练步')
    figure.text(.065, .075, '各面板纵轴按本身数据范围标注；红虚线均为0.005。图(c)不是同源精度判决或独立故障。',
                fontsize=10.2, color=GRAY)
    figure.text(.065, .035, '新实现选项的功能验证与本次精度FAIL同时成立；不据单次曲线确认唯一NS、batching或kernel根因。',
                fontsize=10.2, color=GRAY)
    save(figure, output, 'three-relations-main-lm200')


def main() -> None:
    """Replot offline from the portable CSV and summary only."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=REPORT / 'data/curves.csv')
    parser.add_argument('--summary', type=Path, default=REPORT / 'data/summary.json')
    parser.add_argument('--output-dir', type=Path, default=REPORT / 'figures')
    args = parser.parse_args()
    rows, metadata = read(args.data, args.summary)
    plot(rows, metadata, args.output_dir)
    print(json.dumps({'status': 'PLOTTED_PORTABLE_MAIN_LM_ONLY', 'figures': 2, 'steps': len(rows)}))


if __name__ == '__main__':
    main()
