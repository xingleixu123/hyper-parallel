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

"""只用随附CSV重绘三臂主LM图，不需要原任务目录或任何训练依赖。"""

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
INK, GRAY, BLUE, AMBER, GREEN = '#1e2b38', '#657583', '#235c88', '#c8792b', '#278467'
LM = 'training/foundation_loss/lm'


def require(condition: bool, message: str) -> None:
    """拒绝不完整或不一致的便携图数据。"""
    if not condition:
        raise ValueError(message)


def load(data: Path) -> list[dict]:
    """核两份CSV的SHA、72行覆盖及全部八rank最大差。"""
    summary = json.loads((data / 'summary.json').read_bytes())
    tables = {}
    for name, count in (('metrics-72rows.csv', 72), ('main-lm-3steps.csv', 3)):
        raw = (data / name).read_bytes()
        require(hashlib.sha256(raw).hexdigest() == summary['csv_sha256'][name], 'CSV的SHA不同：' + name)
        rows = list(csv.DictReader(io.StringIO(raw.decode('utf-8'))))
        require(len(rows) == count, 'CSV行数不同：' + name)
        tables[name] = rows
    metrics = tables['metrics-72rows.csv']
    observed = {(row['arm'], int(row['rank']), int(row['step'])): row for row in metrics}
    expected = {(arm, rank, step) for arm in ('F', 'B', 'S') for rank in range(8) for step in (1, 2, 3)}
    require(len(observed) == 72 and set(observed) == expected, '三臂完整72行覆盖不同')
    steps = tables['main-lm-3steps.csv']
    require([int(row['step']) for row in steps] == [1, 2, 3], '步骤范围不同')
    for row in steps:
        step = int(row['step'])
        for arm in ('F', 'B', 'S'):
            values = [float(observed[arm, rank, step][LM]) for rank in range(8)]
            require(all(math.isfinite(value) for value in values) and len(set(values)) == 1,
                    '同臂归约主LM不一致或非有限')
            require(float(row[arm + '_main_LM']) == values[0], '主LM轨迹不是原CSV值')
        for arm in ('B', 'S'):
            maximum = max(abs(float(observed[arm, rank, step][LM]) - float(observed['F', rank, step][LM]))
                          for rank in range(8))
            require(maximum == float(row[arm + '_vs_F_all8rank_max_absdiff']), '主LM八rank最大差不同')
    return steps


def style() -> None:
    """使用可用中文字体，SVG保留可移植字形轮廓。"""
    font_path = Path('C:/Windows/Fonts/msyh.ttc')
    if font_path.is_file():
        font_manager.fontManager.addfont(str(font_path))
    names = {item.name for item in font_manager.fontManager.ttflist}
    font = next((name for name in ('Microsoft YaHei', 'Noto Sans CJK SC', 'Source Han Sans SC', 'SimHei')
                 if name in names), None)
    require(font is not None, '需要可用中文字体')
    matplotlib.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': [font, 'DejaVu Sans'], 'font.size': 11,
        'axes.unicode_minus': False, 'text.color': INK, 'axes.labelcolor': INK,
        'axes.edgecolor': '#b8c5d0', 'xtick.color': INK, 'ytick.color': GRAY,
        'axes.spines.right': False, 'axes.spines.top': False, 'figure.facecolor': 'white',
        'svg.fonttype': 'path', 'svg.hashsalt': 'selected-ns-unbatch-lm-v1',
    })


def plot(rows: list[dict], output: Path) -> None:
    """以线性零基线呈现第二步归零与第三步增大的完整结果。"""
    output.mkdir(parents=True, exist_ok=True)
    png, svg = output / 'main-lm-three-step.png', output / 'main-lm-three-step.svg'
    require(not png.exists() and not svg.exists(), '保留已有图，请使用新输出目录')
    style()
    figure = plt.figure(figsize=(12.8, 8.6))
    figure.text(.075, .955, '首步两组NS拆批：第2步主LM差为0，第3步差异增大', fontsize=19, weight='bold')
    figure.text(.075, .919, '544526f7 · 同129主机 · 每臂3步×8 ranks · 单次运行 · 定位消融', color=GRAY, fontsize=11)
    grid = figure.add_gridspec(2, 1, left=.09, right=.96, bottom=.19, top=.865, hspace=.38,
                              height_ratios=(1, 1.2))
    top, bottom = figure.add_subplot(grid[0]), figure.add_subplot(grid[1])
    steps = [int(row['step']) for row in rows]
    labels = {'F': 'F 纯FSDP参考', 'B': 'B 专家梯度＋norm对齐', 'S': 'S 在B基础上首步两组NS拆批'}
    colors = {'F': BLUE, 'B': AMBER, 'S': GREEN}
    for arm, marker in (('F', 'o'), ('B', 's'), ('S', '^')):
        top.plot(steps, [float(row[arm + '_main_LM']) for row in rows], color=colors[arm], marker=marker,
                 linewidth=2, markersize=6, label=labels[arm], linestyle='-' if arm == 'F' else '--')
    top.set_ylabel('主LM（日志值）')
    top.set_xticks(steps, ['第1步', '第2步', '第3步'])
    top.grid(axis='y', color='#e6edf2', linewidth=.8)
    top.legend(loc='lower left', frameon=False, fontsize=9.5)
    top.text(.01, .65, '上图近乎重合；精确差异见下图与CSV', color=GRAY, transform=top.transAxes, fontsize=10)
    width = .28
    for arm, shift in (('B', -width / 2), ('S', width / 2)):
        values = [float(row[arm + '_vs_F_all8rank_max_absdiff']) for row in rows]
        bottom.bar([step + shift for step in steps], values, width, color=colors[arm],
                   label=arm + '/F：每步8rank maxabs')
        for step, value in zip(steps, values):
            bottom.annotate('0' if value == 0 else f'{value:.7g}', (step + shift, value),
                            xytext=(0, 7), textcoords='offset points', ha='center', color=colors[arm],
                            fontsize=10, weight='bold')
    bottom.set_ylim(0, max(float(row['S_vs_F_all8rank_max_absdiff']) for row in rows) * 1.23)
    bottom.set_ylabel('主LM绝对差：maxabs（相对F）')
    bottom.set_xticks(steps, ['第1步', '第2步', '第3步'])
    bottom.ticklabel_format(axis='y', style='sci', scilimits=(-4, -4), useMathText=True)
    bottom.set_axisbelow(True)
    bottom.grid(axis='y', color='#e6edf2', linewidth=.8)
    bottom.legend(loc='upper left', frameon=False, fontsize=10)
    figure.text(.075, .123, 'S仅在第1步将两组共4个成员逐成员执行原NS；B/S均保留专家梯度＋norm供体消融。',
                color=GRAY, fontsize=10.5)
    figure.text(.075, .088, '8 ranks为同一运行的归约标量，非8次独立重复；零值按真实0绘制。', color=GRAY, fontsize=10.5)
    figure.text(.075, .053, '仅3步功能／精度定位；不是产品修复或200步验收，不比较性能。', color=GRAY, fontsize=10.5)
    figure.savefig(png, dpi=180, metadata={'Software': 'Selected NS unbatch metrics candidate v1'})
    figure.savefig(svg, metadata={'Date': None, 'Creator': 'Selected NS unbatch metrics candidate v1'})
    svg.write_text('\n'.join(line.rstrip(' \t') for line in svg.read_text(encoding='utf-8').splitlines()) + '\n',
                   encoding='utf-8', newline='\n')
    plt.close(figure)


def main() -> None:
    """由小型CSV排他创建图形，支持脱离原任务目录重绘。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=REPORT / 'data')
    parser.add_argument('--output-dir', type=Path, default=REPORT / 'figures')
    args = parser.parse_args()
    plot(load(args.data_dir), args.output_dir)
    print(json.dumps({'status': 'NOT_PUBLISHED_FIGURE_FROM_72_BOUND_METRIC_ROWS', 'figures': 1}, ensure_ascii=False))


if __name__ == '__main__':
    main()
