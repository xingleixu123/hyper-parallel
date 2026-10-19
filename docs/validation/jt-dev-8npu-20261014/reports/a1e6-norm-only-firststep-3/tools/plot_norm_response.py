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

"""仅用便携CSV和摘要重绘中文首步范数干预诊断图。"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
from matplotlib import font_manager, pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402


REPORT = Path(__file__).resolve().parents[1]
INK, GRAY, BLUE, ORANGE, GREEN = '#1e2b38', '#657583', '#235c88', '#bf762b', '#d4e9dd'


def style() -> None:
    """选择实际中文字体，SVG保留确定性轮廓。"""
    font_path = Path('C:/Windows/Fonts/msyh.ttc')
    if font_path.is_file():
        font_manager.fontManager.addfont(str(font_path))
    installed = {item.name for item in font_manager.fontManager.ttflist}
    font = next((name for name in ('Microsoft YaHei', 'Noto Sans CJK SC', 'Source Han Sans SC', 'SimHei')
                 if name in installed), None)
    if font is None:
        raise ValueError('需要可用中文字体：微软雅黑或Noto Sans CJK SC')
    matplotlib.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': [font, 'DejaVu Sans'],
                               'font.size': 11, 'text.color': INK, 'axes.labelcolor': INK,
                               'axes.edgecolor': '#b8c5d0', 'xtick.color': INK, 'ytick.color': GRAY,
                               'axes.spines.top': False, 'axes.spines.right': False,
                               'axes.unicode_minus': False, 'figure.facecolor': 'white',
                               'svg.fonttype': 'path', 'svg.hashsalt': 'norm-only-response-N2-v1'})


def read(directory: Path) -> tuple[dict, list[dict], list[dict]]:
    """核全部CSV SHA，并从72行数据重新计算三步全rank最大差。"""
    summary = json.loads((directory / 'summary.json').read_bytes())
    tables = {}
    for name, descriptor in summary['csv'].items():
        raw = (directory / name).read_bytes()
        if len(raw) != descriptor['bytes'] or hashlib.sha256(raw).hexdigest() != descriptor['sha256']:
            raise ValueError('便携CSV字节或SHA不符')
        tables[name] = list(csv.DictReader(raw.decode().splitlines()))
    records = tables['main-lm-rank-step.csv']
    values = {(row['arm'], int(row['rank']), int(row['step'])): float(row['main_lm']) for row in records}
    if len(records) != 72 or len(values) != 72 or any(not math.isfinite(x) for x in values.values()):
        raise ValueError('必须包含三臂完整3×8有限主LM记录')
    for row in summary['curves']:
        step = row['step']
        for arm in ('F', 'EP', 'N'):
            if {values[(arm, rank, step)] for rank in range(8)} != {row[arm]}:
                raise ValueError('原72行不支持每臂每步rank共识')
        for arm, key in (('EP', 'EP_vs_F_maxabs'), ('N', 'N_vs_F_maxabs')):
            maximum = max(abs(values[(arm, rank, step)] - values[('F', rank, step)]) for rank in range(8))
            if maximum != row[key]:
                raise ValueError('图上差值与全rank记录不一致')
    stages = tables['stage-equality.csv']
    if len(stages) != 16:
        raise ValueError('两种关系×四阶段×两类参数必须完整')
    for row, expected in zip(stages, summary['stage_counts']):
        cast = {key: (int(value) if key in ('equal', 'total', 'equal_fqns', 'total_fqns') else value)
                for key, value in row.items()}
        if cast != expected:
            raise ValueError('阶段CSV与摘要不符')
    return summary, stages, summary['curves']


def plot(summary: dict, stages: list[dict], curves: list[dict], output: Path) -> None:
    """编码计数与LM误差分面；真0使用线性轴，不赋予三步验收含义。"""
    style()
    figure, (top, bottom) = plt.subplots(2, 1, figsize=(12.8, 9.3),
                                       gridspec_kw={'height_ratios': [.85, 1.25]})
    figure.subplots_adjust(left=.17, right=.95, top=.79, bottom=.19, hspace=.75)
    figure.text(.055, .957, '裁剪系数对齐后，非专家首步更新编码对齐；主LM仍有后续残差',
                fontsize=17, weight='bold')
    figure.text(.055, .911, '59b25cdd · 首步范数单变量干预 · 实际记录配方：FP32 NS，不跨参数批处理',
                fontsize=11, color=GRAY)
    figure.text(.055, .868, 'N保留实测范数9fc9b341，仅裁剪使用9ec9b341；系数7042363d与F相同（FP32原4字节）',
                fontsize=10.5, color=GRAY)
    phases = ('P0', 'G0', 'G1', 'P1')
    selected = {(row['group'], row['phase']): row for row in stages if row['reference'] == 'F'}
    groups = ('nonexpert', 'expert')
    matrix = [[int(selected[(group, phase)]['equal']) == int(selected[(group, phase)]['total'])
               for phase in phases] for group in groups]
    top.imshow(matrix, aspect='auto', cmap=ListedColormap(['#edf0f2', GREEN]), vmin=0, vmax=1)
    for row, group in enumerate(groups):
        for col, phase in enumerate(phases):
            record = selected[(group, phase)]
            top.text(col, row, '%s/%s' % (record['equal'], record['total']), ha='center', va='center',
                     fontsize=16, weight='bold')
    top.set_xticks(range(4), ['P0 初始参数', 'G0 裁剪前梯度', 'G1 裁剪后梯度', 'P1 首步更新后'])
    top.set_yticks(range(2), ['非专家\n42 FQN × 8逻辑片', '专家\n4参数族 × 8专家'])
    top.tick_params(axis='both', length=0, pad=10)
    top.set_title('(a) N 相对 F：完整逻辑片原生SHA相同数 / 总数', loc='left', fontsize=11, pad=15)
    top.legend(handles=[Patch(facecolor=GREEN, label='此单元全部编码相同'),
                        Patch(facecolor='#edf0f2', label='存在编码不同；不表示数值误差幅度')],
               frameon=False, loc='upper center', bbox_to_anchor=(.5, -.35), ncol=2, fontsize=9)
    steps = [row['step'] for row in curves]
    bottom.plot(steps, [row['EP_vs_F_maxabs'] for row in curves], color=ORANGE, marker='s',
                linewidth=1.8, markersize=6, label='原 EP 对 F')
    bottom.plot(steps, [row['N_vs_F_maxabs'] for row in curves], color=BLUE, marker='o',
                linewidth=2, markersize=6, label='N 范数干预对 F')
    bottom.set_xlim(.85, 3.25)
    bottom.set_ylim(-.000009, .000117)
    bottom.set_xticks([1, 2, 3])
    bottom.set_yticks([0, .00002, .00004, .00006, .00008, .0001], ['0', '2e−5', '4e−5', '6e−5', '8e−5', '1e−4'])
    bottom.yaxis.grid(True, color='#e5ebf0')
    bottom.set_axisbelow(True)
    bottom.set_xlabel('训练步')
    bottom.set_ylabel('每步8rank最大 |主LM − F|')
    bottom.set_title('(b) 三步主LM响应：第2步N对齐，第3步差异反增', loc='left', fontsize=11, pad=14)
    bottom.legend(frameon=False, loc='upper left', fontsize=10)
    bottom.annotate('两臂均0', (1, 0), xytext=(10, 12), textcoords='offset points', fontsize=10, color=GRAY)
    bottom.annotate('0', (2, 0), xytext=(-4, -17), textcoords='offset points', fontsize=10, color=BLUE)
    bottom.annotate('1.1444092e−5', (2, curves[1]['EP_vs_F_maxabs']), xytext=(-65, 12),
                    textcoords='offset points', fontsize=10, color=ORANGE)
    bottom.annotate('3.3378601e−5', (3, curves[2]['EP_vs_F_maxabs']), xytext=(-99, 12),
                    textcoords='offset points', fontsize=10, color=ORANGE)
    bottom.annotate('9.6321106e−5', (3, curves[2]['N_vs_F_maxabs']), xytext=(-103, 12),
                    textcoords='offset points', fontsize=10, color=BLUE)
    figure.text(.055, .11, '每臂单次3步 × 8 ranks；归约后同值标量不是8次独立重复。0按真实线性坐标显示。',
                fontsize=10.5, color=GRAY)
    figure.text(.055, .069, '仅首步替换used norm，无专家梯度回写；这是因果传播定位，不是200步修复或唯一根因结论。',
                fontsize=10.5, color=GRAY)
    figure.text(.055, .028, '既有59b完整200步主LM判决仍FAIL：max 0.010309219360351562（step146）。本图不作性能分析。',
                fontsize=10.2, color=GRAY)
    output.mkdir(parents=True, exist_ok=True)
    png, svg = output / 'norm-only-firststep-response.png', output / 'norm-only-firststep-response.svg'
    if png.exists() or svg.exists():
        raise ValueError('已有图像不能覆盖')
    figure.savefig(png, dpi=180, metadata={'Software': 'Norm-only diagnostic N2 v1'})
    figure.savefig(svg, metadata={'Date': None, 'Creator': 'Norm-only diagnostic N2 v1'})
    svg.write_text('\n'.join(line.rstrip(' \t') for line in svg.read_text(encoding='utf-8').splitlines()) + '\n',
                   encoding='utf-8', newline='\n')
    plt.close(figure)


def main() -> None:
    """允许脱离私有原目录，仅用本报告data重绘到全新目录。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=REPORT / 'data')
    parser.add_argument('--output-dir', type=Path, default=REPORT / 'figures')
    args = parser.parse_args()
    summary, stages, curves = read(args.data_dir)
    plot(summary, stages, curves, args.output_dir)
    print(json.dumps({'status': 'PRIVATE_DIAGNOSTIC_FIGURE_RENDERED', 'figures': 1, 'metric_rows': 72}))


if __name__ == '__main__':
    main()
