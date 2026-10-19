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

"""Plot the bounded five-arm first-update evidence from small portable CSVs."""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
from matplotlib import font_manager, pyplot as plt  # noqa: E402


REPORT = Path(__file__).resolve().parents[1]
ROLES = ('capture', 'self', 'norm', 'grad', 'both')
LABELS = ('F 参照自身', 'S 自重放', 'N 仅used norm', 'G 仅专家梯度', 'B 两者同时')
COLORS = ('#8b98a3', '#235c88', '#c8792b', '#8058a8', '#278467')
INK, GRAY = '#1e2b38', '#657583'


def load(data: Path) -> tuple[dict, list[dict], list[dict], list[dict]]:
    """Verify all three portable CSV hashes and their fixed complete shapes."""
    summary = json.loads((data / 'summary.json').read_bytes())
    result = []
    for name, size in (('losses.csv', 15), ('norms.csv', 5), ('parameter-hashes.csv', 160)):
        raw = (data / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != summary['csv_sha256']['data/' + name]:
            raise ValueError('CSV SHA differs: ' + name)
        rows = list(csv.DictReader(io.StringIO(raw.decode('utf-8'))))
        if len(rows) != size:
            raise ValueError('CSV row coverage differs: ' + name)
        result.append(rows)
    losses, norms, parameters = result
    if {(row['role'], int(row['step'])) for row in losses} != {(role, step) for role in ROLES for step in (1, 2, 3)}:
        raise ValueError('Five-arm three-step coverage differs')
    if [row['role'] for row in norms] != list(ROLES):
        raise ValueError('Norm role order differs')
    for row in parameters:
        if row['matches_F_hash'] not in ('True', 'False'):
            raise ValueError('Invalid hash equality marker')
    return summary, losses, norms, parameters


def style() -> None:
    """Use Chinese labels and portable SVG glyph outlines."""
    font_path = Path('C:/Windows/Fonts/msyh.ttc')
    if font_path.is_file():
        font_manager.fontManager.addfont(str(font_path))
    installed = {item.name for item in font_manager.fontManager.ttflist}
    font = next((name for name in ('Microsoft YaHei', 'Noto Sans CJK SC', 'Source Han Sans SC', 'SimHei')
                 if name in installed), None)
    if font is None:
        raise ValueError('A Chinese font is required for scientific labels')
    matplotlib.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': [font, 'DejaVu Sans'], 'font.size': 11,
        'axes.unicode_minus': False, 'text.color': INK, 'axes.labelcolor': INK,
        'axes.edgecolor': '#b8c5d0', 'xtick.color': INK, 'ytick.color': GRAY,
        'axes.spines.right': False, 'axes.spines.top': False, 'figure.facecolor': 'white',
        'svg.fonttype': 'path', 'svg.hashsalt': 'expert-replay-first-update-v1',
    })


def save(figure: plt.Figure, output: Path, name: str) -> None:
    """Export deterministic PNG/SVG without overwriting an existing figure."""
    output.mkdir(parents=True, exist_ok=True)
    png, svg = output / (name + '.png'), output / (name + '.svg')
    if png.exists() or svg.exists():
        plt.close(figure)
        raise ValueError('Existing figure files are preserved')
    figure.savefig(png, dpi=180, metadata={'Software': 'Expert replay first-update report v1'})
    figure.savefig(svg, metadata={'Date': None, 'Creator': 'Expert replay first-update report v1'})
    svg.write_text('\n'.join(line.rstrip(' \t') for line in svg.read_text(encoding='utf-8').splitlines()) + '\n',
                   encoding='utf-8', newline='\n')
    plt.close(figure)


def plot_loss(losses: list[dict], output: Path) -> None:
    """Show actual three-step all-rank residuals without a200-step acceptance line."""
    figure, axis = plt.subplots(figsize=(12, 7.4))
    figure.subplots_adjust(left=.105, right=.96, bottom=.20, top=.76)
    figure.text(.07, .943, '首步干预后，B 臂第2–3步主 LM 仍有残差', fontsize=20, weight='bold')
    figure.text(.07, .89, '129 · 544526f7 · 五臂各3步 × 8rank · 相对F的逐步8rank最大绝对差',
                fontsize=11, color=GRAY)
    markers = ('o', 's', '^', 'D', 'o')
    for role, label, color, marker in zip(ROLES, LABELS, COLORS, markers):
        rows = [row for row in losses if row['role'] == role]
        axis.plot([int(row['step']) for row in rows],
                  [float(row['main_lm_absdiff_vs_F_8rankmax']) / 1e-4 for row in rows],
                  color=color, marker=marker, linewidth=2, markersize=7, label=label,
                  linestyle=':' if role == 'capture' else '-',
                  markerfacecolor='white' if role == 'capture' else color)
    axis.set_xlim(.85, 3.15)
    axis.set_xticks([1, 2, 3])
    axis.set_ylim(-.18, 5.1)
    axis.set_yticks([0, 1, 2, 3, 4, 5])
    axis.set_xlabel('训练步')
    axis.set_ylabel('主 LM 最大绝对差（单位1e-4）', labelpad=12)
    axis.set_axisbelow(True)
    axis.yaxis.grid(True, color='#e5ebf0', linewidth=.8)
    handles, labels = axis.get_legend_handles_labels()
    figure.legend(handles, labels, loc='upper left', bbox_to_anchor=(.10, .842), frameon=False,
                  ncol=5, fontsize=10)
    for row in (row for row in losses if row['role'] == 'both' and int(row['step']) > 1):
        step, value = int(row['step']), float(row['main_lm_absdiff_vs_F_8rankmax'])
        axis.annotate(f'B: {value:.7g}', (step, value / 1e-4), xytext=(0, 18),
                      textcoords='offset points', ha='center', color=COLORS[-1], fontsize=10)
    axis.annotate('step1 五臂loss相同', (1, 0), xytext=(1.08, .75), fontsize=10, color=GRAY,
                  arrowprops={'arrowstyle': '-', 'color': '#a8b6c2'})
    figure.text(.07, .11, '干预仅在首步backward/RS后、clip前；第2步前已卸载自有包装。线性轴保留真实0。',
                fontsize=10.2, color=GRAY)
    figure.text(.07, .057, '每臂仅1次运行；8rank为分布式归约日志，不是8次独立重复。本图不作200步验收。',
                fontsize=10.2, color=GRAY)
    save(figure, output, 'main-lm-three-step')


def plot_hashes(summary: dict, norms: list[dict], parameters: list[dict], output: Path) -> None:
    """Separate whole-expert hash counts from the measured/used native scalar bytes."""
    figure = plt.figure(figsize=(12, 9.1))
    axis = figure.add_axes([.105, .55, .85, .24])
    table_axis = figure.add_axes([.075, .18, .88, .25])
    table_axis.axis('off')
    figure.text(.07, .948, 'B 臂首更新后：32个专家目标参数哈希全部匹配F', fontsize=19, weight='bold')
    figure.text(.07, .90, '4个专家族 × 8 global experts；整专家FP32原生SHA，不是参数数值误差',
                fontsize=11, color=GRAY)
    values = []
    for role in ROLES[1:]:
        rows = [row for row in parameters if row['role'] == role]
        if len(rows) != 32:
            raise ValueError('Missing whole-expert parameter target rows')
        value = sum(row['matches_F_hash'] == 'True' for row in rows)
        if value != summary['source_closure'][role]['stage_recorded_native_hash_matches_F']['parameter_after']:
            raise ValueError('Hash CSV count differs from summary')
        values.append(value)
    bars = axis.bar(range(4), values, width=.5, color=COLORS[1:])
    axis.set_xticks(range(4), ['S 自重放', 'N 仅used norm', 'G 仅专家梯度', 'B 两者同时'])
    axis.set_ylim(-1.1, 36.5)
    axis.set_yticks([0, 8, 16, 24, 32])
    axis.set_ylabel('parameter_after\n匹配F的整专家哈希数', labelpad=12)
    axis.set_axisbelow(True)
    axis.yaxis.grid(True, color='#e5ebf0', linewidth=.8)
    for bar, value in zip(bars, values):
        center = bar.get_x() + bar.get_width() / 2
        if value == 0:
            axis.scatter([center], [0], marker='D', color=bar.get_facecolor(), s=32, zorder=4)
        axis.text(center, value + 1, f'{value}/32', ha='center', weight='bold', fontsize=12)
    figure.text(.08, .463, '首步 norm：原4字节（小端hex）；每臂8rank各自一致', fontsize=11, weight='bold')
    cells = [[row['arm'], row['measured_native_hex_le'], row['used_native_hex_le'],
              row['measured_minus_F_ULP'] + ' / ' + row['used_minus_F_ULP']] for row in norms]
    table = table_axis.table(cellText=cells,
                             colLabels=['臂', 'measured norm', 'used norm', '相对F：measured / used（ULP）'],
                             colWidths=[.11, .24, .24, .41], cellLoc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor('#d6dfe6')
        cell.set_facecolor('#edf2f6' if row == 0 else 'white')
        if row == 0:
            cell.set_text_props(weight='bold')
    figure.text(.075, .12, 'F: 22.473445892333984；EP measured: 22.473447799682617；差1ULP = 1.9073486328125e-6。',
                fontsize=9.8, color=GRAY)
    figure.text(.075, .077, 'N/B仅替换used norm，实际返回/日志仍保留measured norm。F自身32/32仅是参照。',
                fontsize=10, color=GRAY)
    figure.text(.075, .034, '哈希不等不给出max/RMS误差；不覆盖其余参数、全部optimizer state，也不证明Muon全正确或200步修复。',
                fontsize=9.6, color=GRAY)
    save(figure, output, 'parameter-hashes-and-norm')


def main() -> None:
    """Render two figures without reading private native journals or donor tensors."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=REPORT / 'data')
    parser.add_argument('--output-dir', type=Path, default=REPORT / 'figures')
    args = parser.parse_args()
    summary, losses, norms, parameters = load(args.data_dir)
    style()
    plot_loss(losses, args.output_dir)
    plot_hashes(summary, norms, parameters, args.output_dir)
    print(json.dumps({'status': 'PLOTTED_FROM_SMALL_REAL_DATA', 'figures': 2,
                      'no_200step_gate_or_parameter_error_inferred': True}))


if __name__ == '__main__':
    main()
