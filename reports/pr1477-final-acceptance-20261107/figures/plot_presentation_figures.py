"""Seven Chinese Matplotlib figures from the sealed public scalar archive.

同数值汇总排版; these are not the training website's original exports.
Only public CSV/metadata are read. No browser/network/framework/device access.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import struct
import subprocess
import sys
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FixedLocator, ScalarFormatter

from verify_public_csv_stats import CASES, STEPS, load_and_verify, sha, write_json, require

LABELS = {'fsdp_full': 'FSDP 全重算 · 496', 'fsdp_backbone': 'FSDP 骨干重算 · 496',
          'ep2': 'EP2 · 496', 'cp2dp2': 'CP2×DP2 · 496',
          'tp2cp2': 'TP2×CP2 · 3292', 'tp2ep2sp': 'TP2×EP2 + SP · 3292'}
TABLE_LABELS = {'fsdp_full': 'FSDP 全重算 / 496', 'fsdp_backbone': 'FSDP 骨干 / 496',
                'ep2': 'EP2 / 496', 'cp2dp2': 'CP2×DP2 / 496',
                'tp2cp2': 'TP2×CP2 / 3292', 'tp2ep2sp': 'TP2×EP2+SP / 3292'}
COLORS = {'fsdp_full': '#4477AA', 'fsdp_backbone': '#228833', 'ep2': '#AA7700',
          'cp2dp2': '#AA3377', 'tp2cp2': '#CC4444', 'tp2ep2sp': '#0099BB'}
HP, MF, ZERO, THRESHOLD = '#007F86', '#B24645', '#68727C', '#B64C55'
FIGURES, REGISTRY, CONTROLS = [], [], []
CURRENT = None


def header(fig, title, subtitle):
    fig.text(.06, .972, title, fontsize=21, fontweight='bold', va='top', color='#172B3A')
    fig.text(.06, .925, subtitle, fontsize=11.4, va='top', color='#52616C')


def footer(fig, single=False, aux=False, note=None):
    if aux:
        first = 'AUX 仅观察，无阈值｜单卡 GBS2 / 四卡 GBS4，LBS1；各用各的参考，不跨 GBS 混比'
    elif single:
        first = '单卡 GBS2 / LBS1 / GA2；从 MF1500 独立恢复，参考冻结 MF52f 连续训练'
    else:
        first = '四卡 GBS4 / LBS1；主 LOSS 与 raw MTP 统一参考旧 continuous a852，GA 与布局见 README'
    model = '2 个基础层（1 Dense + 1 MoE）+ 1 MTP，S=4096；fused mHC；clamp、CMHC 关闭'
    versions = ('MF actual52f / MS2.10 / CANN9.1；HP source49bf / PTA2.12 / CANN9.2' if single else
                'TP1 四配置 source496；TP/LP 两配置 source3292（含366）；HP PTA2.12 / CANN9.2')
    if aux:
        versions = 'MF52f / MS2.10 / CANN9.1；HP49bf、496、3292 / PTA2.12 / CANN9.2；四卡AUX参考仅新FSDP496'
    final = note or '同数值汇总排版｜1501–2500，各1000点，无平滑/插值/裁剪；raw MTP 含既有 weighted→raw 舍入'
    for y, text in zip((.126, .095, .064, .033), (first, model, versions, final)):
        fig.text(.06, y, text, fontsize=9.7, color='#52616C', va='center')


def style(ax, title, ylabel, scientific=False):
    ax.set_title(title, loc='left', fontsize=13.5, pad=12, color='#172B3A', fontweight='bold')
    ax.set_xlabel('训练步', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xlim(1501, 2500)
    ax.xaxis.set_major_locator(FixedLocator([1501, 1750, 2000, 2250, 2500]))
    ax.tick_params(axis='both', labelsize=10, colors='#52616C', length=3)
    ax.grid(axis='y', color='#E6EAED', linewidth=.75)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    for edge in ('left', 'bottom'):
        ax.spines[edge].set_color('#C3CCD2')
    formatter = ScalarFormatter(useOffset=False, useMathText=False)
    formatter.set_scientific(scientific)
    if scientific:
        formatter.set_powerlimits((-3, 3))
    ax.yaxis.set_major_formatter(formatter)
    ax.margins(y=.12)


def vector_sha(values):
    return hashlib.sha256(struct.pack('<' + str(len(values)) + 'd', *values)).hexdigest()


def add(ax, values, label, color, *, case, metric, kind, width=.85, alpha=.88, linestyle='-'):
    require(len(values) == 1000 and all(math.isfinite(v) for v in values), 'numeric plot series differs')
    line, = ax.plot(STEPS, values, label=label, color=color, linewidth=width, alpha=alpha,
                    linestyle=linestyle, solid_capstyle='round', antialiased=True)
    line.get_path().should_simplify = False
    REGISTRY.append({'figure': CURRENT, 'line': line, 'axis': ax, 'case': case, 'metric': metric,
                     'kind': kind, 'label': label, 'x': STEPS[:], 'y': values[:],
                     'points': 1000, 'x_sha256_float64_le': vector_sha(STEPS),
                     'y_sha256_float64_le': vector_sha(values)})
    return line


def zero(ax):
    ax.axhline(0, color=ZERO, linewidth=.85, linestyle=(0, (4, 3)), zorder=1)
    CONTROLS.append({'figure': CURRENT, 'kind': 'zero-reference'})


def criterion(ax):
    for value in (-.005, .005):
        ax.axhline(value, color=THRESHOLD, linewidth=1, linestyle=(0, (5, 3)), zorder=1)
        CONTROLS.append({'figure': CURRENT, 'kind': 'complete-window-mean-criterion', 'value': value})
    zero(ax)


def common_legend(fig, handles, *, columns=3, y=.884, fontsize=10.6):
    return fig.legend(handles=handles, labels=[x.get_label() for x in handles], loc='upper left',
                      bbox_to_anchor=(.055, y), ncol=columns, frameon=False, fontsize=fontsize,
                      handlelength=2.3, columnspacing=1.55, labelspacing=.75)


def stats_card(ax, name, data, color):
    ax.set_axis_off()
    stats = data['stats']
    ax.text(.04, .97, name + ' · 完整1000步', fontsize=13.5, fontweight='bold', color='#172B3A', va='top')
    ax.text(.04, .77, f"{stats['signed_mean']:+.8f}", fontsize=23, fontweight='bold', color=color)
    ax.text(.04, .65, '有符号均差 mean(HP − MF)', fontsize=10.3, color='#52616C')
    rows = [('MAE', f"{stats['MAE']:.8f}"), ('最大 |Δ|', f"{stats['max_abs']:.8f}"),
            ('最大差值发生步', str(stats['max_step'])),
            ('正 / 负 / 零', f"{stats['positive']} / {stats['negative']} / {stats['zero']}")]
    for y, (key, value) in zip((.49, .35, .21, .07), rows):
        ax.text(.04, y, key, fontsize=10.6, color='#52616C')
        ax.text(.97, y, value, fontsize=11.1, ha='right', color='#172B3A')


def final_single_card(ax, data):
    ax.set_axis_off()
    ax.text(.02, .97, '完整窗口终值', fontsize=16, fontweight='bold', va='top', color='#172B3A')
    ax.text(.02, .86, '验收对象：1000步 signed mean\n不是逐步 Δ、MAE 或最大差值', fontsize=11, linespacing=1.8, color='#52616C', va='top')
    for y, metric, title, color in ((.59, 'loss', 'LOSS', HP), (.36, 'mtp', 'raw MTPLOSS', '#6950A3')):
        mean = data['metrics'][metric]['stats']['signed_mean']
        ax.text(.02, y + .08, title, fontsize=12, color='#52616C')
        ax.text(.02, y, f'{mean:+.8f}', fontsize=22, fontweight='bold', color=color)
        ax.text(.02, y - .055, f'|mean| = {abs(mean):.8f} ≤ 0.005', fontsize=11, color='#247644')
    ax.text(.02, .13, '两项完整窗口均值规则满足\n中间累计值仅供观察', fontsize=11.2, linespacing=1.9, color='#247644')


def start(name):
    global CURRENT
    CURRENT = name
    return plt.figure(figsize=(16, 9), facecolor='white')


def save(fig, output, name):
    fig.canvas.draw()
    canvas = fig.get_window_extent()
    for text in fig.texts:
        bbox = text.get_window_extent(fig.canvas.get_renderer())
        require(bbox.x0 >= -1 and bbox.x1 <= canvas.width + 1 and bbox.y0 >= -1 and bbox.y1 <= canvas.height + 1,
                name + ': figure text outside canvas: ' + text.get_text())
    rows = [r for r in REGISTRY if r['figure'] == name]
    for row in rows:
        require(list(row['line'].get_xdata()) == STEPS and list(row['line'].get_ydata()) == row['y'], name + ': plotted data changed')
        require(tuple(row['axis'].get_xlim()) == (1501., 2500.), name + ': x range differs')
        low, high = row['axis'].get_ylim()
        require(low <= min(row['y']) and high >= max(row['y']), name + ': y data clipped')
    description = '同数值汇总排版；公开原CSV全部1000点；无平滑/数值插值/裁剪；不是网站原始导出。'
    fig.savefig(output / (name + '.png'), dpi=150, facecolor='white', metadata={'Title': name, 'Description': description})
    fig.savefig(output / (name + '.svg'), facecolor='white', metadata={'Title': name, 'Description': description,
                'Creator': 'Matplotlib', 'Date': '2026-11-07'})
    FIGURES.append({'name': name, 'PNG': name + '.png', 'SVG': name + '.svg',
                    'size_inches': [16, 9], 'PNG_size_pixels': [2400, 1350],
                    'numeric_series': len(rows), 'numeric_points': sum(r['points'] for r in rows),
                    'legend_boxes': [list(legend.get_window_extent(fig.canvas.get_renderer()).bounds) for legend in fig.legends]})
    plt.close(fig)


def draw_all(cases, output):
    single = cases['single']
    # 01: original scalar arrays, separately labelled LOSS and unweighted MTP.
    name = '01_single_raw'; fig = start(name)
    header(fig, '单卡续训｜LOSS 与 raw MTPLOSS', 'HP 对比冻结 MF52f 连续参考；两个原始序列各1000点，细小差异见逐步差值图')
    gs = fig.add_gridspec(1, 2, left=.065, right=.98, bottom=.21, top=.79, wspace=.15)
    handles = []
    for index, (metric, title) in enumerate((('loss', '主 LM LOSS'), ('mtp', 'raw MTPLOSS（未乘0.1）'))):
        ax = fig.add_subplot(gs[0, index]); data = single['metrics'][metric]
        reference = add(ax, data['reference'], 'MF52f 连续参考', MF, case='single', metric=metric, kind='raw_reference', width=1.35, alpha=.65)
        actual = add(ax, data['actual'], 'HP source49bf 续训', HP, case='single', metric=metric, kind='raw_actual', width=.85, alpha=.9)
        style(ax, title, title)
        if not handles: handles = [reference, actual]
    common_legend(fig, handles, columns=2, y=.887, fontsize=11.2); footer(fig, single=True); save(fig, output, name)
    # 02: zero-referenced per-step differences; independent complete-window statistics.
    name = '02_single_signed_deltas'; fig = start(name)
    header(fig, '单卡逐步差值｜保留全部正负方向', 'Δ = HP − MF；零线用于定位，0.005规则只约束完整1000步有符号均差')
    gs = fig.add_gridspec(2, 2, left=.065, right=.98, bottom=.21, top=.825, width_ratios=(3.3, 1.55), hspace=.38, wspace=.16)
    for row, (metric, title, color) in enumerate((('loss', 'LOSS', HP), ('mtp', 'raw MTPLOSS', '#6950A3'))):
        ax = fig.add_subplot(gs[row, 0]); data = single['metrics'][metric]
        zero(ax); add(ax, data['delta'], 'Δ = HP − MF', color, case='single', metric=metric, kind='signed_delta', width=.9)
        style(ax, title + ' 逐步 Δ', '有符号差值 Δ', scientific=True)
        stats_card(fig.add_subplot(gs[row, 1]), title, data, color)
    footer(fig, single=True); save(fig, output, name)
    # 03: independent fsum prefix means; criterion lines apply to full end only.
    name = '03_single_cumulative'; fig = start(name)
    header(fig, '单卡累计有符号均差｜完整窗口终值', '第 n 点 = fsum(前 n 个逐步差值) / n；±0.005线用于显示完整1000步的验收尺度')
    gs = fig.add_gridspec(2, 2, left=.065, right=.98, bottom=.21, top=.825, width_ratios=(3.35, 1.45), hspace=.38, wspace=.18)
    for row, (metric, title, color) in enumerate((('loss', 'LOSS', HP), ('mtp', 'raw MTPLOSS', '#6950A3'))):
        ax = fig.add_subplot(gs[row, 0]); criterion(ax)
        add(ax, single['metrics'][metric]['cumulative'], title, color, case='single', metric=metric, kind='cumulative_signed_mean', width=1.65)
        style(ax, title + ' 累计均差', '累计 mean(Δ)', scientific=True)
    final_single_card(fig.add_subplot(gs[:, 1]), single); footer(fig, single=True); save(fig, output, name)
    # 04: all six complete per-step differences, same historical primary reference.
    name = '04_four_signed_deltas'; fig = start(name)
    header(fig, '四卡六配置｜逐步有符号差值', 'Δ = 本配置 actual − 旧 continuous a852；LOSS 与 raw MTP 独立展示，未用新 FSDP 替换主参考')
    gs = fig.add_gridspec(2, 1, left=.065, right=.98, bottom=.21, top=.765, hspace=.43)
    handles = []
    for row, (metric, title) in enumerate((('loss', 'LOSS'), ('mtp', 'raw MTPLOSS'))):
        ax = fig.add_subplot(gs[row]); zero(ax)
        for case in CASES:
            line = add(ax, cases[case]['metrics'][metric]['delta'], LABELS[case], COLORS[case], case=case, metric=metric, kind='signed_delta', width=.75, alpha=.75, linestyle='--' if case.startswith('tp') else '-')
            if row == 0: handles.append(line)
        style(ax, title + ' 逐步 Δ', 'actual − reference', scientific=True)
    common_legend(fig, handles, columns=3, y=.887); footer(fig); save(fig, output, name)
    # 05: six prefix means and a separate end-value table.
    name = '05_four_cumulative'; fig = start(name)
    header(fig, '四卡六配置｜累计均差与完整窗口终值', '每条曲线含1000个精确 fsum 前缀；主参考仍为旧 continuous a852，AUX不参与此验收')
    gs = fig.add_gridspec(2, 2, left=.065, right=.98, bottom=.21, top=.765, width_ratios=(3.2, 1.75), hspace=.43, wspace=.18)
    handles = []
    for row, (metric, title) in enumerate((('loss', 'LOSS'), ('mtp', 'raw MTPLOSS'))):
        ax = fig.add_subplot(gs[row, 0]); criterion(ax)
        for case in CASES:
            line = add(ax, cases[case]['metrics'][metric]['cumulative'], LABELS[case], COLORS[case], case=case, metric=metric, kind='cumulative_signed_mean', width=1.3, alpha=.95, linestyle='--' if case.startswith('tp') else '-')
            if row == 0: handles.append(line)
        style(ax, title + ' 累计均差', '累计 mean(Δ)', scientific=True)
    table_ax = fig.add_subplot(gs[:, 1]); table_ax.axis('off')
    table_ax.text(0, .97, '完整1000步：终值表', fontsize=14, fontweight='bold', va='top', color='#172B3A')
    cells = [[TABLE_LABELS[case], f"{cases[case]['metrics']['loss']['stats']['signed_mean']:+.3e}",
              f"{cases[case]['metrics']['mtp']['stats']['signed_mean']:+.3e}"] for case in CASES]
    table = table_ax.table(cellText=cells, colLabels=['配置 / 源', 'LOSS μΔ', 'MTP μΔ'],
                           colWidths=[.52, .24, .24], cellLoc='center', bbox=[0, .33, 1, .55])
    table.auto_set_font_size(False); table.set_fontsize(9.7)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#DDE4E9'); cell.set_linewidth(.7)
        if row == 0: cell.set_facecolor('#EAF0F4'); cell.get_text().set_fontweight('bold')
        else:
            cell.set_facecolor('#FAFBFC' if row % 2 else 'white')
            if col == 0: cell.get_text().set_color(COLORS[CASES[row - 1]])
    table_ax.text(0, .23, '12/12 完整窗口均值规则满足', fontsize=13, fontweight='bold', color='#247644')
    table_ax.text(0, .15, '|mean(Δ)| ≤ 0.005，分别检验两指标', fontsize=10.4, color='#52616C')
    table_ax.text(0, .065, '中间累计值不作提前通过判断\n原生/CPU状态门见原证据包', fontsize=10.4, linespacing=1.7, color='#52616C')
    common_legend(fig, handles, columns=3, y=.887); footer(fig); save(fig, output, name)
    # 06: old native continuous reference plus all six actual raw series.
    name = '06_four_raw'; fig = start(name)
    header(fig, '四卡原始轨迹｜旧连续参考与六个 actual', '七条真实原始序列逐点叠加；相邻点仅用折线连接，未补点或重新计算原生MTP')
    gs = fig.add_gridspec(2, 1, left=.065, right=.98, bottom=.21, top=.765, hspace=.43)
    handles = []
    for row, (metric, title) in enumerate((('loss', '主 LM LOSS'), ('mtp', 'raw MTPLOSS（未乘0.1）'))):
        ax = fig.add_subplot(gs[row])
        reference = add(ax, cases['fsdp_full']['metrics'][metric]['reference'], '旧 continuous a852', '#202D37', case='four_reference', metric=metric, kind='raw_reference', width=1.5, alpha=.60, linestyle=(0, (3, 2)))
        if row == 0: handles.append(reference)
        for case in CASES:
            line = add(ax, cases[case]['metrics'][metric]['actual'], LABELS[case], COLORS[case], case=case, metric=metric, kind='raw_actual', width=.72, alpha=.78)
            if row == 0: handles.append(line)
        style(ax, title, title)
    common_legend(fig, handles, columns=4, y=.887, fontsize=10.0); footer(fig); save(fig, output, name)
    # 07: AUX never borrows the historical incomplete four-card AUX reference.
    name = '07_aux_diagnostics'; fig = start(name)
    header(fig, 'AUX 观察｜单卡与四卡各自对照', 'AUX没有新增验收阈值；四卡比较只使用新 FSDP496 全rank/micro/router 汇总，保留公开CSV实际值')
    gs = fig.add_gridspec(2, 2, left=.065, right=.98, bottom=.285, top=.77, hspace=.62, wspace=.15)
    ax = fig.add_subplot(gs[0, 0]); data = single['metrics']['aux']
    h1 = add(ax, data['reference'], 'MF52f raw AUX', MF, case='single', metric='aux', kind='raw_reference', width=1.25, alpha=.65)
    h2 = add(ax, data['actual'], 'HP49bf raw AUX', HP, case='single', metric='aux', kind='raw_actual', width=.85)
    style(ax, '单卡 raw AUX（MF参考）', 'raw AUX')
    ax = fig.add_subplot(gs[0, 1]); zero(ax)
    add(ax, data['delta'], 'HP − MF', HP, case='single', metric='aux', kind='signed_delta', width=.85)
    style(ax, f"单卡 ΔAUX · μ={data['stats']['signed_mean']:+.3e}", 'HP − MF', scientific=True)
    h4 = []
    for col, kind in ((0, 'actual'), (1, 'delta')):
        ax = fig.add_subplot(gs[1, col])
        if col == 1: zero(ax)
        for case in CASES:
            line = add(ax, cases[case]['metrics']['aux'][kind], LABELS[case], COLORS[case], case=case, metric='aux', kind='raw_actual' if col == 0 else 'signed_delta', width=.8, alpha=.82)
            if col == 0: h4.append(line)
        style(ax, '四卡 raw AUX（含新FSDP）' if col == 0 else '四卡 ΔAUX（只对新FSDP）', 'raw AUX' if col == 0 else 'actual − 新FSDP', scientific=(col == 1))
    common_legend(fig, [h1, h2], columns=2, y=.887, fontsize=10.7)
    common_legend(fig, h4, columns=3, y=.215, fontsize=10.1)
    footer(fig, aux=True, note='同数值汇总排版｜两种GBS的AUX不混比｜1501–2500各1000点，无平滑/插值/裁剪，无AUX阈值线')
    save(fig, output, name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=Path(__file__).resolve().parent.parent / 'numerical')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--font', type=Path, default=Path(os.environ.get('WINDIR', 'C:/Windows')) / 'Fonts/msyh.ttc')
    args = parser.parse_args()
    require(args.font.is_file(), 'Microsoft YaHei font required; provide --font to an installed YaHei font')
    args.output.mkdir(parents=True, exist_ok=True)
    font_manager.fontManager.addfont(str(args.font))
    bold = args.font.with_name('msyhbd.ttc')
    if bold.exists(): font_manager.fontManager.addfont(str(bold))
    family = font_manager.FontProperties(fname=args.font).get_name()
    plt.rcParams.update({'font.family': family, 'font.size': 11, 'axes.unicode_minus': False,
                         'svg.fonttype': 'none', 'svg.hashsalt': 'PR1477-presentation-figures',
                         'path.simplify': False, 'agg.path.chunksize': 0, 'figure.dpi': 150,
                         'savefig.dpi': 150, 'axes.formatter.useoffset': False})
    warnings.filterwarnings('error', message='Glyph .* missing from font')
    cases, verification, metadata = load_and_verify(args.archive)
    write_json(args.output / 'CSV-verification.json', verification)
    draw_all(cases, args.output)
    require(len(FIGURES) == 7 and sum(f['numeric_series'] for f in FIGURES) == 61, 'figure/series coverage differs')
    require(sum(f['numeric_points'] for f in FIGURES) == 61000, 'full point coverage differs')
    require(not any(c['figure'] == '07_aux_diagnostics' and c['kind'] == 'complete-window-mean-criterion' for c in CONTROLS), 'AUX criterion not allowed')
    for relative, item in verification['CSV_files'].items():
        require(sha(args.archive / relative) == item['sha256'], 'source CSV changed while plotting')
    serial_registry = [{k: v for k, v in r.items() if k not in ('line', 'axis', 'x', 'y')} for r in REGISTRY]
    write_json(args.output / 'plot-data-audit.json', {'status': 'all61_numeric_lines_have_exact1000_original_or_fsum_points',
               'window': [1501, 2500], 'numeric_points_plotted': 61000, 'figures': FIGURES,
               'series': serial_registry, 'control_lines': CONTROLS,
               'simplification': False, 'resampling': False, 'smoothing': False, 'interpolated_numeric_samples': False,
               'x_limits_all': [1501, 2500], 'all_data_inside_y_limits': True,
               'SVG_fonttype': 'none; text and paths editable',
               'display_note': '同数值汇总排版，非网站原始导出。PNG has finite display pixels; SVG preserves each plotted path vertex without simplification.'})
    freeze = subprocess.run([sys.executable, '-m', 'pip', 'freeze', '--all'], capture_output=True, text=True, check=True).stdout
    (args.output / 'requirements-freeze.txt').write_text(freeze, encoding='utf8', newline='\n')
    write_json(args.output / 'render-runtime.json', {'python': platform.python_version(), 'matplotlib': matplotlib.__version__,
               'platform': platform.system(), 'backend': matplotlib.get_backend(), 'font_family': family,
               'font_sha256': sha(args.font), 'font_file_name': args.font.name,
               'PNG_dpi': 150, 'size_inches': [16, 9], 'browser_used': False, 'network_used': False,
               'framework_or_device_imported': False, 'original_package_modified': False})
    # README is a hand-maintained portable companion, listed in the output manifest.
    files = {}
    for path in sorted(args.output.iterdir()):
        if path.is_file() and path.name not in ('manifest.json', 'SHA256SUMS'):
            files[path.name] = {'bytes': path.stat().st_size, 'sha256': sha(path)}
    manifest = {'schema': 'pr1477-matplotlib-presentation-figures-v1',
                'artifact_kind': '同数值汇总排版', 'website_original_export': False,
                'source_archive_manifest_sha256': verification['source_archive_manifest_sha256'],
                'source_CSV_files': verification['CSV_files'], 'figures': FIGURES, 'files': files,
                'no_original_data_package_or_website_files_changed': True}
    write_json(args.output / 'manifest.json', manifest)
    sums = ''.join(item['sha256'] + '  ' + name + '\n' for name, item in files.items())
    sums += sha(args.output / 'manifest.json') + '  manifest.json\n'
    (args.output / 'SHA256SUMS').write_text(sums, encoding='utf8', newline='\n')
    print(json.dumps({'figures': 7, 'numeric_series': 61, 'numeric_points': 61000,
                      'CSV_files_SHA_verified': 26, 'required_mean_rules_verified': 14,
                      'AUX_threshold': None, 'PNG': '2400x1350', 'SVG_editable_text': True,
                      'manifest_sha256': sha(args.output / 'manifest.json')}))


if __name__ == '__main__':
    main()
