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

"""Render one portable Chinese figure from all-rank CSV without private raw data."""

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


def require(condition: bool, message: str) -> None:
    """Reject incomplete, inconsistent or nonfinite portable data."""
    if not condition:
        raise ValueError(message)


def read(data: Path) -> tuple[list[dict], dict]:
    """Recompute all 1600 main-LM pairs rather than trusting a copied maximum."""
    summary = json.loads((data / 'summary.json').read_bytes())
    encoded = {name: (data / name).read_bytes() for name in ('main-lm-allranks.csv', 'curves.csv')}
    for name, raw in encoded.items():
        expected = summary['csv'][name]
        require(len(raw) == expected['bytes'] and hashlib.sha256(raw).hexdigest() == expected['sha256'], 'CSV SHA不符')
    raw_rows = list(csv.DictReader(encoded['main-lm-allranks.csv'].decode('utf-8').splitlines()))
    lookup = {}
    for row in raw_rows:
        role, rank, step, value = row['role'], int(row['rank']), int(row['step']), float(row['main_lm'])
        require(role in ('continuous400', 'reference200') and 0 <= rank < 8 and math.isfinite(value), '非法指标行')
        key = role, rank, step
        require(key not in lookup, '重复指标行')
        lookup[key] = value
    expected_keys = {(role, rank, step) for role, last in [('continuous400', 400), ('reference200', 200)]
                     for rank in range(8) for step in range(1, last + 1)}
    require(set(lookup) == expected_keys and len(raw_rows) == 4800, '4800行完整轴不符')
    curves = list(csv.DictReader(encoded['curves.csv'].decode('utf-8').splitlines()))
    require(len(curves) == 400 and [int(row['step']) for row in curves] == list(range(1, 401)), '400步轴不符')
    for index, row in enumerate(curves, 1):
        require(float(row['continuous400_rank0_lm']) == lookup['continuous400', 0, index], '连续曲线不符')
        require(all(lookup['continuous400', rank, index] == lookup['continuous400', 0, index]
                    for rank in range(8)), '连续全局标量不同rank不等')
        if index <= 200:
            require(all(lookup['reference200', rank, index] == lookup['reference200', 0, index]
                        for rank in range(8)), '参考全局标量不同rank不等')
            error = max(abs(lookup['continuous400', rank, index] - lookup['reference200', rank, index])
                        for rank in range(8))
            require(float(row['common_maxabs']) == error and error == 0, '共同窗口并非逐rank零差')
            require(float(row['reference200_rank0_lm']) == lookup['reference200', 0, index], '参考曲线不符')
        else:
            require(row['common_maxabs'] == '' and row['reference200_rank0_lm'] == '', '后200步不得补造参考')
    require(summary['resume201_to400_validated'] is False and summary['current59b_ep200']['pass'] is False,
            '恢复/EP边界不符')
    return curves, summary


def style() -> None:
    """Use available Chinese fonts and deterministic vector glyphs."""
    windows_font = Path('C:/Windows/Fonts/msyh.ttc')
    if windows_font.is_file():
        font_manager.fontManager.addfont(str(windows_font))
    installed = {item.name for item in font_manager.fontManager.ttflist}
    font = next((name for name in ('Microsoft YaHei', 'Noto Sans CJK SC', 'Source Han Sans SC', 'SimHei')
                 if name in installed), None)
    require(font is not None, '需要微软雅黑或Noto Sans CJK SC等中文字体')
    matplotlib.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': [font, 'DejaVu Sans'],
                               'font.size': 11, 'text.color': INK, 'axes.labelcolor': INK,
                               'axes.edgecolor': '#b8c5d0', 'xtick.color': INK, 'ytick.color': GRAY,
                               'axes.spines.top': False, 'axes.spines.right': False, 'axes.unicode_minus': False,
                               'figure.facecolor': 'white', 'svg.fonttype': 'path',
                               'svg.hashsalt': 'paused-goal-continuous400-v1'})


def plot(rows: list[dict], output: Path) -> None:
    """Separate the actual continuous trajectory from its limited common comparison window."""
    style()
    figure, (top, bottom) = plt.subplots(2, 1, figsize=(12, 8.8), gridspec_kw={'height_ratios': [1.4, 1]})
    figure.subplots_adjust(left=.105, right=.97, top=.82, bottom=.18, hspace=.52)
    figure.text(.065, .95, '59b：连续400步完成，共同前200步主LM零差', fontsize=18.5, weight='bold')
    figure.text(.065, .902, 'FSDP8 · CANN 9.2.0-beta.2 · FP32 NS · ns_batch_across_params=False',
                fontsize=10.5, color=GRAY)
    steps = list(range(1, 401))
    top.plot(steps, [float(row['continuous400_rank0_lm']) for row in rows], color=BLUE, linewidth=1.65,
             label='125：连续1–400（rank0）')
    top.plot(steps[:200], [float(row['reference200_rank0_lm']) for row in rows[:200]], color=ORANGE,
             linewidth=1.25, linestyle=(0, (4, 3)), label='129：同源参考1–200（重合）')
    top.set_title('(a) 原始主LM轨迹，无平滑；第201–400步仍属于同一次连续训练', loc='left', fontsize=11, pad=13)
    top.set_ylabel('主LM loss（未加权）')
    top.legend(loc='upper right', frameon=False, fontsize=9.5)
    top.set_ylim(0, max(float(row['continuous400_rank0_lm']) for row in rows) * 1.12)
    bottom.axvspan(200.5, 400, color='#eef1f4', zorder=0)
    bottom.plot(steps[:200], [float(row['common_maxabs']) for row in rows[:200]], color=BLUE, linewidth=2,
                label='共同1–200：每步8rank最大绝对差')
    bottom.plot([1, 200], [.005, .005], color=RED, linestyle='--', linewidth=1.4, label='共同窗口阈值0.005')
    bottom.text(300, .0026, '没有对应参考\n未执行checkpoint恢复\n不画resume误差曲线', color=GRAY,
                ha='center', va='center', fontsize=10.5)
    bottom.set_title('(b) 共同窗口1600对主LM记录：maxabs=0，非零点=0', loc='left', fontsize=11, pad=13)
    bottom.set_ylabel('每步最大 |LM_125 − LM_129|')
    bottom.set_xlabel('训练步（第一条loss在step1；不补造step0 loss）')
    bottom.set_ylim(-.00045, .0063)
    bottom.set_yticks([0, .0025, .005], ['0', '0.0025', '0.005'])
    bottom.legend(loc='upper left', bbox_to_anchor=(.005, .76), frameon=False, fontsize=9.2)
    for axis in (top, bottom):
        axis.set_xlim(0, 400)
        axis.set_xticks([0, 50, 100, 150, 200, 250, 300, 350, 400])
        axis.axvline(200.5, color='#b8c5d0', linestyle=':', linewidth=1)
        axis.set_axisbelow(True)
        axis.yaxis.grid(True, color='#e5ebf0', linewidth=.8)
    figure.text(.065, .09, '8rank为归约后的同值全局标量，不是8次独立重复；后200步只证明连续执行。',
                fontsize=10.2, color=GRAY)
    figure.text(.065, .047, '同源59b的EP200主LM仍FAIL：maxabs=0.0103092194（step146）；本图不作性能或修复结论。',
                fontsize=10.2, color=GRAY)
    output.mkdir(parents=True, exist_ok=True)
    png, svg = output / 'continuous400-main-lm.png', output / 'continuous400-main-lm.svg'
    require(not png.exists() and not svg.exists(), '已有图像不覆盖')
    figure.savefig(png, dpi=180, metadata={'Software': 'Continuous400 paused closeout v1'})
    figure.savefig(svg, metadata={'Date': None, 'Creator': 'Continuous400 paused closeout v1'})
    svg.write_text('\n'.join(line.rstrip(' \t') for line in svg.read_text(encoding='utf-8').splitlines()) + '\n',
                   encoding='utf-8', newline='\n')
    plt.close(figure)


def main() -> None:
    """Replot from the portable data directory; no raw evidence or backend is imported."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=REPORT / 'data')
    parser.add_argument('--output-dir', type=Path, default=REPORT / 'figures')
    args = parser.parse_args()
    rows, _ = read(args.data_dir)
    plot(rows, args.output_dir)
    print(json.dumps({'status': 'RENDERED', 'all_rank_rows_verified': 4800, 'common_pairs_recomputed': 1600}))


if __name__ == '__main__':
    main()
