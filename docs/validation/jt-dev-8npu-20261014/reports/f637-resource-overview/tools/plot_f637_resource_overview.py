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

"""Plot fixed-source resource observations with explicit, separate precision verdicts."""

import csv
import hashlib
import json
from pathlib import Path

from all_rank_resources import summarize
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position

ROOT = Path(__file__).resolve().parents[1]
SPECS = [
    ('FSDP8 · 关闭重算', 'f637-fsdp8-r4', 'f637-fsdp8-baseline-independent-comparison'),
    ('FSDP8 · 完全重算', 'f637-fsdp8-full-200', 'f637-fsdp8-full-200-strict-comparison'),
    ('FSDP8 · 选择重算', 'f637-fsdp8-selective-r3', 'f637-fsdp8-selective-r3-strict-comparison'),
    ('FSDP + EP2', 'f637-ep2-200', 'f637-ep2-200-strict-comparison'),
    ('FSDP + CP2', 'f637-cp2-200', 'f637-cp2-200-strict-comparison'),
    ('FSDP + TP2 / SP', 'f637-tp2-r4', 'f637-tp2-r4-strict-comparison'),
]


def collect_rows() -> tuple[list, dict]:
    """Recompute existing reports from all 48 raw rank files and verify their hashes."""
    rows, sources = [], {}
    for label, resource_name, strict_name in SPECS:
        resource_path = ROOT / 'evidence' / (resource_name + '-all-rank-resources.json')
        strict_path = ROOT / 'evidence' / (strict_name + '.json')
        resource = json.loads(resource_path.read_bytes())
        strict = json.loads(strict_path.read_bytes())
        if strict['steps'] != [1, 200] or strict['findings'] or strict['candidate'] != resource['case']:
            raise AssertionError('Incomplete or mismatched training comparison')
        raw_ranks = []
        for rank in range(8):
            path = ROOT / 'runs' / resource['case'] / f'metrics-rank{rank}.jsonl'
            data = path.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            if digest != resource['input_sha256'][path.name]:
                raise AssertionError('Resource input hash differs from its audited source')
            values = [json.loads(line) for line in data.splitlines()]
            if [row['step'] for row in values] != list(range(1, 201)):
                raise AssertionError('Expected one complete independent 200-step run')
            raw_ranks.append({row['step']: row for row in values})
            sources[path.relative_to(ROOT).as_posix()] = digest
        calculated = summarize(raw_ranks, 1, 200)
        if any(value != resource[key] for key, value in calculated.items()):
            raise AssertionError('Raw resource recalculation differs from the audited report')
        for path in (resource_path, strict_path):
            sources[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append({'label': label, 'case': resource['case'], 'precision_status': strict['status'],
                     'allocated_gib': resource['allocated_gib'], 'reserved_gib': resource['reserved_gib'],
                     'median_step_seconds': resource['median_step_seconds'],
                     'supervised_tokens_per_second': resource['supervised_tokens_per_second'],
                     'lm_max_abs': strict['unweighted_error_diagnostics']['lm'],
                     'mtp_derived_max_abs': strict['unweighted_error_diagnostics']['mtp'],
                     'aux_derived_max_abs': strict['unweighted_error_diagnostics']['aux']})
    if [row['precision_status'] for row in rows] != ['PASS'] * 3 + ['FAIL'] * 3:
        raise AssertionError('Chart labels need to be updated for changed precision evidence')
    for row in rows:
        row['allocated_saving_percent'] = 100 * (1 - row['allocated_gib'] / rows[0]['allocated_gib'])
        row['step_time_increase_percent'] = 100 * (row['median_step_seconds'] / rows[0]['median_step_seconds'] - 1)
    return rows, sources


def plot(rows: list) -> None:
    """Render resource charts without implying that failed precision is acceptable."""
    plt.rcParams.update({'font.family': 'Microsoft YaHei', 'font.size': 11, 'svg.fonttype': 'none',
                         'axes.spines.top': False, 'axes.spines.right': False})
    figure, axes = plt.subplots(1, 2, figsize=(14.8, 7.4), sharey=True)
    positions = list(range(len(rows)))
    colors = ['#64748B', '#126B68', '#369990', '#B55B4F', '#B55B4F', '#B55B4F']
    for index, (key, limit, title, xlabel) in enumerate([
            ('allocated_gib', 29, '整轮 allocated 显存峰值', 'GiB / 卡，全部 8 个 rank 的最大值'),
            ('median_step_seconds', 2.38, '第 21–200 步耗时中位数', '秒 / 步，先逐步取全部 rank 的最大值')]):
        axis = axes[index]
        bars = axis.barh(positions, [row[key] for row in rows], color=colors, height=0.58)
        for position, (rectangle, row) in enumerate(zip(bars, rows)):
            if row['precision_status'] == 'FAIL':
                rectangle.set_hatch('//')
                rectangle.set_edgecolor('#F8EDEB')
            if index == 0:
                suffix = '基线' if position == 0 else f"节省 {row['allocated_saving_percent']:.1f}%"
                label = f"{row[key]:.3f}  ·  {suffix}"
            else:
                suffix = '基线' if position == 0 else f"+{row['step_time_increase_percent']:.1f}%"
                label = f"{row[key]:.3f}  ·  {suffix}"
            axis.text(row[key] + limit * 0.015, position, label, va='center', fontsize=10)
        axis.axvline(rows[0][key], color='#64748B', linestyle='--', linewidth=1)
        axis.set_xlim(0, limit)
        axis.set_ylim(len(rows) - 0.4, -0.6)
        axis.set_title(title, loc='left', weight='bold', pad=18)
        axis.set_xlabel(xlabel, labelpad=12)
        axis.grid(axis='x', alpha=0.16)
        axis.set_axisbelow(True)
    axes[0].set_yticks(positions, [row['label'] + ('（精度通过）' if row['precision_status'] == 'PASS'
                                               else '（精度未通过）') for row in rows])
    figure.text(0.035, 0.945, '当前固定配方：显存下降，步耗时上升', fontsize=20, weight='bold')
    figure.text(0.035, 0.898, '129 · 8×Ascend 910B · CANN 9.2.0-beta.2 · JT_dev f637ef7 + 修复源 18602c4',
                fontsize=11, color='#475569')
    figure.text(0.035, 0.073,
                '每项独立完整 200 步；绿色重算结果全 loss 差 0，红色斜线的 EP/CP/TP 仍超 0.005，不能视作已验收收益。\n'
                '单次匹配观测，保留相同输入身份采集开销；不是无插桩或端到端性能。显存含前 20 步；reserved 与原始指标见配套数据。\n'
                'TP2 同时使用 SP / loss parallel；各布局保持 GBS=8，微批次数随 DP 变化，不能只按并行度解释耗时。',
                fontsize=10, color='#475569', linespacing=1.6)
    figure.subplots_adjust(left=0.25, right=0.985, top=0.79, bottom=0.25, wspace=0.2)
    for extension in ('png', 'svg'):
        path = ROOT / 'figures' / ('f637-current-resource-overview-zh.' + extension)
        if path.exists():
            raise FileExistsError(path)
        figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    """Save reproducible numerical inputs alongside the scientific chart."""
    rows, sources = collect_rows()
    report = {'source_commit': '18602c4524dc664b81b2bad9a7b624455ce1494f', 'rows': rows,
              'input_sha256': sources,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'scope': '各策略单次完整200步；资源统计已从48份原始rank指标重算，精度结论来自各自严格验收报告。'}
    with (ROOT / 'evidence/f637-current-resource-overview.json').open('x', encoding='utf-8') as handle:
        handle.write(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    with (ROOT / 'evidence/f637-current-resource-overview.csv').open('x', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot(rows)
    print(json.dumps({'strategies': len(rows), 'source_files_verified': len(sources),
                      'rows': rows}, ensure_ascii=False))


if __name__ == '__main__':
    main()
