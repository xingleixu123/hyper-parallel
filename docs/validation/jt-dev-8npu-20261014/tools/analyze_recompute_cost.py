"""Summarize measured recompute A/B peaks and bounded inclusive module events."""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
POLICIES = ('off', 'full', 'selective', 'attention', 'mlp', 'norm', 'mtp')


def main():
    rows = []
    for policy in POLICIES:
        case = f'module-{policy}-diagnostic40'
        report = json.loads((ROOT / 'evidence' / f'{case}-comparison.json').read_text())
        if report['status'] != 'PASS' or report['steps'] != [1, 40] or report['ranks'] != 8:
            raise ValueError(f'Incomplete or failed A/B: {case}')
        metrics = [json.loads(line) for line in
                   (ROOT / 'runs' / case / 'metrics-rank0.jsonl').read_text().splitlines()]
        if [item['step'] for item in metrics] != list(range(1, 41)):
            raise ValueError(f'Invalid step coverage: {case}')
        times = [item['performance/step_time'] for item in metrics[20:]]
        row = {'policy': policy, 'run': case, **report['performance'],
               'timed_steps': 20, 'p10_step_seconds': float(np.percentile(times, 10)),
               'p90_step_seconds': float(np.percentile(times, 90)),
               'max_loss_error': max(value['max_abs'] for key, value in report['errors'].items()
                                     if key != 'training/grad_norm')}
        rows.append(row)
    baseline = rows[0]
    for row in rows:
        row['saved_allocated_gib'] = baseline['allocated_gib'] - row['allocated_gib']
        row['extra_step_seconds'] = row['median_step_seconds'] - baseline['median_step_seconds']
        row['step_time_change_percent'] = 100 * row['extra_step_seconds'] / baseline['median_step_seconds']
        row['saved_gib_per_extra_second'] = (row['saved_allocated_gib'] / row['extra_step_seconds']
                                           if row['extra_step_seconds'] > 0.005 else None)
    event_rows = []
    for policy in ('off', 'full', 'selective'):
        case = ROOT / 'runs' / f'module-{policy}-events12'
        if (case / 'EXIT_CODE').read_text().strip() != '0':
            raise ValueError(f'Incomplete event run: {case.name}')
        groups = defaultdict(list)
        for rank in range(8):
            records = [json.loads(line) for line in
                       (case / f'module-profile-rank{rank}.jsonl').read_text().splitlines()]
            if {item['step'] for item in records} != set(range(6, 11)):
                raise ValueError(f'Unexpected measured event window: {case.name}, rank {rank}')
            for item in records:
                if item['rank'] != rank or item['elapsed_ms_inclusive'] < 0:
                    raise ValueError('Invalid event record')
                groups[(item['name'], item['phase'], item['completed'])].append(item)
        for (name, phase, completed), records in sorted(groups.items()):
            event_rows.append({
                'policy': policy, 'module': name, 'phase': phase, 'completed': completed,
                'calls': len(records), 'ranks': len({item['rank'] for item in records}),
                'median_inclusive_ms': statistics.median(item['elapsed_ms_inclusive'] for item in records),
                'p90_inclusive_ms': float(np.percentile([item['elapsed_ms_inclusive'] for item in records], 90)),
                'median_forward_allocation_delta_mib': statistics.median(
                    item['allocated_delta_bytes'] for item in records) / 2**20,
            })
    report = {
        'source': 'ec2e16bce26e497519c573f0a8d87a538a329197',
        'scope': '40-step uninstrumented A/B, timing steps 21-40; 12-step separate event runs, events steps 6-10.',
        'limitations': [
            'One run per policy; within-run percentiles are not confidence intervals.',
            'Module allocation deltas include outputs and temporaries, not saved activation sizes.',
            'Module event times are inclusive; nested module times must not be summed.',
            'Replay may stop early; completed=false records are partial calls.',
            'The module-only policies are experimental hooks, not supported YAML module selectors.',
            '40-step exact agreement does not replace 200-step acceptance.',
        ],
        'policies': rows, 'module_events': event_rows,
    }
    (ROOT / 'evidence/recompute-cost.json').write_text(json.dumps(report, indent=2) + '\n')
    for name, records in [('recompute-policy-cost', rows), ('recompute-module-events', event_rows)]:
        with (ROOT / 'evidence' / f'{name}.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    plot(rows)
    plot_events(event_rows)
    print(json.dumps({'policies': rows, 'event_groups': len(event_rows)}, indent=2))


def plot(rows, chinese=False):
    plt.rcParams.update({'font.family': 'Microsoft YaHei' if chinese else 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none'})
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.7))
    labels = (['关闭重算', '完整解码器', '选择算子', '仅注意力', '仅 MLP', '仅归一化', '仅 MTP 解码器'] if chinese else
              ['Off', 'Full', 'Selective ops', 'Attention only', 'MLP only', 'Norm only', 'MTP block only'])
    positions = np.arange(len(rows))
    colors = ['#8B9AAF', '#165D8D', '#32A0A7', '#45769C', '#25745C', '#A18A46', '#8C6DA0']
    memory = [row['allocated_gib'] for row in rows]
    bars = axes[0].barh(positions, memory, color=colors)
    axes[0].bar_label(bars, labels=[f'{value:.2f}' for value in memory], padding=4)
    axes[0].set_yticks(positions, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, max(memory) * 1.14)
    axes[0].set_xlabel('单卡 GiB（全部 8 个 rank 的最大值）' if chinese else
                       'GiB / device (maximum across 8 ranks)')
    axes[0].set_title('已分配显存峰值' if chinese else 'Peak allocated memory', loc='left', weight='bold')
    times = np.array([row['median_step_seconds'] for row in rows])
    low = times - np.array([row['p10_step_seconds'] for row in rows])
    high = np.array([row['p90_step_seconds'] for row in rows]) - times
    axes[1].errorbar(times, positions, xerr=[low, high], fmt='none', ecolor='#8B9AAF', capsize=3)
    axes[1].scatter(times, positions, c=colors, s=45, zorder=3)
    for index, row in enumerate(rows):
        axes[1].annotate(f"{row['median_step_seconds']:.3f}s ({row['step_time_change_percent']:+.1f}%)",
                         (times[index], index), xytext=(8, -13), textcoords='offset points', fontsize=9)
    axes[1].axvline(times[0], color='#8B9AAF', linestyle='--', linewidth=1)
    axes[1].set_yticks(positions, labels)
    axes[1].invert_yaxis()
    axes[1].set_xlim(min(times - low) - 0.015, max(times + high) + 0.11)
    axes[1].set_xlabel('秒/步；横线表示单次运行内的 P10–P90' if chinese else
                       'Seconds / step; bars show within-run P10-P90')
    axes[1].set_title('步耗时中位数（第 21–40 步）' if chinese else
                      'Median step time (steps 21-40)', loc='left', weight='bold')
    for axis in axes:
        axis.set_ylim(len(rows) - 0.1, -0.6)
        axis.grid(axis='x', alpha=0.18)
        axis.set_axisbelow(True)
    figure.text(0.05, 0.94, '重计算成本：模块策略的实测对照' if chinese else
                'Recomputation cost: measured module A/B', weight='bold', fontsize=18)
    figure.text(0.05, 0.885,
                '历史配方 ec2e16b｜FSDP8｜8×910B3｜CANN 9.2.0-beta.2｜非最新 f637ef7 基点结果' if chinese else
                'FSDP8 | 8 x Ascend 910B3 | CANN 9.2.0-beta.2 | Original BF16-forward Muon recipe',
                fontsize=10, color='#526276')
    figure.text(0.05, 0.035,
                 '每种策略独立运行 40 步；全部 8 个 rank 在此窗口内的 LM/MTP/aux/总损失精确一致。\n'
                 '每策略仅运行一次，分位区间不是置信区间；不替代 200 步验收。模块策略使用已记录目标的实验 wrapper。' if chinese else
                 '40 steps per policy; all 8 ranks have exact LM/MTP/aux/total agreement in this window.\n'
                 'Single-run diagnostics, not 200-step acceptance. Module-only policies use explicit experiment hooks.',
                 fontsize=9, color='#526276')
    figure.subplots_adjust(left=0.14, right=0.98, bottom=0.19, top=0.79, wspace=0.5)
    for extension in ('png', 'svg'):
        suffix = '-zh' if chinese else ''
        figure.savefig(ROOT / 'figures' / f'recompute-module-cost{suffix}.{extension}', dpi=180)
    plt.close(figure)


def plot_events(rows, chinese=False):
    plt.rcParams.update({'font.family': 'Microsoft YaHei' if chinese else 'DejaVu Sans'})
    policy_labels = {'off': '关闭重算', 'full': '完整重算', 'selective': '选择算子'}
    modules = [(f'{prefix}.{name}', f'{label} / {title}')
               for prefix, label in [('model.layers.0', '主干 0' if chinese else 'Trunk 0'),
                                     ('model.layers.1', '主干 1' if chinese else 'Trunk 1'),
                                     ('mtp.layers.0.transformer_layer', 'MTP 0')]
               for name, title in [('self_attn', '注意力' if chinese else 'Attention'), ('mlp', 'MLP')]]
    index = {(row['policy'], row['module'], row['phase']): row for row in rows
             if any(row['module'] == name for name, _ in modules)}
    positions = np.arange(len(modules))
    figure, axes = plt.subplots(1, 3, figsize=(15, 5.6), sharey=True)
    forward = [index[('off', name, 'forward')]['median_inclusive_ms'] for name, _ in modules]
    bars = axes[0].barh(positions, forward, height=0.52, color='#8B9AAF')
    axes[0].bar_label(bars, fmt='%.2f', padding=3, fontsize=9)
    axes[0].set_yticks(positions, [label for _, label in modules])
    axes[0].set_xlim(0, max(forward) * 1.24)
    for offset, policy, color in [(-0.17, 'full', '#165D8D'), (0.17, 'selective', '#32A0A7')]:
        records = [index[(policy, name, 'recompute')] for name, _ in modules]
        values = [row['median_inclusive_ms'] for row in records]
        bars = axes[1].barh(positions + offset, values, height=0.31, color=color,
                           label=policy_labels[policy] if chinese else policy.title())
        axes[1].bar_label(bars, labels=[f"{value:.2f}{'*' if not row['completed'] else ''}"
                                       for value, row in zip(values, records)], padding=3, fontsize=8.5)
    axes[1].set_xlim(0, 24)
    axes[1].legend(frameon=False, loc='lower right', fontsize=9)
    for offset, policy, color in [(-0.23, 'off', '#8B9AAF'), (0, 'full', '#165D8D'),
                                 (0.23, 'selective', '#32A0A7')]:
        values = [index[(policy, name, 'forward')]['median_forward_allocation_delta_mib'] for name, _ in modules]
        axes[2].barh(positions + offset, values, height=0.21, color=color,
                     label=policy_labels[policy] if chinese else policy.title())
    axes[2].legend(frameon=False, loc='lower right', fontsize=9)
    titles = (('关闭重算时的前向调用', '反向中的重放调用', '前向分配量的变化') if chinese else
              ('Forward call, recompute off', 'Replay call', 'Forward allocation change'))
    units = (('含子调用的 NPU 耗时中位数 / ms', '含子调用的 NPU 耗时中位数 / ms', '退出减进入的中位数 / MiB')
             if chinese else ('Median inclusive NPU time / ms', 'Median inclusive NPU time / ms',
                              'Median exit minus entry / MiB'))
    for axis, title, unit in zip(axes, titles, units):
        axis.set_title(title, loc='left', weight='bold', fontsize=11)
        axis.set_xlabel(unit, fontsize=9)
        axis.grid(axis='x', alpha=0.18)
        axis.set_axisbelow(True)
    axes[0].set_ylim(len(modules) - 0.25, -0.6)
    figure.text(0.05, 0.94, '重计算的模块耗时与分配量变化' if chinese else
                'Where recomputation spends time and changes allocations', weight='bold', fontsize=17)
    figure.text(0.05, 0.885,
                '历史配方 ec2e16b｜独立诊断运行｜8 个 rank×第 6–10 步｜FSDP8｜非最新基点结果' if chinese else
                'Separate diagnostic runs | 8 ranks x steps 6-10 | FSDP8 | CANN 9.2.0-beta.2',
                color='#526276', fontsize=10)
    figure.text(0.05, 0.045,
                '* 表示所需张量已生成后提前结束的重放，不是完整前向耗时。模块耗时含子调用，嵌套模块不可相加。\n'
                '分配量差额包含输出与临时存储，不是保存激活的大小，也不是整步显存峰值的节省量。' if chinese else
                '* Replay stopped early after needed tensors were regenerated. Inclusive times are not additive.\n'
                'Allocation deltas include outputs and temporary storage; they are not saved-activation sizes or whole-step peak savings.',
                fontsize=9, color='#526276')
    figure.subplots_adjust(left=0.14, right=0.98, top=0.79, bottom=0.20, wspace=0.27)
    for extension in ('png', 'svg'):
        suffix = '-zh' if chinese else ''
        figure.savefig(ROOT / 'figures' / f'recompute-module-events{suffix}.{extension}', dpi=180)
    plt.close(figure)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--chinese', action='store_true')
    parser.add_argument('--plots-only', action='store_true')
    args = parser.parse_args()
    if args.plots_only:
        data = json.loads((ROOT / 'evidence/recompute-cost.json').read_text())
        plot(data['policies'], chinese=args.chinese)
        plot_events(data['module_events'], chinese=args.chinese)
    else:
        main()
        if args.chinese:
            data = json.loads((ROOT / 'evidence/recompute-cost.json').read_text())
            plot(data['policies'], chinese=True)
            plot_events(data['module_events'], chinese=True)
