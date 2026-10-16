"""Draw the four complete independent trajectories against the fresh MF run."""
from pathlib import Path
import json
import csv
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

ROOT = Path(__file__).resolve().parents[1]
FONT = FontProperties(fname=os.environ.get('JT_PLOT_FONT', 'C:/Windows/Fonts/msyh.ttc'))
RUNS = {'hp-product200': '原生基线 + PR213', 'hp-fused200': '再加融合矩阵修复',
        'hp-archive200': '兼容首版（舍入顺序错误）', 'hp-archive200-r2': '正式兼容修正版'}


def values(run, family):
    result = {}
    for line in (ROOT / 'runs' / run / 'metrics.jsonl').read_text().splitlines():
        item = json.loads(line)
        if family == 'hp':
            parts = item['components']
            row = (item['loss'], parts['foundation_loss/mtp'] / .3, parts['foundation_loss/aux'] / 1e-4)
        else:
            # Match the authoritative numerical auditor's MF field conventions.
            from audit_clean_runs import metrics
            return metrics(ROOT / 'runs' / run / 'metrics.jsonl', 'mf')
        result[item['step']] = dict(zip(('total', 'mtp_raw', 'aux_raw'), row))
    return result


def main():
    reference = values('mf-reference200', 'mf')
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    metrics = [('total', '总 loss'), ('mtp_raw', '未加权 MTP loss'), ('aux_raw', '未加权 aux loss')]
    colors = ['#8C8C8C', '#E69F00', '#D55E00', '#0072B2']
    records = []
    for (run, label), color in zip(RUNS.items(), colors):
        actual = values(run, 'hp')
        assert set(actual) == set(reference) == set(range(201, 401))
        for axis, (key, title) in zip(axes, metrics):
            error = [abs(actual[step][key] - reference[step][key]) for step in range(201, 401)]
            axis.semilogy(range(201, 401), [max(v, 1e-8) for v in error], label=label, color=color,
                          linewidth=1.2 if run != 'hp-archive200-r2' else 1.6, alpha=.9)
            axis.set_ylabel(title + '\n绝对误差', fontproperties=FONT)
            records += [{'run': run, 'step': step, 'metric': key, 'absolute_error': error[step-201]}
                        for step in range(201, 401)]
    for axis in axes:
        axis.axhline(.005, color='#B00020', linestyle='--', linewidth=1.2, label='逐步门槛 0.005')
        axis.grid(alpha=.2)
        axis.set_ylim(7e-9, 3)
    axes[0].legend(prop=FONT, ncol=3, fontsize=9, loc='upper left')
    axes[-1].set_xlabel('续训全局步数（201–400）', fontproperties=FONT)
    fig.suptitle('最新 JT_dev 干净复跑：四个完整对照与同一次 MF 参考', fontproperties=FONT, fontsize=16)
    fig.text(.5, .012, '零误差仅为对数绘图显示在 1e-8；审计及 CSV 保留真实值。不同背景的最大误差变化不能相加为贡献率。',
             ha='center', fontproperties=FONT, fontsize=9)
    fig.tight_layout(rect=(0, .035, 1, .96))
    fig.savefig(ROOT / 'evidence/clean-200-loss-errors.png', dpi=170)
    with (ROOT / 'evidence/clean-200-loss-errors.csv').open('w', newline='', encoding='utf-8') as output:
        writer = csv.DictWriter(output, fieldnames=['run', 'step', 'metric', 'absolute_error'])
        writer.writeheader(); writer.writerows(records)
    print('Saved 2400 error records and three-panel figure.')


if __name__ == '__main__':
    main()
