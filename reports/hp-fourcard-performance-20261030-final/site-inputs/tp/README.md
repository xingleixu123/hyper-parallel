# TP 网站绘图输入候选（仅本地）

本包由已封存的 `tp-perf-final-public-candidate-v1/SHA256SUMS`（SHA256 `060ffe710cfbf5129d7131d852500fbff1e74880b204addfd9f7cdb685c8af0c`）及已封本地 raw rank 文件生成；只准备 CPU 数据，**未启动网站/浏览器、未外发、未进行新的模型运行**。构建器先核清单所有文件与 10 个原始 rank0/rank 文件的 SHA，再逐点验证源值；`python -B build_inputs.py --verify` 在原本地归档仍存在时可重新从源逐字节验证，不覆盖输出。

## 网站输入

- `data/01-fsdp-loss.jsonl`：历史 1a FSDP4/selective 的 rank0 已规约 `loss`，1501–2500。
- `data/02-tp-loss.jsonl`：TP2/DP2、full recompute、deferred 的 rank0 `loss`，相同 1000 个真实 steps。
- `data/03-zero-guides.jsonl`：`signed_delta=0`、`cumulative_signed_mean=0` 仅为每个真实 step 的数学参考线，不是训练观测。
- `data/04-tp-delta.jsonl`：`signed_delta=TP−FSDP` 与从1501累计的有符号均值；所有点含 1804，未平滑或移除。
- `data/05-tp-sync-perf.jsonl` / `data/06-tp-deferred-perf.jsonl`：**仅 1511–1540 计时 30 步**，每步四 rank 同步 `train_step` 时长最大值，及 `16384/该步时长` 的名义 tokens/s；后者每步值的均值不等于整体名义吞吐 `16384/mean(时长)`。前 10 个 warmup 步不在这两文件中。
- `data/07-tp-sync-allocator.jsonl` / `data/08-tp-deferred-allocator.jsonl`：同 30 步逐步 `memory_after.peak_{allocated,reserved}_bytes`，分别按 rank 0–3 改单位为 GiB，来自各自 rank 当步原始记录；不是用 `memory.csv` 汇总峰值造出的平线。rank 0–3 对应物理卡 4–7。每步会 reset peak，但 reserved 仍可包含缓存；allocator 值不是整卡 HBM、active 或 activation。warm 唯一 storage 是独立快照，未生成伪时间序列。

1000 步差异均值及分布、峰值和 30 步时间均与封存 `metrics.json` 一致，`validation.json` 保存验证结果。千步 loss 对照同时改变并行/重计算/同步策略，不可归因某一项；**0.005 仅约束整窗口有符号均值绝对值，不应绘为逐步 loss 差验收线**。1501–2500 的 `cumulative_signed_mean` 是前缀，不是逐步阈值或 MAE。

## 源行血缘

`lineage.json` 为每个 1000 步写明 derived JSONL 行号、公开候选 CSV 行号、历史/TP 原始 rank0 文件及行号，TP 前40与后960分别对应 perf 和 continuation；synthetic zero 明确标记。30 步计时/allocator 行记录 first40 CSV 行、所有四 rank 原始 perf 同步行及实际耗时最大 rank。`raw_files` 提供原始文件 SHA 与相对 P 路径；source candidate 和 `tp-perf-memory-attribution-max-v1/analysis.json` 分别以封存 SHA 核对。引用 `plot.cjs` 未来输出的 `source-points.csv` 后，可由其中 JSONL 文件/行号连接此血缘，恢复原始来源；本包自身不是原始 tensor 权重。

将四个 loss 文件顺序作为 helper 输入：FSDP、TP loss、zero guide、TP delta；相同字段会产生主图与额外差值图，zero guide 为零时网站相对误差不可靠。性能与 allocator 用各自两个文件在独立页面查看，避免把历史 FSDP loss 参考误当同期性能基线。按照 `plot-training-logs`，未来若执行绘图，应保持可见 Edge 在线窗口，另用相同输入和新空目录独立运行 `--headless` 批量导出并复核结果；本准备阶段不运行。

作者模型为 GPT-6.1 Terra max；无可选择的 Fast selector。

`node parser-preflight.cjs` 在 CPU 本地检查固定 helper SHA、所选指标和行覆盖，并给出纯本地输入计划；`parser-preflight.json` 保存此准备时的输出。计划中的 `original` 仅指网站将接收本包派生 JSONL 原字节，不是原始训练日志；网站 worker、画面与联网行为尚未运行或验证。
