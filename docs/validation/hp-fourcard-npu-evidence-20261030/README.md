# NPU 性能与 SAC 诊断证据

本包是供复核的草稿证据归档。模型源码固定 `1a39572fef23819a2d2651b4bd53047a05a4892f`；环境为 CANN 9.2.0、PyTorch 2.12.0+cpu、torch_npu 2.12.0.post3。配置文件、模型权重、checkpoint 和完整运行环境记录不随包公开。`manifest.json` 固定包内文件哈希及所引用配置的哈希；哈希标识来源，不公开配置内容。

- [CP2 审计](cp2/审计.md)、[聚合 JSON](cp2/aggregate.json) 与 [CSV](cp2/aggregate.csv)：125 上一次单样本描述性结果，warmup 1501–1510、计时 1511–1540；每步取四 rank 本地时长最大值，名义 token 数用于吞吐。没有同条件配对重复，不能给普遍性能排序或因果归因。
- [TP2 conservative 审计](tp2/审计.md)、[聚合 JSON](tp2/aggregate.json) 与 [CSV](tp2/aggregate.csv)：同一数据窗口的单样本；10 条额外 PID 查询区间与计时窗口重叠，因此整个样本标为污染并排除排名。离散查询不能证明其余时间连续独占，也不能推定额外进程身份或影响大小。与自动专家分片 TP 是不同策略。
- 两个目录各有四个 `rank-N-perf.jsonl`；每 rank 原始40步记录未经改动。聚合 JSON 为便于分发修改了原机器绝对路径的副本，数值未重算；原 aggregate 哈希保存在 manifest 和 JSON 的注释中。完整原始占用日志及其他运行证据保留于原归档，未装入本包。
- [SAC v3 数值审计](sac/actual-v3/报告.md) 与 [结果](sac/actual-v3/result.json)：GMM 合成模块比较记录逐位一致；attention 无 checkpoint 基线自身梯度不逐位一致，严格结论 **inconclusive**，不能以逐位相等判定 attention 数值验收通过或失败。这不是整网、optimizer 或性能验收。
- [SAC replay 审计](sac/actual-replay-v2/报告.md) 与 [结果](sac/actual-replay-v2/result.json)：记录 below-SAC dispatch 的阶段与次数；它们不是 CANN kernel 执行次数或耗时。[CPU replay fixture 输出](sac/cpu-fixture/stdout.log) 和 [结果](sac/cpu-fixture/result.json) 说明缓存拦截的受控测试范围。

包内不包含配置 YAML、checkpoint、权重、完整 Docker inspect、supervisor 或命令参数转储；不从本包推断未记录的设备连续独占、数值接受或其他环境的性能。审计 Markdown 为相对引用整理副本；被引用而未随包提供的完整日志和复核程序保留在原归档。
