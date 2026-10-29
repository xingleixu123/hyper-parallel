# 四卡重计算性能首轮观测（草稿）

这两张图是已复核的本地 CPU 绘图预览，**不是 log.javazero.top 导出，也不是最终性能验收**。
三组都已真实完成 NPU 更新1501–1540；1501–1510预热，1511–1540计时。每策略目前只有一个样本。
复用同一个 FSDP 第1500步完整训练状态；源码1a39572f、GBS4/LBS1、序列4096和原2500步日程固定。

- [逐步耗时、吞吐与峰值](plots-recompute-cpu-preview-v2/01-per-step.png)
- [吞吐与 allocator 峰值权衡](plots-recompute-cpu-preview-v2/02-tradeoff.png)
- [90行原始汇总值](joined.csv)、[版本/口径/关键统计](summary.json)、[文件 SHA256](manifest.json)
- measurement-v2/ 保留三组原始四 rank JSONL及对应30步aggregate CSV；均与本地源文件逐字节一致。
- [绘图源码](plots-recompute-cpu-preview-v2/plot_cpu.py)：在保留当前目录结构的情况下，安装ReportLab、pypdfium2、Pillow后运行即可重建PNG/PDF。

步耗时是同一步四rank本地同步计时的最大值，不是跨rank全局wall-clock；窗口吞吐为30×16384/各步最大耗时之和。
图二误差线为本run内30步min–max，不是重复实验置信区间。图一不平滑、不降采样。
仅有设备占用的起点/终点证据，不能证明整个窗口连续独占。不能凭该样本排名认定策略普遍优劣或实现缺陷。
allocated/reserved为框架allocator口径，不等于物理HBM。当前基线未包含SAC注册修复，不能将图中差异当作修复收益。

配置YAML按用户要求保留于私有归档，这里只发布图、真实逐步数值和必要统计元数据。
调查与待验收项：[总 issue](https://github.com/hf1s/hyper-parallel/issues/304)；
[SAC策略覆盖 issue](https://github.com/mindspore-ai/hyper-parallel/issues/1357)、[Draft PR](https://github.com/mindspore-ai/hyper-parallel/pull/1358)。
