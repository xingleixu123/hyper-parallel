# 129 性能证据：本地公开候选

两轮 HSDP/FSDP 顺序 H1→F1→H2→F2；GBS4/LBS1/seq4096，1500 恢复，10 warmup + 30 timed。`data/hf-steps.csv` 保留全部 1511–1540，每步为四 rank 同步时长最大值；两对 H/F 均值比分别 1.498541、1.495846，仅两对、同顺序，不证明因果。生产 checkpoint 的 source 是 33096，执行 source 是 1a，两个身份不可混淆。

SAC 的 `data/sac-steps.csv` 含原策略与 fixed 各两轮，r1 original→fixed、r2 fixed→original。两轮 fixed−original 平均分别 +0.012528、+0.000801 秒；fixed r1 的训练 child exit 0、40步及 post40 finite，但 supervisor 原状态因 proc 权限错误为 failed，不能改写为完整通过。fixed 执行 source 03ea、original 1a；40 步不代替 1000 步验收。

`data/memory.csv` 每 rank 分列 allocator peak allocated、peak reserved 与 final unique persistent storage；图的两种柱子分别为跨 rank 最大 allocated 与去重 storage，不相加，也不是激活峰值拆分。`data/profile-steps.csv` 保留两份独立 profiler 的每 rank×三 active step 整步数据；计算和未重叠通信是 profiler 分类，可有依赖，不能相加或作因果归因。已审摘要的整步 HSDP−FSDP 未重叠通信均值约 316.176 ms；通信 operation duration sum 不能当未重叠关键路径。本归档不对具体 process group 或关键路径作归因，后续 probe 单独记录。

各 observer 原状态不同且部分 incomplete；离散采样不证明整个窗口独占，监控有开销。公开图是本地 CPU 科学预览（ReportLab + PDFium），不是 log.javazero 网站导出；`plot_cpu.py` 可编辑，`traceable.log` 保留原 step。输入摘要和 SHA 在 `data/sources.json`；数据为标量，不包含配置正文、参数、张量或凭据。EP、125 TP 等后续测量另行归档，本图只展示列明的两对样本与独立 profiler。
