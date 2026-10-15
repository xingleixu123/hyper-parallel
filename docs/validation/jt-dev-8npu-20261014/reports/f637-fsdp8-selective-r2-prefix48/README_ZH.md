# 选择性重算第二轮：48步前缀及让卡

**完整200步验收未完成。** `r129-f637-fp32-fsdp8-selective-200-r2`从step0重新开始，八rank各1–48步；在5号卡出现外部进程后仅释放本任务，exit137、190.053905秒，剩余本任务PID为零。上一轮109步与本轮48步是独立前缀，不能相加或拼接。

LM、日志及按系数还原MTP/aux、total和grad_norm最大差均为0。输入、48项初态、计数、LR及数值配方匹配。严格结果保持FAIL：缺少后152步、完成标记和运行内EXIT_CODE。两项独立审计确认数值、归属、释放证据，未将审计PASS改写成训练PASS；不据中断数据发布性能收益。

[严格未完成比较](f637-fsdp8-selective-r2-prefix48-strict-incomplete.json) · [数值独立复算](f637-fsdp8-selective-r2-prefix48-independent-review.json) · [归属与释放独立复核](f637-fsdp8-selective-r2-interruption-independent-review.json) · [公开文件清单](public-manifest.json) · [摘要](analysis.json)。

全部68份原始文件SHA/字节数核验通过，66份运行文件原样公开；涉及外部身份的两份调度原文仅本地保留，公开明确标注的匿名回执与协调器副本。两端重复归档已删除。

r3保持数值文本，仅调整未启用checkpoint输出路径，实际镜像/YAML/Trainer/模型解析通过，无NPU映射或初始化，CPU容器已删除。下一次同时满足墙钟2026-10-16 03:22:26.767554（UTC+8）、monotonic2968540.593及空卡检查。为减少短暂空档内反复启动，本任务另观察每60秒的空闲采样，累计10分钟后再交给原守卫复查；采样间是否持续空闲未知，不是设备预约或隔离保证。
