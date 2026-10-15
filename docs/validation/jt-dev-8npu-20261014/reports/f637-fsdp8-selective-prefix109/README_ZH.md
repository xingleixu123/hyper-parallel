# 选择性重算：109步前缀与让卡记录

**完整200步验收未完成。** `r129-f637-fp32-fsdp8-selective-200`八rank均只有1–109步；在卡4出现外部实验后，守卫只停止本任务容器，exit137、270.352861秒运行后释放，无本任务PID残留。原始严格比较保持FAIL；绘图副本明确标为INCOMPLETE_OBSERVED_PREFIX，不改变验收。

共同已观测前缀中LM、日志MTP/aux、按0.3/0.0001系数还原MTP/aux、总loss及梯度范数最大差均为0；输入、初态和配方一致。尚缺每rank后91步，不由此判断最终性能收益；末尾资源记录可能受争用影响。还原loss不是另行采集的加权前张量，梯度范数不是完整梯度张量。

[严格未完成结果](f637-fsdp8-selective-prefix109-strict-incomplete.json) · [分析](f637-fsdp8-selective-prefix109-analysis.json) · [数值独立审计公开副本](f637-fsdp8-selective-prefix109-independent-review-public.json) · [归属与释放独立审计](f637-fsdp8-selective-interruption-independent-review.json) · [公开文件清单](public-manifest.json)。

68份本地采集文件大小/SHA全通过。66份运行文件原字节公开；原始调度回执及身份事件涉及其他实验进程，仅保留本地，其SHA仍在原清单内。回执、协调器和Spec审计的公开副本已明确注明匿名化，未修改数值或成功/失败状态。其他任务的进程身份不属于本任务公开证据的必要内容。两端一次性归档已清理。

后续`selective-200-r2`使用固定镜像、源码及原配置数值文本，只替换未启用checkpoint输出目录；实际YAML、Trainer及模型配置解析均通过，未初始化NPU，CPU容器退出0并删除。下一轮须同时满足释放后1200秒墙钟/monotonic门槛（最早2026-10-16 02:54:53.613277，UTC+8）并重新检查空闲卡。

[最新首差定位计划](precision-localization-plan-v6.md)只记录候选解释，不把未证实归因作为新缺陷提交。
