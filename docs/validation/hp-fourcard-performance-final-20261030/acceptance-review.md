# [草稿] 固定四卡性能任务收尾复核

截至2026-10-30，保留原始数据、私有YAML、关键checkpoint/source及profiler；用户最新要求暂停画图，因此网站输入和页面保留，未继续导出或绘制。

1. root独立复算14个实际NPU run：4rank×40更新；10步预热后30步计时，共420条跨rank最大耗时、56条逐卡显存记录。与公开CSV/JSON一致，无删除尖峰。窗口均值/CV并不构成长时间统计平稳性检验。
2. 配置SHA/模型字典逐run核验：模型、optimizer、精度、4096序列和2500步日程一致，仅10个被准许的并行、重算、运行隔离字段路径有变化。实际DP/TP/CP rank映射、微批次数与GBS4/LBS1对应关系已核。128条首两步样本记录逐字段hash与6000–6007参考一致。样本报告未逐字段记录shape/dtype，依赖固定运行时断言及checked标记；不能声称1503–1540均有完整字段hash。
3. 新增[四卡HSDP FP32绝对梯度对照](hsdp-normalization/README.md)关闭当前后端RS AVG数值语义缺口。该toy验证不修改模型配置或生产代码，不替代实际训练；48项梯度比较和独立通信算术通过。
4. 已确证的SAC注册缺陷已有中文[issue1357](https://github.com/mindspore-ai/hyper-parallel/issues/1357)和[Draft PR1358](https://github.com/mindspore-ai/hyper-parallel/pull/1358)。CPU、default NPU module/replay、短测与1000步窗口回归保留。attention无checkpoint梯度重复本身存在波动，严格梯度等价仍inconclusive；compile和GMM.List的独立设备验证尚未完成，不声称通用路径或CI全部通过。
5. 1000步修复−原版的abs(mean ΔLOSS)=0.0000907969475≤0.005；step1804最大绝对差0.07693529129完整保留，根因未知。统计通过不等于解释异常，也不等于MF/HP新验收。不同host的千步耗时不能拿来判断补丁加速。
6. HSDP有实际持久storage增加、部分Muon通信标签及RS→AR流水证据；pack/FQN和完整critical path尚未闭环，不能把约1.50倍耗时归为单一缺陷。保守TP仅部分attention分片，DP2导致每rank两micro，CP参数仍实采[4,1]mesh；这些约束须随结果解释。CP/EP非同时参考、离散占卡与I/O未知边界均保留。

最新检查上游master96143af9…、Draft1358 head8104a78a…未变；没有发现需要重跑已固定基线的上游变更。当前报告给出有边界的性能筛选与已确认问题修复，不声称所有数值/性能未知已经解决。若进一步主张attention严格等价、1804根因或HSDP完整因果，须分别设计同状态/同设备或关键路径实验，不能从曲线更接近或平均门限通过推断。

已实际委派GPT-6.1 Terra medium复核日志、恢复/配置、理论与交付；接口没有Fast选择器，一次host候选委派遇model capacity，root接管并保存未采用候选。网站20份输入、8960个scalar及配置记录保留；网站最终CSV/PNG/分享链接仍待用户恢复后验证，已有PNG仅为本地科学预览。

新增：收尾发现的整数序列GMM接口兼容问题，已经同字节Tensor控制、CPU红绿及八条件NPU验证；见[GMM独立报告](gmm-sequence/README.md)。原生List overload未进入SAC的失败和设备枚举失败完整保留，不与已确认的SAC注册缺陷混合。
