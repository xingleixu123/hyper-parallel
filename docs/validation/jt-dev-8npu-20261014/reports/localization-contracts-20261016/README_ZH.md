# 具体问题定位与最新参考实现契约

本轮为14组已确认缺陷或可选接口需求补充具体函数、固定提交行锚、触发条件和两侧正确性依据。数值、重算及优化器另对照核验时最新的MF官方GitCode master `1c9258ca845e21ddd66fde83e262d5635359bd30` 和Megatron官方main `3e2807ce5e787f81b7275be2383092fe0c3bfc71`，并跟踪其实际Muon依赖；MF GitHub镜像当时尚未同步。没有把源码相似、局部消融改善或同实现分片对照当作完整数学正确性/200步通过证明。

- [最新六类正确性契约](latest-reference-correctness-map-20261016-v1.md)：目标与分母、TP/CP输入映射、Linear前向/dX/dW、归约输出与主梯度、Muon数值配方、重算持久副作用。
- [43文件、211函数/类与54固定引用索引](latest-reference-correctness-map-20261016-v1.json)及[root字节/声明/引用范围复核](latest-reference-correctness-root-review-20261016-v2.json)。它们是源码读取和契约核对，不是跨框架训练实测。
- [14组问题的更新前缺口审计](issue-localization-evidence-audit-20261016.md)及[完整定位JSON](issue-localization-evidence-audit-20261016.json)。该快照中的“参考pending”和旧body指纹描述审计时状态；最新参考已由上面的六类契约补齐。已合入#230/关闭#229保留合并前正文，当前合入状态以总单和GitHub实际状态为准。
- [从消融到具体根因的执行约束](precision-causal-bisection-plan-20261016.md)：双侧数学参考、首次可信分歧、局部因果干预、反例/修复回归，再做完整200步。普通Linear FP32 dX/dW已有失败前缀，head padding三步效应不一致，均不升级为已确认修复。
- [13组已审阅中文issue/PR正文归档](descriptions/README_ZH.md)：每组包含具体函数行锚及正确性边界；[主体审核](description-review/body-review-v1.md)与[最终参考增补审核](description-review/final-review-v2.md)均通过，实际远端写入另以读回确认。
- [本轮修改说明前的current-head CI快照](github-current-head-gates-20261016-143506.json)：14个head均一个commit；0 statuses/0 checks，不能声明CI通过。后续正文更新与实际状态由GitHub读回另核验。

当前先完成全部200步、8rank主LM loss最大绝对差≤0.005；主loss全矩阵通过后先汇报，再由用户决定MTP/aux专项。历史18602c4的基线、续训和重算通过结论保持原范围；最新544526f7目前没有共享机匹配参考和跨机准入通过结果。总问题单：[hf1s/hyper-parallel#202](https://github.com/hf1s/hyper-parallel/issues/202)。

除[明确记录的外部占用标识脱敏](publication-redactions.json)外，归档保留原报告字节和历史证据链接。脱敏只替换他人cgroup/PID namespace标识子树，数值、步数、退出与中断判定不变；原文件和原子树SHA保留以便本机审计。JSON中的本机路径用于标识原始检查位置；最新报告的JT/MF/Megatron源码链接均固定到实际提交。历史报告中本机链接保留其原始语义，不能作新的公开运行证据。
