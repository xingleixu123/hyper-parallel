# [草稿] 已发布：FSDP no-sync 两卡 NPU 复核材料

本补充已由任务负责人审核并随本 Git 提交公开，关联 [issue1368](https://github.com/mindspore-ai/hyper-parallel/issues/1368) 与 [Draft PR1369](https://github.com/mindspore-ai/hyper-parallel/pull/1369)。[原始结果与封存说明](README.md)中“候选、尚未发布”的文字记录的是发布前封存时的状态；原文及清单保留原字节。本页为发布入口。

实际 Ascend910B3、两 rank HCCL、FP32 SUM、两种 fusion（zero-copy 关闭）：原头12/16、sole scheduler补丁16/16。任务负责人重新运行纯CPU有理数验证器复核32份真实记录的初值、实际输入、逐micro loss与梯度；这不替代NPU运行收据，也不证明整网性能。两次有效运行间gate TTL180→600且停止三个旧闲置容器，因此不作性能因果A/B。前次green-v1超时无数值记录，另行保留。

本次源固定5cde7c54；之后上游8b187eca的参数视图修复正在独立评估，不能将本材料改称新master测试。本目录保持草稿评审范围：HSDP partial-only、跨rank不同控制路径、默认zero-copy、真实模型收益尚不由此证明。

54个原候选文件的哈希见 [SHA256SUMS](SHA256SUMS)，本发布页由发布回读清单单独核验；全部55文件按Git blob原字节回读。完整进程身份/环境/设备映射原件和闭合归档留在任务私有证据目录；本补充不含YAML、真实样本或进程映射。
