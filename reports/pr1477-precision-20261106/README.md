# PR1477 公开审计证据

单卡完整 1501–2500 已通过统计、实际样本、干净退出和独立 CPU 最终状态验收；独立 router/boundary NPU 探针 12/12 配对通过。四卡完整训练矩阵仍待独立验收。本目录归档已复核的原始数值、检视处置和实验边界。

- [单卡完整结果与准确口径](native-single-results.md)、[1000 行三指标对照](single-1501-2500-pairs.csv)、[根任务复算](single-root-statistics.json)、[最终验收](single-acceptance.json)。精确源码为原 [#1477](https://github.com/mindspore-ai/hyper-parallel/pull/1477) 加 #1420/#1470/#1441，source manifest `49bffc6761d24e79ab59af1758b028145ba2bcc3c09030fec7b840cc9e005f85`，1542文件；不称裸PR1477。目标为 world1 legacy 微批等权。
- [12 配对 NPU 表与样本覆盖](router-boundary-results.md)、[CSV](router-boundary-pairs.csv)、[状态](router-boundary-status.json)。只有 TP2/EP2/SP 的半序列目标需要修复；两个完整序列场景 AUX/梯度 exact不变。candidate 四rank结果已保存，但退出监控停止 wrapper 为−15，原控制器异常保留；没有整体controller exit0或完整MoE/FSDP/1000步声明。
- [44线程处置清单](review-disposition.md) / [148项原timeline与全部comment IDs](review-disposition.json)：44线程、82 inline评论、50 review事件、16 Conversation；远端线程未resolve。最新独立跟进单列，不覆盖原head的历史结论。
- [Draft #363](https://github.com/hf1s/hyper-parallel/pull/363) / [两行diff](activation-policy/activation-policy.patch)：关联1456，仅fail-fast拒绝非空未支持策略；空策略112字段保持，不等于实现策略。原1477和完整单卡组合没有该增量。
- [Draft #364](https://github.com/hf1s/hyper-parallel/pull/364) / [固定身份](router-statistics/pr.json)：关联338、对应T40/T44，按实际resolved输入是否Shard(sequence)选择CP/TP统计组并移出CP适配；不合并独立DP样本，不改bias counts。与355/1441的MTP倍率问题独立，原1477和完整单卡组合没有该增量。
- [CPU语义证据](semantic-probe-results.json) / [脚本](semantic_probe.py) 保留实际源SHA与单rankcollective限制；[MF2500三指标](mf-baseline-2500-three-losses.csv) 保持原raw口径，不平滑、不插值、不混入加权目标。

恢复、完整1000步及最终CPU读回已经取代旧a3零更新状态；[npu-status.json](npu-status.json)只把a3作为历史失败记录保留。vendor原挂载路径问题已纠正，不把旧V2符号错误解释为CANN缺算子。

后续四卡必须另锁定目标/梯度和全局AUX采集。旧四卡仅rank0最后微批的.6g AUX显示值不能作为全样本参考；本轮rank-local诊断也不自动当全局日志。

本公开包没有任何私有YAML、完整配置字段映射、凭据、训练payload或checkpoint。manifest与SHA256SUMS覆盖全部材料，执行 `python -B verify_public_audit.py` 只核公开材料，不启动训练或NPU。来源与有限变换见 [package-provenance.json](package-provenance.json)。
