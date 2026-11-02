# 四卡 1501–2500 数值与性能报告

**跨框架算法边界：**单卡 MF/HP 的完整窗口数值达到既定条件，并不能证明两框架 Muon 的参数分组、矩阵切分和更新算法等价；相关分组选择仍在确认。四卡以 HP 自身连续 FSDP 为参照，不能消除这一限制。

同一闭合 FSDP step1500 原生 checkpoint 为恢复起点，模型为 Xing40 配置的 3 层（2 Dense + 1 MoE）、序列4096、GBS4/LBS1、每次更新16,384有效 token，MTP关闭（N/A）。冻结 HP source `2dd33e02f46fc5d6810e806b9d7d4717cf10a3e4`，来自 master `4144bb242f1a696308e75bd609cb61c9143b3d8f`、本轮六 PR 和单独 CE兼容提交；本报告的1000步原始数据不包含后续 Muon checkpoint 清理修复或 CP Ulysses2 候选。

每个已闭合实验均比较 1501–2500 的完整1000次更新。表中差值为各配置 rank0 的 native global loss 减125主机连续 FSDP rank0 native global loss；CP/TP 副本 rank 的 global loss 不重复求和。既定数值条件是**完整窗口最终** `abs(mean(diff)) ≤ 0.005`，不是每个 step 的差值阈值；MAE 和最大绝对差用于诊断。原始逐步值见 [fourcard-loss-1501-2500.csv](fourcard-loss-1501-2500.csv)。

| 配置 | signed mean | abs(mean) | 正 / 负 / 零 | MAE | max abs（step） |
|---|---:|---:|---:|---:|---:|
| FSDP4 自恢复 | -3.56531143e-06 | 3.56531143e-06 | 513 / 485 / 2 | 0.000300830364 | 0.00223207474（1534） |
| CP2×DP2 | -4.6339035e-06 | 4.6339035e-06 | 492 / 507 / 1 | 0.000297410011 | 0.00225448608（2163） |
| EP2×DP4 | -1.93595886e-05 | 1.93595886e-05 | 474 / 525 / 1 | 0.000285603523 | 0.00245189667（1534） |
| HSDP2×2 | -7.17258453e-06 | 7.17258453e-06 | 489 / 511 / 0 | 0.000286643982 | 0.00173807144（2389） |
| TP2×DP2（FFN replicate） | -2.88155079e-05 | 2.88155079e-05 | 473 / 527 / 0 | 0.00034701848 | 0.00506782532（2446） |
| FSDP4 + selective AC | -1.67369843e-05 | 1.67369843e-05 | 467 / 532 / 1 | 0.000292843819 | 0.00165557861（1534） |

TP 只在 attention、embedding 和 head 上使用 TP2，Dense 与 MoE/FFN 边界为 replicate，使用框架标准 `loss_parallel`；不得将该结果描述为完整 FFN TP。selective AC 一步诊断实测覆盖三个 decoder layer 和 `model.hc_head.input_norm` 四个包装器，见下文；不能单凭 loss 或性能推断覆盖。

## 979步带观测时序

以下为1521–2499，每步取4个 rank 的最大 elapsed，tokens/s = 16,384 / 该时间；以线性分位数给 IQR/p95。原训练内有额外实际样本、prepared 输入和 scalar 观测，结束同步在计时内，步间 fsync 不在其中。allocated 和 reserved 是不同 allocator 视角，不相加。125 与129主机/driver不同，表中跨主机数值不作为受控性能比值。

| 配置 | 主机 | median 秒/更新 | IQR 秒 | p95 秒 | median tokens/s | 最大 rank peak allocated / reserved GiB |
|---|---|---:|---:|---:|---:|---:|
| FSDP 连续 | 125 | 0.618751 | 0.004060 | 0.624210 | 26479.1 | 13.491 / 15.580 |
| FSDP 自恢复 | 129 | 0.619548 | 0.004654 | 0.625567 | 26445.1 | 13.491 / 15.580 |
| CP2×DP2 | 129 | 0.967195 | 0.005378 | 0.975133 | 16939.7 | 14.799 / 16.785 |
| EP2×DP4 | 125 | 0.634827 | 0.003894 | 0.640082 | 25808.6 | 11.966 / 15.959 |
| HSDP2×2 | 125 | 1.077568 | 0.007706 | 1.086927 | 15204.6 | 18.730 / 23.711 |
| TP2×DP2（FFN replicate） | 129 | 1.400695 | 0.002891 | 1.404687 | 11697.0 | 16.460 / 18.668 |
| FSDP + selective AC | 125 | 0.613311 | 0.004088 | 0.619021 | 26714.0 | 13.746 / 15.875 |

## 129 同机短窗 ABBA 观测消融

每臂从同一 step1500 恢复，1501–1520 热身、1521–1570 采50步，顺序 A1/B1/B2/A2。A 启用额外 prepared 张量校验和 micro 指标读取；B 关闭这两项，仍保留真实 CPU raw 样本核验、native 训练和相同计时边界。B 没有逐 micro prepared/额外指标证据。8臂均 controller rc0、holder removed，四rank账本闭合；[abba-steps-1521-1570.csv](abba-steps-1521-1570.csv) 保留每步4 rank 时间、max、吞吐和 rank0 loss。

| 129 布局 | A1 秒 | B1 秒 | B2 秒 | A2 秒 | A两臂中位数均值 − B两臂均值 |
|---|---:|---:|---:|---:|---:|
| FSDP4 | 0.618835 | 0.617625 | 0.617754 | 0.619069 | +1.263 ms（+0.204%） |
| CP2×DP2 | 0.965251 | 0.960730 | 0.963709 | 0.966843 | +3.828 ms（+0.398%） |

这里 A−B 是两臂中位数均值之差，仅描述该同机短窗；50步相邻样本相关，离散卡进程采样不能排除所有背景干扰，不能把差值直接认作生产环境的全部观测开销，也不是1000步数值验收或无插桩吞吐。

## 证据范围

- FSDP、CP、EP、HSDP、TP 与 selective 均有四rank完整1000步；配置/阶段说明和原始统计见 [statistics.json](statistics.json)，来源文件 SHA 与口径见 [manifest.json](manifest.json)。这些小型公开文件不包含私有 YAML、训练数据或完整配置。
- HSDP 和 TP 保存时观察到的**非 owner** momentum 副本同步变化，已按实际 owner 与保存前权威值核对；这不等同于权威参数、owner 状态或组发生变化。step2500 不在本报告中声称经过独立全局 CPU tensor 读回；恢复起点 step1500 已有独立验证。
- 此四卡结果使用 HP native FSDP 连续轨迹作参照，并非 MF/HP 跨框架比较。单卡 [MF / HP 报告](https://github.com/xingleixu123/hyper-parallel/blob/10af321dbd34ac5f728d3a1fbead66cd0f153e72/reports/hp-mf-sixpr-cann92-20261102/README.md) 保留 MF 训练调用返回但终态 maps guard 误判 IPC 的条件性边界，不称 MF 正常退出。
- CP Ulysses2 未纳入本报告；各 2500 checkpoint 全局 CPU 读回仍待单独验证。后续 checkpoint 清理修复的70步与 tiny 原生读回是独立诊断，见 [checkpoint 状态证据](../muon-checkpoint-state-v1/README.md)。



## Selective AC 一步实际覆盖

同一 step1500 恢复的独立 step1501 诊断，每rank发现四个 `HSDPCheckpointWrapper`：三个 decoder layer 与 `model.hc_head.input_norm`；每个包装器的被包装模块 pre-hook 各触发两次。每rank native policy 委托记录中，前向阶段 `MUST_RECOMPUTE` 249、`MUST_SAVE` 14；重算阶段分别247、14。计数是函数策略事件，不是 kernel 性能。实际 restore proof 与原 full SAC 逐字节相等，真实 raw 样本与 RNG before 一致；B观测的 prepared 和 micro metrics 未采集。四rank loss 和计数见 [单步摘要](sac-one-step-summary.json)。

原 collector 对该子类给出 count 0 是类型识别问题，不代表未执行重算。已有官方 [PR #1315](https://github.com/mindspore-ai/hyper-parallel/pull/1315) 和 hf1s [issue #229](https://github.com/hf1s/hyper-parallel/issues/229)、[PR #230](https://github.com/hf1s/hyper-parallel/pull/230) 相关；不另建重复问题。此单步覆盖证据不替代上方1000步数值窗口。
