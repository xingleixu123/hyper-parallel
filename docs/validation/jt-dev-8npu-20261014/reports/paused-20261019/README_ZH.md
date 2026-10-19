# JT DeepSeek V3 阶段收尾：已暂停，问题与修复状态

2026-10-19，按用户要求暂停。**不再启动实验；Goal 与已有接续自动任务均为 PAUSED。** 本次收尾仅收集已经自然结束的记录、独立复核并整理问题单及 PR。125、129 本任务最后两个训练容器已结束、释放并移除，没有停止他人的任务。

本阶段只验功能和相对纯八卡 FSDP 的主 LM 自洽；完整窗口为 200 步 × 8 rank，要求每一点绝对差不超过 **0.005**。性能、显存收益和重算性价比已按用户指示暂停；MTP/aux 保留既有日志观察，不宣布专项全矩阵通过。201 机器已取消。

## 首要未解决问题

**跨并行精度尚未闭环。** 当前集成源 `59b25cdd15505a4f6fb1990484d4e890e3d3724c` 的 EP4 与同源 FSDP8 均完整执行 200 步，但主 LM 最大绝对差 **0.010309219360351562，step 146**，超过 0.005；首次超阈值为 step 97，共 8 个超阈值步骤、64 个 rank-step 点。因此状态仍为 **FAIL**。[200 步完整证据及曲线](../a1e6-muon-no-cross-param-200/README_ZH.md)。

![当前59b同源FSDP与EP4完整200步：精度仍未通过](../a1e6-muon-no-cross-param-200/figures/new59b-main-lm200.png)

[问题 #250](https://github.com/hf1s/hyper-parallel/issues/250) / [PR #251](https://github.com/hf1s/hyper-parallel/pull/251) 提供可恢复的 Muon NS 跨参数合批开关，属于已明确定位的策略配置能力缺口。**该 PR 不是整个精度漂移的完整修复。** 产品 PR 裸 head `b32bded51f2ff352ff09e97270815b63d1c36620` 基于 `JT_dev a1e6c222`；本次 NPU 数值是集成源 `59b25cdd` 的结果，两者不能互换。

新的原生专家 G0 三步比较已经完成并独审：观测器与原 EP3 的十字段、输入和全部捕获阶段等价；比较的是首步实际 post-reduce-scatter、pre-norm/pre-clip 专家梯度。32 个矩阵共 377,487,360 个 FP32 元素，其中 325,468,098 个元素有数值差；最大绝对差的矩阵范围为 `1.4551915228366852e-10` 至 `1.210719347000122e-8`。这证明该边界已有很小的差异，**不能单凭此结果唯一归因于 RS、GMM 或 NS**，也没有证明 200 步已经修复。[原生 G0 详细报告、32 矩阵图与可复核数据](ep-native-g0/README_ZH.md)。

此前仅统一首步 norm 的三步受控实验恢复了非专家 G1/P1 的 336/336 个匹配项，专家仍为 0/32；主 LM 局部变化有改善也有恶化。它不能证明 norm 是唯一根因。[范数干预证据](../a1e6-norm-only-firststep-3/README_ZH.md)。未定位的其他 TP/CP/EP 漂移仍汇总在[总 issue #202](https://github.com/hf1s/hyper-parallel/issues/202)，不冒充已经确认的独立修复。

## 已完成与尚未完成

| 来源与范围 | 已取得证据 | 结论边界 |
|---|---|---|
| 历史集成源 `544526f7`：FSDP 0–200、checkpoint 续训 201–400、独立连续 400 | 两个对应 200 步窗口均覆盖全部 8 rank，主 LM 和所记其他字段最大差为 0 | 历史验收成立，不能代替新 `59b25cdd` 的续训验收 |
| 历史 `544526f7`：FSDP off/full/SAC selective | 各完整 200 步 × 8 rank，主 LM 最大差为 0 | 仅固定配方和源码；不能扩大到所有拓扑的重算 |
| 历史 `544526f7`：TP/CP/EP 及所测组合 | 完整运行与曲线已经公开，多项主 LM 超过 0.005 | 功能完成不等于精度通过；保留各报告的具体数值和未定位结论 |
| 当前 `59b25cdd`：129 纯 FSDP8 200 步与 EP4 200 步 | 两次普通训练完整且释放，EP4 最大主 LM 差 0.010309219360351562 | EP4 精度 FAIL |
| 当前 `59b25cdd`：125 从初态独立连续 400 步 | 400 × 8 全记录完整，十字段各 3,200 点有限；前 200 步对 clean 129 FSDP200 的十字段、输入、捕获初态均相同 | 连续训练功能与共同前窗比较通过；**不是 checkpoint 恢复训练** |
| 当前 `59b25cdd`：125 checkpoint 续训 201–400 | step200 文件保留，续训控制候选未封版、未部署、未执行 | **未验证**，不得把候选、本地检查或文件 SHA 相同写成通过 |
| 当前 `59b25cdd`：其余 TP/CP/EP/组合与重算矩阵 | 尚未完成全部新源码窗口 | 暂停，不能继承历史通过 |

新的连续 400 步独审原件 SHA256 为 `9c4521a661edcc060fd6eceafb347aa1fcbcbf4a20293fe6c84b543af09ff99b`；94 份原文 / 19 批已逐 SHA 和 chunk 字节核验。前 200 步对 clean 129 FSDP 基线，total/LM/MTP/aux/grad norm/LR/四项 data counter 各 1,600 对记录的最大差均为 0。与原 125200 的数值也为 0，但原 125200 的 `owned-process-observation-unavailable` 与 incomplete 分类原样保留，不能据数值相同升级其执行门。[新连续 400 步曲线和独立复核数据](fsdp-continuous400/README_ZH.md)。

历史报告：[FSDP 续训](../a1e6-fsdp-resume-400/README_ZH.md)、[FSDP 重算](../a1e6-recompute-200/README_ZH.md)、[TP2](../a1e6-tp2-200/README_ZH.md)、[TP4/CP2](../a1e6-tp4-cp2-200/README_ZH.md)、[CP4/EP2](../a1e6-cp4-ep2-200/README_ZH.md)、[EP4/EP8](../a1e6-ep4-ep8-200/README_ZH.md)、[TP2EP2](../a1e6-tp2ep2-200-tp4ep4-interrupted/README_ZH.md)、[TP2CP2EP2/EP4](../a1e6-tp2cp2-ep2-ep4-200/README_ZH.md)、[TP4EP4/CP2EP2](../a1e6-tp4ep4-cp2ep2-200/README_ZH.md)。这些报告的“当前”均指各自固定提交与记录时间，不是新源码的继承验收。

## 独立问题和单提交 PR

以下按对精度主线、恢复/重算正确性、资源与测试可靠性的影响排序。同一主题下的顺序是当前排查优先级，不代表已证明其对总 loss 误差的因果贡献。所有 issue 与 PR 都已有详细中文正文、互链和复现/测试边界。

| 优先序 | 问题与触发范围 | Issue | 单提交 PR | 已验证边界 / 尚缺内容 |
|---|---|---|---|---|
| 1 | 【子问题·跨并行自洽】Muon 缺少可恢复的跨参数 NS 合批开关 | [#250](https://github.com/hf1s/hyper-parallel/issues/250) | [#251](https://github.com/hf1s/hyper-parallel/pull/251) | PR251仅提供可恢复NS跨参数合批策略；裸b32与集成59b分明，当前EP200主LMFAIL，不等于产品已完成精度修复。 |
| 2 | Trainer 梯度累积按微批有效token归一化，改变整步目标 | [#203](https://github.com/hf1s/hyper-parallel/issues/203) | [#194](https://github.com/hf1s/hyper-parallel/pull/194) | 整步 token 分母缺陷的定向数学回归；历史多补丁200步改善不等于完整并行达标。 |
| 3 | 专家权重梯度局部BF16舍入后，FSDP FP32归约无法恢复精度 | [#209](https://github.com/hf1s/hyper-parallel/issues/209) | [#200](https://github.com/hf1s/hyper-parallel/pull/200) | FP32专家主梯度是显式能力；当前59b已启用仍有post-RS数值差，不等于该补丁错误或已解决全部EP漂移。 |
| 4 | Muon支持可持久化的NS精度配置并保留BF16默认 | [#210](https://github.com/hf1s/hyper-parallel/issues/210) | [#201](https://github.com/hf1s/hyper-parallel/pull/201) | NS dtype可恢复能力与默认BF16策略；当前59b实际ns_dtype=float32，不能把当前残差归为默认BF16。 |
| 5 | [缺陷] Trainer 高精度初始化未关闭 NPU HF32，错误属性写入被静默忽略 | [#227](https://github.com/hf1s/hyper-parallel/issues/227) | [#228](https://github.com/hf1s/hyper-parallel/pull/228) | HF32设置契约独立修复；当前59b实际allow_hf32=False，未据此定位长程残差。 |
| 6 | 断点续训后累计样本与 token 监控计数重置 | [#208](https://github.com/hf1s/hyper-parallel/issues/208) | [#197](https://github.com/hf1s/hyper-parallel/pull/197) | 历史544续训/连续400计数与loss结果保留；当前59b resume尚未执行，不能继承旧通过。 |
| 7 | 完整重计算未覆盖 JT 独立 MTP 解码器 | [#206](https://github.com/hf1s/hyper-parallel/issues/206) | [#195](https://github.com/hf1s/hyper-parallel/pull/195) | 历史544 FSDP full/selective200通过可保留；未证明59b全部拓扑和重算矩阵通过。 |
| 8 | JT MoE checkpoint重放重复提交tokens_per_expert计数 | [#207](https://github.com/hf1s/hyper-parallel/issues/207) | [#196](https://github.com/hf1s/hyper-parallel/pull/196) | 持久路由计数重放缺陷成立；不能据此推断旧loss必然漂移，历史FSDP重算零差不扩大为新矩阵。 |
| 9 | CP 分块输出损失的支持与多 MTP、TP/SP 回归覆盖 | [#205](https://github.com/hf1s/hyper-parallel/issues/205) | [#199](https://github.com/hf1s/hyper-parallel/pull/199) | 功能已由历史PR186合入；PR199是独立回归覆盖，不应将该PR说成功能尚未存在。 |
| 10 | FSDP 分片初始化后保留完整源Parameter，EP/CP额外占用显存 | [#204](https://github.com/hf1s/hyper-parallel/issues/204) | [#198](https://github.com/hf1s/hyper-parallel/pull/198) | 源 Parameter 释放的历史同拓扑数值与显存证据；性能专项目前暂停，不能将历史收益当新59b验收。 |
| 11 | 普通非整除 Shard 的重建形状错误及 Partial 归约入口缺少一致检查 | [#211](https://github.com/hf1s/hyper-parallel/issues/211) | [#214](https://github.com/hf1s/hyper-parallel/pull/214) | 明确拒绝不支持普通非整除Shard，定向CPU/Gloo/NPU覆盖，不等于实现任意布局或完整训练验收。 |
| 12 | 【缺陷】并行测试启动器覆盖上层 NPU 选卡，导致用例越界分配设备 | [#223](https://github.com/hf1s/hyper-parallel/issues/223) | [#224](https://github.com/hf1s/hyper-parallel/pull/224) | 启动器可见设备边界的受控回归，不等于新NPU训练或资源规则已全部动态回归。 |
| 13 | 【缺陷】FSDP 动态派生重算 wrapper 被集成报告漏计 | [#229](https://github.com/hf1s/hyper-parallel/issues/229) | [#230](https://github.com/hf1s/hyper-parallel/pull/230) | PR230实际merged、issue229closed，中文原文已准确记录；不改为draft/open。 |
| 14 | Trainer 资源统计遗漏其他 TP/PP rank 的耗时与显存峰值 | [#225](https://github.com/hf1s/hyper-parallel/issues/225) | [#226](https://github.com/hf1s/hyper-parallel/pull/226) | 全rank资源统计契约的定向CPU/Gloo与历史日志证据；当前性能/收益分析暂停。 |
| 15 | [缺陷] CPU 设备索引返回字符串导致 JT 模型构建失败 | [#218](https://github.com/hf1s/hyper-parallel/issues/218) | [#219](https://github.com/hf1s/hyper-parallel/pull/219) | CPU设备索引修复；与125环境设备发现及当前运行归属记录分开。 |

核验时 15 个产品 PR 均为 **1 commit**；14 个 open/draft，#230 已 merged，#229 已 closed。所有 15 个当前 head 的 statuses/check-runs 均为空，**CI 状态为未知，不是通过**。本次只修正文与关联，不改产品代码、PR head、草稿/合入状态或 issue 状态。

## 源码、证据与保留

上游 `JT_dev` 在本次只读核验时仍为 `a1e6c222eab097c9609d6e744a3c5f98aeed6bdc`；用户要求的 #180 已是固定源祖先。`544526f7` 是多修复集成验证源；`59b25cdd` 是其上加入 NS 跨参数合批策略的验证源，不能统称裸上游。实际环境为 CANN **9.2.0-beta.2**、Torch/TorchNPU 2.12、Python 3.12.14、8×Ascend 910B3。两侧保留 seed42、原数据、LR 1e-4/scheduler1000、MTP 0.3/aux 1e-4、FP32 NS 与实际 `ns_batch_across_params=False` 等配方，不通过改数值配方贴齐。

图与逐步数据、校验摘要一同公开；必要 checkpoint、原始诊断供体和可复现证据仍保留原位置，本包不上传几十 GiB 训练载荷。旧 125 监控 warning、失败/未封版草稿均保留真实分类。全局间隔的最新规则在暂停前为每次八卡释放后 600 秒，旧报告中的 1200 秒是历史执行规则；暂停期间没有新启动计划。

完整问题清单见 [catalog.json](catalog.json)，公开文件哈希见 [public-manifest.json](public-manifest.json)。原始 GitHub API 审计包含 104 次 GET、15 组问题及 3 个历史引用；公开摘要不包含凭据和其他用户的进程身份。
