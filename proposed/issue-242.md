<!-- correctness-review-20261018 -->
## 2026-10-18 正确性复核：融合投影：依据 JT 的逻辑存储契约修复，不按 MF 默认判定

**独立产品依据**：当前 JT 明确从原 q_a 与 kv_a 两个投影拼接存储，并在前向按同一边界拆开；expert gate/up 在 SwiGLU 中也承担不同逻辑角色。若该变化仅是存储融合，Muon 更新应与原独立矩阵一致。整体 NS 的归一化/Gram 会产生跨块耦合，一般不与分块 NS 等价。

**当前实现范围**：PR #243 只恢复 Q 与整个 latent-KV 投影的两个边界，以及每 expert gate/up 的两个边界。latent-KV 内的 512/64 行本来属于同一个原投影；进一步拆 RoPE、Q-up/KV-up 是另一个优化配方，不应夹带进本修复。

**行业限定**：[MF master PyNative 矩阵 schema](https://gitcode.com/mindspore/mindformers/blob/409019aac0c6579bb4fb8a9b04a75c259d98a96e/mindformers/pynative/base_models/gpt/gpt_model.py#L798) 的分组更宽；[Megatron 可选 QKV 拆分](https://github.com/NVIDIA/Megatron-LM/blob/1effc75a46b26c6b93478355ff9fc3bb8ac1fafc/megatron/core/optimizer/emerging_optimizers.py#L458) 只有在配置启用且模式支持时按 QKV 分块，`split_qkv` 默认 false，部分 GTP 情形仍 fallback 整体优化。不能声称“Megatron 一律拆”“任何融合整体都是 bug”。若用户有意选择整个融合矩阵优化，它是另一个明确配方。

**验收及迁移**：独立参数与融合存储在同一 NS/scale 定义下更新严格相等是直接判据，已有两个修复前失败的用例。默认 true 会改变旧整体配方下一步更新，应显式 false 保留旧配置，自定义回调冲突必须报错。moment 状态仍按原 shape 保存，不重新初始化。

**验证边界**：既有独立 14 UT、集成 33 UT 和只开本项的单卡 200 步是已执行证据。本次未重新执行，也没有全部 TP/EP 训练通过的证据。单因素仍未达三项门槛，不将归档组合收益归给该 PR。


本次固定参考：MF GitCode master `409019aac0c6579bb4fb8a9b04a75c259d98a96e`（只核查 `mindformers/pynative/`，未用旧 training_graph 冒充），Megatron main `1effc75a46b26c6b93478355ff9fc3bb8ac1fafc`，当前 JT_dev `a1e6c222eab097c9609d6e744a3c5f98aeed6bdc`。源码路径/行号/SHA-256 见[完整正确性复核](https://github.com/xingleixu123/hyper-parallel/blob/xxl-dev/jt-correctness-review-20261018/AUDIT.md)与[来源清单](https://github.com/xingleixu123/hyper-parallel/blob/xxl-dev/jt-correctness-review-20261018/source-catalog.json)。本轮不新增训练结果，不以 MF loss 距离代替独立正确性判据。


<details><summary>原问题与历史实验记录（适用旧基线；当前判断以上方复核为准）</summary>

## 问题范围

总问题：#217。历史调查集合：#149、#165。本单只处理 **JT 将独立 Q/latent-KV、expert gate/up 投影融合存储后，Muon 默认入口没有恢复逻辑矩阵边界**。Q-up/RoPE 的进一步分组、旧 MF 最后分片缩放、CANN 舍入、MTP 文档边界均不属于本单修复。

### 复现环境与触发方式

- 目标分支：`JT_dev`，审计基线 `a1e6c222eab097c9609d6e744a3c5f98aeed6bdc`。
- 125 / 单张 Ascend 910B3 / HP CANN 9.2.0-beta.2 / Torch 与 torch_npu 2.12。
- JT 原生 YAML 的 `hyper_parallel.models.jt_deepseek_v3.adapter.runtime.jt_optimizer.build_optimizer`；Muon legacy、NS 5 次、momentum=0.95、matched_adamw_rms=0.2。
- 模型将 Q-down `[1536,5120]` 和 latent-KV-down `[576,5120]` 存为 `linear_qkv.weight [2112,5120]`；expert gate/up 存为 `[8,3072,5120]`。

### 实际与期望行为

实际：工厂只构造普通 Muon，没有设置逻辑矩阵变换。NS 的归一化、Gram 和 shape scaling 使用整个融合矩阵，存储重构因此改变了优化算法。

期望：在“只合并参数存储”的情况下，Q 和 latent-KV 分别做 NS；每个 expert 的 gate/up 分别做 NS；各自按实际逻辑形状计算缩放，再恢复融合布局。`NS(cat(A,B))` 一般不等于 `cat(NS(A),NS(B))`，所以这不是可以忽略的存储细节。

注意：latent-KV 的 512 个潜变量行和 64 个 rotary 行本来同属一个 576 行投影。本修复将 576 行保留为一个矩阵。进一步分成 512/64、或者把 Q-up/KV-up 按 head 拆分属于另一个显式优化配方，不应与本单“存储融合不改变原投影边界”的主张混淆。

### 最小复现与独立判据

新增 `tests/ut/model_integration/test_jt_muon_fused_matrices.py`，构造固定随机梯度，分别用：

1. 两个独立 Parameter，调用原生 Muon NS 与 shape scaling；
2. 一个融合 Parameter，调用 JT 工厂实际配置的变换，再调用同一个原生 Muon 实现。

直接比较最终更新，要求 `rtol=0, atol=0`。Q/latent-KV 和三维 expert gate/up 两项在修复前均失败，修复后均严格相等；同时覆盖历史退出开关、自定义回调冲突、错误形状和不完整恢复结果拒绝。不是用 MF 未来梯度、记录坐标或目标 loss 作为期望值。

```bash
python -m pytest -q tests/ut/model_integration/test_jt_muon_fused_matrices.py tests/ut/core/optimizer/test_muon.py
```

已提交 PR #243 的独立分支（仅本问题，基于 JT_dev，无 PR #213 夹带）实测 **14 passed**；集成 PR #213 的复跑分支实测 **23 passed**。pytest 的未注册项目 mark 警告单独记录，不表示数值测试失败。

### 200 步单因素影响（不能替代三项达标验收）

两个 HP 分支都从完全相同的 MF 第 200 步模型、优化器、训练步数和采样游标恢复；MF 固定压缩包、CANN 8.5.0、4K Arrow，续训 201–400。

| 指标（MTP/aux 未加权） | 基线最大绝对误差 | 只加入融合拆分后的最大绝对误差 | 新最大误差步 | 新超 0.005 步数 |
|---|---:|---:|---:|---:|
| 总 loss | 0.5514106750 | 0.1251254082 | 358 | 138/200 |
| MTP loss | 0.2737975121 | 0.1774811745 | 334 | 149/200 |
| aux loss | 1.2377402866 | 0.9918190278 | 383 | 170/200 |

首步误差保持总 loss `0.0015773773`、MTP `0.0112557411`、aux `0.0018870337`，因为首步前向发生在第一次优化器更新之前。这轮 **三项仍未达标**，不能声称本修复独自解决了 #217；误差轨迹具有非线性，最大值变化也不等于可相加的“贡献率”。

HP 单因素指标 SHA-256：`22728a192464ef1c2721e273f044e4e12cf92606b0503820822d982d445dd6c8`。

### 正确性参考与边界

- [JT 原生 MLA 的参数融合代码](https://github.com/hf1s/hyper-parallel/blob/a1e6c222eab097c9609d6e744a3c5f98aeed6bdc/hyper_parallel/models/jt_deepseek_v3/modeling_jt_deepseek_v3.py#L185) 是本修复 Q/latent-KV 两个边界的直接依据。
- [Megatron 当前 emerging optimizer 的 QKV 拆分](https://github.com/NVIDIA/Megatron-LM/blob/c6e38c6c3fc308108486add972974c4c74d1d860/megatron/core/optimizer/emerging_optimizers.py#L458) 在配置启用时，先恢复逻辑投影，再分别正交化和缩放、最后拼回。它的普通 Q/K/V 布局不能直接套到 JT 的 MLA Q/latent-KV 布局；可借鉴的是保持明确逻辑矩阵边界的原则。
- [MF 最新 master 的 Pynative 模型 Muon schema](https://gitcode.com/mindspore/mindformers/blob/1c9258ca845e21ddd66fde83e262d5635359bd30/mindformers/pynative/base_models/gpt/gpt_model.py#L798) 显式拆 expert gate/up，也对 MLA 进一步按配方拆 Q/KV/RoPE。本 PR 不声称已完全复制该 schema。

### 配置与兼容

新增 `optimizer.split_fused_matrices: true`（默认），只作用于 JT 工厂。已有训练若有意继续历史“融合整体作为矩阵”的配方，显式设为 `false`。已提供 `reshape_fn`/`ns_transform_fn` 的调用者也需设为 `false`，工厂会报错而不会静默覆盖自定义配置。

不更改参数名、形状、优化器 moment 的存储格式、学习率或损失定义。新默认会改变旧配方的下一步更新，续训迁移必须明确选择；本次长程验证范围为单卡，不据此宣称所有 TP/EP 组合都已验证。

### 与既有问题和 PR 的关系

这不是首次发现 Muon 逻辑分组差异。#165 已记录旧分支的模型 schema；PR #24 曾在 `codex/deepseek-v32-sft-trainer` 合入更宽的 MLA 拆分，PR #137 仍为面向旧基线、多项配方的 Draft。本单在当前 `JT_dev a1e6c222` 重现的是默认 JT 工厂未连接融合矩阵变换，聚焦 Q/latent-KV 与 expert gate/up 的存储等价性，避免把旧 PR 的其它精度、MTP 和缩放行为一并作为默认修复。#137 的既有实验不作为本单最新代码的验证证据。

### 验收条件

- 独立参数与融合存储两种布局的逐矩阵 Muon 更新严格相等。
- 自定义回调不得被默认回调静默覆盖；旧融合配方可显式选择。
- 不重置恢复的模型或优化器状态。
- 200 步影响与三项门槛分别报告；只有这一项通过不能关闭总问题 #217。


配对提交：[#242](https://github.com/hf1s/hyper-parallel/issues/242) ↔ [PR #243](https://github.com/hf1s/hyper-parallel/pull/243)。

原生基线和仅融合拆分两组 HP 都保留 `reset_position_ids: true`，归档 MF 实际为绝对位置；双方有效位置不同的 104 步已在审计中明确记录。两个 HP 对照之间只有本项增量，仍可判断该项在同一背景下的影响；不能称这两轮已经解决了与 MF 的全部语义差异。


<!-- clean-200-final-evidence -->
### 本次最终证据与实际提交组合

[完整中文复跑报告](https://github.com/xingleixu123/hyper-parallel/blob/3f1afbb1ff6b8a4d4c36cd1880a64d1e0b982410/FINAL-REPORT.md) · [严格审计](https://github.com/xingleixu123/hyper-parallel/blob/3f1afbb1ff6b8a4d4c36cd1880a64d1e0b982410/evidence/clean-final-validation.json) · [三项误差曲线](https://github.com/xingleixu123/hyper-parallel/blob/3f1afbb1ff6b8a4d4c36cd1880a64d1e0b982410/evidence/clean-200-loss-errors.png) · [正式 PR 组合与实测代码树等价](https://github.com/xingleixu123/hyper-parallel/blob/3f1afbb1ff6b8a4d4c36cd1880a64d1e0b982410/evidence/pr-series-tree-equivalence.json)。

从最新 JT_dev 仅依次应用 PR #213、#243、#245，得到 `fe16c866cba6f40e5fea432e8147e7446fbf1e7f`，与实际冻结复跑 `dcfdf9b03cd353f79af840f9c65436eecefc3c87` 的完整 Git tree 严格相同：`05d9494e1d9851af165a3a181d546b3542051ab9`。没有另外必须加载的实验补丁。数值通过限定于显式归档配方；当前 PR CI 仍无通过记录。


</details>
