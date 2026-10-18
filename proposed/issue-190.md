<!-- correctness-review-20261018 -->
## 2026-10-18 正确性复核：总差异表：历史窗口、当前源代码和修复必要性分开

**适用范围**：本单是固定旧基线的 301–600 窗口与若干消融；#217 后续独立验收是 201–400。两者不能拼接成同一个 200 步结果，也不能将不同消融背景的最大值变化相加成贡献率。

**本次最新复核**：MF master PyNative `409019aac0c6579bb4fb8a9b04a75c259d98a96e`、Megatron main `1effc75a46b26c6b93478355ff9fc3bb8ac1fafc`；[MF master PyNative 矩阵 schema](https://gitcode.com/mindspore/mindformers/blob/409019aac0c6579bb4fb8a9b04a75c259d98a96e/mindformers/pynative/base_models/gpt/gpt_model.py#L798)、[MF master PyNative NS](https://gitcode.com/mindspore/mindformers/blob/409019aac0c6579bb4fb8a9b04a75c259d98a96e/mindformers/pynative/optimizer/muon.py#L1195)、[Megatron 可选 QKV 拆分](https://github.com/NVIDIA/Megatron-LM/blob/1effc75a46b26c6b93478355ff9fc3bb8ac1fafc/megatron/core/optimizer/emerging_optimizers.py#L458)。当前 MF 逐 piece scale 已按自己的 shape；当前 HP router FP32 修复 #148、token 恒等权重 #180、K-shift 解耦 #232 已合入，旧差异表不代表这些问题仍开放于当前代码。

**必要修改**：#243 有 JT 原投影存储等价性依据；#213 保护特定 FP32 computed update 写回，但默认 BF16 路径不应被夸大为全面受损。RMSNorm、embedding、SwiGLU、RoPE 等精度差异，先核对定义、dtype 与导数契约。

**不推荐默认化**：旧跨文档 MTP、Q-up 末片 scale、固定 CANN 归约与旧 GEMM 只支持归档复现。PR #245 已降为 Draft，不能把它作为通用训练的必要修复清单。#241 是诊断工具。

**证据标准**：每个问题分别列触发条件、独立期望、最小反例、修复范围、状态恢复与真实测试边界。已完成的旧 MF 200 步复现仍有效，但不是最新 MF PyNative 的训练实测，也不能证明全部默认产品路径已对齐。


本次固定参考：MF GitCode master `409019aac0c6579bb4fb8a9b04a75c259d98a96e`（只核查 `mindformers/pynative/`，未用旧 training_graph 冒充），Megatron main `1effc75a46b26c6b93478355ff9fc3bb8ac1fafc`，当前 JT_dev `a1e6c222eab097c9609d6e744a3c5f98aeed6bdc`。源码路径/行号/SHA-256 见[完整正确性复核](https://github.com/xingleixu123/hyper-parallel/blob/xxl-dev/jt-correctness-review-20261018/AUDIT.md)与[来源清单](https://github.com/xingleixu123/hyper-parallel/blob/xxl-dev/jt-correctness-review-20261018/source-catalog.json)。本轮不新增训练结果，不以 MF loss 距离代替独立正确性判据。


<details><summary>原问题与历史实验记录（适用旧基线；当前判断以上方复核为准）</summary>

从 #157 拆出来：#157 本身的问题（300 步续训为什么漂移）已经在它自己的评论里回答完——根因是 #187（K-shift tiling）+ #188（loss 相加顺序），第 301 步前向已做到 0 ULP。本单是在那之上的两件事：(1) 两条线（Muon #149–#156/#157/#187/#188，AdamW #179/#180）加起来查到的全部差异，现在分别是什么状态；(2) #187+#188 都修完之后，跑满 301–600 的最终验证。

相关：#157（300 步漂移的原始问题及根因）、#179（AdamW 线 29 项消融）、#185（第 301 步逐位排查，已关闭）、#187、#188。

## 全部差异的现状分类

对照 JT_dev 实际 git 历史用 `git merge-base --is-ancestor` 逐一核实，不是转述 issue 文字。

### 已合入 JT_dev

| issue | 内容 | PR / commit |
|---|---|---|
| #152-1 | router GEMM 精度改 fp32 | #148，`436a1bdd` |
| #156-2 | RoPE 旋转精度改 fp32 | #162，`34d61bfe` |
| #155 | 单卡 routed experts 改 grouped GEMM + 融合 SwiGLU | #163，`06df6f99` |
| #180 | Trainer token 权重 `(x*n)/n` 不是恒等映射 | **已合入**：PR #180，merge `521d3a8a`（compare 已核：JT_dev 在该提交之后 2 个提交、behind_by=0）|

### 定位完成，没合入，需要落地

| issue | 内容 | 现状 |
|---|---|---|
| #187 | K-shift：`enable_full_determinism` 关闭 CANN matmul tiling，造成 311 bit kv_a 差异 | **产品化已有 draft**：PR #192（`xxl-dev/configurable-npu-matmul-k-shift`）新增 `training.close_matmul_k_shift: bool = true`，默认不变，显式 `false` 时在模型构建前置 0。真机验证：910B3/CANN 9.2.0-beta.2 上 true/false 两态环境变量分别为 1/0，false 态指标与我们入口级 `env.pop` 做法完全一致，待 maintainer 设计评审@twx |
| #149-1/#149-3 | Muon NS 和 MF 不一致：`linear_qkv`/`experts.gate_up_proj` 分块方式 + 缩放后更新的舍入点 | **已落地**：`feat/jt-muon-optimizer-switch`@`a5e7a4f8`（3 文件，+276/−35：新增 `mf_muon_ns.py` 181 行实现 MF 精确 NS，`mf_muon_parity.py`/`muon.py` 接线，`muon.py` 本身只加 2 个通用开关，不含任何 MF 专属逻辑）。新开关 `ns_mf_arith`，默认 `false`=HP 现状，`true`=MF 算术，真机验证 NS 逐位 100%，同一个补丁一并覆盖两项（舍入点/融合是其中一处修复）。 同内容的**存档分支**：`align/all-differences-300step`@`51ef5fb0`（同一适配层 + 完整差异清单文档，5 文件 +405/−35）。详见下方 Clean branch 301 验证段 |
| MTP 投影损失路径 | `chunked_cross_entropy.py` 的 `projected_cross_entropy`（**预计算一阶梯度**）未纳入 #179/#190 的口径比对 | 需决定：与 MF 的 CE 归约/dtype 对齐，或改走 `accelerator.loss_parallel=true`（#137 做法）|
| #213 | 单分片路径把 FP32 Muon update 打包成 BF16（`muon.py::_fused_broadcast_and_apply`）| 已由 PR #213 修（仅单分片分支）；待确认 8 卡多分片路径是否同样需要 |

### 确认是框架/设计差异，决定不改默认实现

| issue | 内容 | 结论 |
|---|---|---|
| #149-2 | Muon 缩放系数取维度（循环变量疑似泄漏） | MF 的 bug，HP 不用改 |
| #150 | MTP 跨文档移位 | MF 的 bug，HP 本来就对（引 Megatron-LM 上游） |
| #152-2 | router 权重精度 | 框架差异；细分为"HP Trainer 缺按模块设精度的配置入口"，非 bug，未列为必须做 |
| #153-1/2 | RMSNorm 权重精度、计算公式 | 框架差异，没有对错 |
| #154 | embedding 反向梯度精度 | 框架差异 |
| #156-1 | RoPE 位置编号（reset vs 全局） | 配置差异，数学等价 |
| #188 | loss 三项相加顺序 | 非 bug，不建议改默认 |
| #179 的 28/29 项（不含 #180） | AdamW 线内部实现：归约顺序、GEMM/布局、CE 归约等精度策略 | P2/P3：精度策略或零影响，逐条引 Megatron-LM 源码，**产品默认不改**；但作为**实验覆盖层**（`precision_probe.py`/`alignment_builder.py`）早已对齐并一直在用——#179 自己的 F 基线（29 项全开）200 步最大误差 9.5367431640625e-7，MAE 2.3e-7，#185/#187/#193 能做到前向 0 ULP 全靠这套覆盖层 |

### 实验分支（`test/align-b0-fwd-*`），均不建议进产品

和 #179 的独立消融结论互相印证：

| 分支 | 对应 #179 因素 | 判定 |
|---|---|---|
| `test/align-b0-fwd-swiglu-dense-fusion` | `swiglu` | P2，数学路径等价 |
| `test/align-b0-fwd-head-fp32-grad` | `head-fp32-grad` | P2，精度策略 |
| `test/align-b0-fwd-embedding-fp32` | `embedding` | P2，精度策略；另多占 1.56GB 显存 |
| `test/align-b0-fwd-ce-dvm` | `ce-dvm` | P2，参考兼容（复现 910B3 硬件 DVM 调度，非通用修复） |
| `test/align-b0-fwd-mf-loss-order` | 对应 #188 | 同上，非 bug |

### 一句话

两条线加起来查了 40 多项差异，真正该进产品代码的只有 5 项：3 项已合（#148/#162/#163），1 项已合入（#180），1 项已产品化 draft（#187，见 PR #192）。其余全部有明确依据（多数引了 Megatron-LM 上游源码）判定为框架设计差异或配置差异，不是 bug，不动默认实现。

反向与长程偏差的当前记录见下方「仍有偏差的位置」；反向段记录偏差位置。

## 第 301 步前向 + 反向均已逐位对齐

**第 301 步**（HP 接上 MF 第 300 步 checkpoint 续训的第一步）在 #187（K-shift）+ mindspore-ai#188（loss 相加顺序）两项修复后：

- **前向**：`loss` / `grad_norm` 与 MF **逐位相同（0 ULP）**；
- **反向**：46 个入口梯度 **45/46 逐位相同**；唯一例外是 `output_layer.weight`，根因是 CANN 9.2 与 8.5 的 GEMM 归约树（同一 `torch.mm(dy.T, x)` 在 CANN 8.5 下两侧 `0/778,567,680` 元素不同 ⇒ 换 8.5 后预期 46/46）。

逐位即上限——单步不存在"更对"的状态。**判据是单步逐位（前向 + 反向），不是 300 步曲线**；301–600 的差异仅在下方「仍有偏差的位置」记录，不在本节展开根因拆解。

【旧数字留档】在更早的配置下测得 300 步（301–600）`mean|Δloss|` 落在 0.04–0.09 的同一噪声带、且与叠加了几项修复无关；该组数字不代表当前配置，仅留档。


## Clean branch 301 验证

| 项 | 结果 | 证据 |
|---|---|---|
| 分支 | [`feat/jt-muon-mf-clean-2step`](https://github.com/hf1s/hyper-parallel/tree/feat/jt-muon-mf-clean-2step) | commit `ae2863a6`；基线 `a5e7a4f8`；patch 位于 `hyper_parallel/models/jt_deepseek_v3/adapter/`，入口为 `mf-alignment-2step.yaml` + `mf-alignment-2step.sh` |
| step-301 | **与已知 good anchor 对齐** | clean run `step300-301-clean-v5`，`EXIT_CODE=0`；日志 `loss=8.14211`、`grad_norm=10.2844`、`lm=6.28432`、`mtp=1.85748`、`aux=0.000314062`，与 anchor `8.142111778259277 / 10.284420013427734 / 6.284317970275879 / 1.8574793338775635 / 0.00031406228663399816` 的显示精度一致 |
| 301 入口 | **Hyper 内入口** | `mf-alignment-2step.sh` 加载 `mf-alignment-2step.yaml`；配置 `mf_muon`、precision factors、`fp32_update`、`ns_per_param`、`mtp_input_document_mask`、`close_matmul_k_shift`；实现位于 `mf_muon_parity.py`、`mf_muon_ns.py`、`precision_probe.py`、trainer/runtime 与 JT model adapter |

## 仍有偏差的位置（只列位置，不写根因拆解）

| 位置 | 状态 | 证据 / 边界 |
|---|---|---|
| step-301 前向与单步指标 | **无已知偏差** | clean branch 已复现 good anchor；本轮没有保留额外诊断 raw |
| step-302 起的连续轨迹 | **仍有偏差** | 旧 301–600 实验从 step 302 开始出现 loss/grad_norm 差异；该数字来自旧实验入口，尚未用当前 clean branch 重跑确认 |
| step-302 之后的具体算子/状态 | **待测量** | 连续轨迹的 loss/grad_norm 从 step-302 出现偏差；具体算子/状态字段待逐项测量 |




</details>
