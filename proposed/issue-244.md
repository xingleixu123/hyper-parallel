<!-- correctness-review-20261018 -->
## 2026-10-18 正确性复核：归档复现候选：PR #245 已降为 Draft，不作为产品必修

**本次纠正**：PR #245 已转 Draft，固定 head 仍为 `4ccb63a8`，没有修改历史训练源码。它复现旧 MF200 锚点具有诊断价值；但保留跨文档 MTP、Q-up 末片 scale、固定 CANN 调度，不能作为新训练通用正确性修复，也不能因 200 步通过就推出约 2309 行产品代码应合入。

**正确方向**：新训练保留独立文档目标、按所选逻辑矩阵 shape 缩放和明确的 dtype 配方；[Megatron packed MTP](https://github.com/NVIDIA/Megatron-LM/blob/1effc75a46b26c6b93478355ff9fc3bb8ac1fafc/megatron/core/transformer/multi_token_prediction.py#L219)、[MF 每个 piece 独立缩放](https://gitcode.com/mindspore/mindformers/blob/409019aac0c6579bb4fb8a9b04a75c259d98a96e/mindformers/pynative/optimizer/muon.py#L1238) 提供对应参考。归档兼容若继续维护，应优先作为独立诊断/复现交付，由维护者确认支持寿命、范围和回归责任。

**尚未解决的接口问题**：归档 Muon 会接受却忽略原生 `momentum_update_fn/reshape_fn/ns_transform_fn/zeropower_fn` 和部分缩放策略；新 `base_trainer_class` 子类入口绕过 `__init__`，带实例状态的子类可失败；固定安装目录版本文件也不能证明实际动态装载的 CANN 库版本。详见本次两轴报告及无设备 AST 反例。

**合入前条件**：拒绝不支持配置或保留其语义（含恢复参数组）、收窄/完善初始化协议、核查实际 runtime，而不是为旧轨迹增加产品公共 seam。当前不推荐作为通用产品改动合入。以上问题属于本候选的接口与支持范围，不另包装为已有默认 JT 缺陷。

**保留证据**：旧归档配方在指定 910B3/CANN/形状上 200 步复现通过，48 模型、64 optimizer 数组和冻结源码审计仍有效。代码树等价只证明正式提交可以重建实测文件，不证明这些改动都必要。此次源码研究不是对最新 MF PyNative 的重新训练对齐。


本次固定参考：MF GitCode master `409019aac0c6579bb4fb8a9b04a75c259d98a96e`（只核查 `mindformers/pynative/`，未用旧 training_graph 冒充），Megatron main `1effc75a46b26c6b93478355ff9fc3bb8ac1fafc`，当前 JT_dev `a1e6c222eab097c9609d6e744a3c5f98aeed6bdc`。源码路径/行号/SHA-256 见[完整正确性复核](https://github.com/xingleixu123/hyper-parallel/blob/xxl-dev/jt-correctness-review-20261018/AUDIT.md)与[来源清单](https://github.com/xingleixu123/hyper-parallel/blob/xxl-dev/jt-correctness-review-20261018/source-catalog.json)。本轮不新增训练结果，不以 MF loss 距离代替独立正确性判据。


<details><summary>原问题与历史实验记录（适用旧基线；当前判断以上方复核为准）</summary>

## 问题范围与交付目标

关联总问题 #217。此前旧 MF 检查点在 HP 上的三项 loss 达标依赖历史实验脚本组合，无法用当前 JT_dev 的正常模型、优化器和训练入口独立复现。需要将**固定归档的续训兼容配方**变成显式、可审查、可重建的入口，并准确说明它与产品正确性的边界。

这是一项兼容能力，不是把旧 MF 的每个数值行为认定为更正确。原生产品缺陷分别跟踪：FP32 单分片更新写回 #215 / PR #213、融合存储矩阵边界 #242 / PR #243；已合入的 token 权重 PR #180 和 K-shift 解耦 PR #232 已在本次基线中。本单不夹带这些独立修复。

## 基线、实际环境与恢复约束

- 当前 JT_dev：`a1e6c222eab097c9609d6e744a3c5f98aeed6bdc`。兼容功能分支仅包含本单变更；完整训练集成另包含 PR #213 和 PR #243。
- 125 实测单张 **Ascend 910B3**，HP Torch/TorchNPU 2.12、Python 3.12、CANN 9.2.0-beta.2；MF 为用户指定 r1.8.0 归档、MindSpore 2.7.2、CANN 8.5.0，归档代码未编辑。不能将硬件写成 910C。
- 4K Arrow，共 4710 条；从同一 MF 第 200 步锚点各自续训至第 400 步，独立产生 200 条训练结果。
- 48 个模型数组、28 个 Muon 动量、18 个 AdamW 一阶矩及 18 个二阶矩，在第一次前向前与离线转换结果逐值、逐 shape、逐 dtype 核对。恢复 optimizer step、scheduler、训练步数和数据游标；禁止只恢复模型。
- 两层主 decoder 加一层 MTP，hidden 5120、vocab 152064、8 routed / 2 shared experts、top-2、batch 1，所有并行度为 1。FP32 主参数，BF16 主计算。
- Muon：lr 1e-4、wd 0.1、momentum 0.95、Nesterov、legacy NS 5 次、eps 1e-7、matched_adamw_rms 0.2、QK clip 100。AdamW：lr 1e-4、wd 0.1、betas=(0.9,0.95)、eps 1e-8，norm 不衰减。延续现有 Muon/AdamW 混合配方，不能称为最早的全 AdamW 实验。

## 实现位置与正常入口

新增 `hyper_parallel/models/jt_deepseek_v3/mf_archive/`，通过 YAML `_target_` 选择 `build_archived_model` 和 `ArchivedOptimizerBuilder`；`python -m ...mf_archive.run recipe.yaml` 使用标准配置解析和 TextTrainer。TextTrainer 增加有类型检查的可选 `base_trainer_class`，默认仍为原 BaseTrainer。

| 文件 | 职责与选择理由 |
|---|---|
| `model.py` / `layers.py` | 普通类和模型工厂组合归档的精度、梯度合并、packed MTP 和位置规则；对设备、模型形状、并行度作显式范围检查 |
| `optimizer.py` | 完整 checkpoint 状态格式、Muon 逻辑分组、动量与 AdamW 算术顺序；继续依赖公共 FP32 写回修复 |
| `reductions.py` | 实时输入上的固定树形/tile/core 归约，保留归档的 FP32/BF16 舍入顺序 |
| `losses.py` | 正常 CE 前向及归档兼容反向归约，不输入参考 loss |
| `legacy_gemm.py` | 仅输出头权重梯度选择 ACL MatMulV2/NZ，表达式仍是 `dy.T @ x`；设置后恢复 TorchNPU 编译和格式开关 |
| `training.py` / `gradient_layout.json` | 按 MF 逻辑梯度顺序计算全局裁剪范数；JSON 只有名称、shape 和顺序，没有梯度值或修正值 |
| `recipe.yaml` / `README.md` | 完整配置模板、标准启动命令、正确性分类、依赖与复现限制 |

没有 `exec`/`eval`、运行时源码字符串替换、全局框架 monkeypatch、对象 `__class__` 替换或历史实验模块导入；没有读取后续 MF 权重、梯度、loss、训练步号或元素坐标来修正本次更新。观测器只记录实际输入、状态和标量。

完整 YAML 明确 `reset_position_ids: false`，`restore_optimizer/restore_train_state/restore_dataloader_state: true`，`train_iters: 400`，`save_ckpt: false`；数据和第 200 步检查点路径由使用者填入。只设置模型 `_target_` 不构成完整复现。

## 正确性：哪些应修，哪些只是旧归档兼容

| 行为 | 判断与依据 |
|---|---|
| FP32 主参数写回、存储融合改变 Muon 矩阵边界 | 独立产品缺陷，分别由 PR #213、#243 修复；不能用兼容类掩盖公共缺陷 |
| packed MTP 跨文档移位 | 本配方保留旧训练目标；独立 packed 文档应使用文档内 shift。Megatron 当前 packed 实现显式保持文档边界。MF 当前 PyNative 所核查的 MTP 调用仍是普通 roll，不能声称它已修复该边界 |
| 绝对 RoPE 位置 | 归档运行时 `use_eod_reset=False`，GPT 丢弃批次 position_ids，RotaryEmbedding 使用 arange。MF 批次元数据仍是重置位置；需区分字段差异和模型实际使用的位置 |
| Q-up 使用最后分片的 shape scale | 旧行为，不推荐为新训练默认。现代逐矩阵更新应使用各自逻辑 shape；为继续旧锚点才显式保留 |
| norm/router/embedding/head FP32 和分支梯度的舍入位置 | 混合精度配方，应明示 dtype 和梯度累加顺序；接近旧 MF 不能单独证明原 HP 是错的 |
| 固定 DVM 归约、epsilon 转换、旧 MatMulV2/NZ | CANN 8.5 归档算术的复现，不是 CANN 9.2 或 HP 通用数学缺陷的证据；本实现不作性能承诺 |
| sum-form 动量、legacy NS 系数、AdamW 运算次序 | 显式优化配方和舍入兼容，需恢复原 moment；不同合法配方不能混称同一算法 |

官方代码参考固定到本次复核的提交：

- [MF master PyNative Muon schema](https://gitcode.com/mindspore/mindformers/blob/1c9258ca845e21ddd66fde83e262d5635359bd30/mindformers/pynative/base_models/gpt/gpt_model.py#L798)：模型显式指定逻辑矩阵分组。
- [MF master PyNative Muon](https://gitcode.com/mindspore/mindformers/blob/1c9258ca845e21ddd66fde83e262d5635359bd30/mindformers/pynative/optimizer/muon.py)：FP32 norm、clamp 下限、BF16 迭代，与旧 `norm + eps` 不是同一个定义。
- [Megatron Muon 接口](https://github.com/NVIDIA/Megatron-LM/blob/c6e38c6c3fc308108486add972974c4c74d1d860/megatron/core/optimizer/emerging_optimizers.py#L458)：按配置恢复独立 QKV 矩阵，再分别正交化和缩放；不能把普通 QKV 布局直接当成 JT MLA 布局。
- [MF master PyNative MTP](https://gitcode.com/mindspore/mindformers/blob/1c9258ca845e21ddd66fde83e262d5635359bd30/mindformers/pynative/transformers/multi_token_prediction.py#L460)：本次读取的 embedding、label 与 mask 移位调用没有传入 packed 文档边界；不能仅凭“最新版本”假定已按文档处理。
- [Megatron packed MTP](https://github.com/NVIDIA/Megatron-LM/blob/c6e38c6c3fc308108486add972974c4c74d1d860/megatron/core/transformer/multi_token_prediction.py#L202)：显式处理 packed sequence 内部移位。

已定位的共享专家 down 样本中，MF norm 标量更近 FP64，但 92,955 个不同 BF16 normalized 元素全部是普通 HP 更近 FP64。这个反例直接说明“更贴近 MF loss”与“每个算子更准确”不能等同，详细因果链见 #240 / PR #241。

## 兼容迁移中的失败对照与修正

首版正式候选 `87bdf43f` 从第 201 步开始训练，前 19 步三项 loss 和 grad_norm 与旧已达标实验相同；首次分叉在第 220 步。完整 200 步结果保留，三项最大误差为 total `0.3796739578`、MTP `0.1705320676`、aux `1.5447999593`，超过 0.005 的步数分别为 86、102、128。

定位到 `reductions.py::normalize_ns` 的非特殊 expert 分支：迁移时提前将 norm 舍入为 BF16，遗漏旧归档在 FP32 中计算 `norm + eps` 后统一转 BF16 的规则。修正只恢复此顺序，不改变 epsilon、NS、学习率或已有状态。最小独立 FP64 参考用例的 4 个 normalized 元素在旧候选中全部不同，最大差 `0.00390625`；修正后通过。它是本次未发布兼容实现的迁移错误，不另包装为已有 JT 默认缺陷。

共享专家 down 的特殊归约保留原已验证实现；不同形状不能凭名称套同一规约。首版失败和修正版均完整保留，以逐步误差检验修正，不能只挑末步或平均值。

## 验证与验收

- 本功能独立分支：归档 9 项、基线 Muon 8 项、Trainer 1 项，共 **18 passed**。
- 集成 PR #213、#243 的冻结复跑源：**33 passed**，新增最小 norm 用例修正前失败、修正后通过。7 个项目 mark 未注册警告保留在原始日志。
- 输出头小矩阵 NPU 测试验证结果和作用域设置恢复；代码静态检查通过。没有宣称所有 CI、硬件或多卡组合通过。
- 完整训练验收见总问题 #217 的本次干净复跑证据：必须恰好覆盖 201–400 的 200 步，逐步输入一致，恢复全部状态，确认独立源码和导入路径，三项逐步误差分别验收。
- 门槛：total 首步 ≤0.0005，**未加权** MTP/aux 首步各 ≤0.001，三项每一步各 ≤0.005。MTP 从记录加权值除以 0.3，HP aux 除以 1e-4，MF aux 原本未加权；同时核对分项重组 total 的误差 ≤2e-5。

## 本次正式 200 步复跑结果

修正版 `hp-archive200-r2` 已完成 201–400、退出 0，三项全部通过。新跑 MF 参考也独立完成 200 步，并与历史 MF 参考的三项指标逐步精确相等。正式源码集成提交 `dcfdf9b03cd353f79af840f9c65436eecefc3c87`；兼容功能独立提交 `4ccb63a84304b4d968fb483f5d86b4a385be23df`，14 个功能文件与实测源码逐字节相同。

| 指标（MTP/aux 未加权） | 首步绝对误差 | 200 步最大绝对误差 | 最大误差步 | 超 0.005 步数 |
|---|---:|---:|---:|---:|
| 总 loss | 9.536743164062e-07 | 1.907348632812e-06 | 210 | 0/200 |
| MTP loss | 7.947285975973e-07 | 3.178914388613e-06 | 254 | 0/200 |
| aux loss | 5.867332220078e-08 | 1.297332346439e-06 | 290 | 0/200 |


总 loss 首步门槛 0.0005，未加权 MTP/aux 首步各 0.001，三项全 200 步逐步门槛均为 0.005。不是平均、末步或加权后口径。HP 原始指标 SHA-256 `e09ca8227561e200762238e33d942357c05ee45fd4afd74ac3840cc6efe102a8`；MF `af3ee47a2c31da10dd2308b05e513f189979c9b21c9406bd0e9bc21593eba300`。

1962 个冻结文件训练后 SHA-256 全部相符；全部实际 HyperParallel 模块来自独立 editable 源码。48 模型和 64 优化器数组恢复精确，step、scheduler、数据游标均为 200。200 条实际 tokens、labels、loss mask 和文档边界一致。

有 104 条 MF 批次的 position_ids 元数据与 HP 不同，原因已核实：MF 归档运行配置 `use_eod_reset=False`，模型实际忽略该字段并采用绝对位置。审计保留原字段差异，依据实际 GPT/RotaryEmbedding 源码和日志哈希核对模型有效位置，200 步有效位置无差异。没有将不同的有效输入静默判为一致。

首版兼容候选的完整失败结果仍保留；修正版三项 loss 和 grad_norm 的全部 200 步与历史达标实验逐值相等仅作额外一致性证据，验收参考始终是本次新跑 MF。


## 适用限制

固定 910B3 / 4K / 特定网络形状、单卡的兼容能力，包含主机参与归约与同步；不承诺其它 CANN、设备、形状、多卡和吞吐。只允许一个训练线程使用编译设置作用域。不修改 JT、公共 Muon、AdamW 和 Trainer 的默认算法选择；模型工厂拒绝未验证范围。

历史 PR #137 是旧基线上的较宽 Draft；本单在最新 JT_dev 上以正常模块实现和严格状态/输入审计重新验证，不把旧实验结果冒记为新代码结果。总问题保持开放以跟踪产品 PR 和维护者评审。


配对提交：[#244](https://github.com/hf1s/hyper-parallel/issues/244) ↔ [PR #245](https://github.com/hf1s/hyper-parallel/pull/245)。


<!-- clean-200-final-evidence -->
### 本次最终证据与实际提交组合

[完整中文复跑报告](https://github.com/xingleixu123/hyper-parallel/blob/3f1afbb1ff6b8a4d4c36cd1880a64d1e0b982410/FINAL-REPORT.md) · [严格审计](https://github.com/xingleixu123/hyper-parallel/blob/3f1afbb1ff6b8a4d4c36cd1880a64d1e0b982410/evidence/clean-final-validation.json) · [三项误差曲线](https://github.com/xingleixu123/hyper-parallel/blob/3f1afbb1ff6b8a4d4c36cd1880a64d1e0b982410/evidence/clean-200-loss-errors.png) · [正式 PR 组合与实测代码树等价](https://github.com/xingleixu123/hyper-parallel/blob/3f1afbb1ff6b8a4d4c36cd1880a64d1e0b982410/evidence/pr-series-tree-equivalence.json)。

从最新 JT_dev 仅依次应用 PR #213、#243、#245，得到 `fe16c866cba6f40e5fea432e8147e7446fbf1e7f`，与实际冻结复跑 `dcfdf9b03cd353f79af840f9c65436eecefc3c87` 的完整 Git tree 严格相同：`05d9494e1d9851af165a3a181d546b3542051ab9`。没有另外必须加载的实验补丁。数值通过限定于显式归档配方；当前 PR CI 仍无通过记录。


</details>
