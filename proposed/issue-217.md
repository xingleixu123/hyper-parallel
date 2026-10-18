<!-- correctness-review-20261018 -->
## 2026-10-18 正确性复核结论

旧 MF 归档显式配方的独立 200 步复现通过记录仍然有效；它只证明该固定配方的复现效果。**不能据此认定所有改动都应进入默认产品，也不能宣称最新 MF PyNative 已经过同一训练实测。**

| 对象 | 当前建议 | 独立于 MF 一致性的理由 / 触发范围 |
|---|---|---|
| #216 / PR #180 | 保留，已合入 | 按全局有效 token 均值推导的恒等梯度；局部 mean 与 Megatron loss sum 路线应按完整链路比较 |
| #215 / PR #213 | 保留条件修复，收紧默认影响 | 单分片收到已有 FP32 update 才触发；原生默认 Muon 更新通常已 BF16，归档覆写路径不能泛化 |
| #242 / PR #243 | 保留 JT 存储等价性修复 | 原独立投影仅合并存储时，独立/融合参数应得到相同逐矩阵更新；不是“Megatron 默认一律拆” |
| #187 / PR #192 | 历史 Draft，不原样合入 | #232 已取消环境接管；#192 默认 true 会重新覆盖调用者 K-shift 设置 |
| #240 / PR #241 | 保留只读诊断工具 | 同时报归档、普通路径和 FP64 误差；不替换产品 optimizer |
| #244 / PR #245 | **已降为 Draft，当前不推荐产品合入** | 跨文档目标、旧 Q-up scale 与固定内核调度是复现手段，且存在配置/初始化/runtime 支持边界缺口 |

行业复核还确认：当前 MF PyNative 已逐 piece 计算 shape scale；当前 JT router FP32（#148）和 FP32 RoPE 已存在，旧问题单不能继续将旧代码写成最新现状。#149、#150、#152、#153、#154、#155、#156、#187、#190、#210、#215、#216、#240、#242、#244 已逐单补充触发条件、正确性判据、最新来源、建议动作和验证边界。

优先处理目标函数、梯度和存储/环境接口契约；其次评审可配置精度的收益与成本。不同合法舍入不自动构成 bug；已低于要求且不改变结论的残差不继续扩大实验。

本轮是源码/契约审查及无设备反例验证，没有新增 NPU 训练。PR source heads 保持不变，#245 仅改 Draft 状态；所有 issue 保持 OPEN，CI 和合入状态单独核查。正式提交可重建实测树只证明可复现性，不证明产品必要性。

本次固定参考：MF GitCode master `409019aac0c6579bb4fb8a9b04a75c259d98a96e`（只核查 `mindformers/pynative/`，未用旧 training_graph 冒充），Megatron main `1effc75a46b26c6b93478355ff9fc3bb8ac1fafc`，当前 JT_dev `a1e6c222eab097c9609d6e744a3c5f98aeed6bdc`。源码路径/行号/SHA-256 见[完整正确性复核](https://github.com/xingleixu123/hyper-parallel/blob/xxl-dev/jt-correctness-review-20261018/AUDIT.md)与[来源清单](https://github.com/xingleixu123/hyper-parallel/blob/xxl-dev/jt-correctness-review-20261018/source-catalog.json)。本轮不新增训练结果，不以 MF loss 距离代替独立正确性判据。

<details><summary>2026-10-17 完整复跑、配置、消融与清理记录（原始结果保留）</summary>

# 最新 JT_dev：MF200 → MF/HP400 干净复跑与正式提交

2026-10-17，最新 JT_dev `a1e6c222` 加正式修复和显式归档兼容配方，新的独立 200 步复跑三项全部通过。该结论限定于固定 MF 归档兼容配方；原生 JT 默认加产品修复的结果单独保留，不能由此声称所有默认路径已经与旧 MF 相同。

## 完整验收

| 指标（MTP/aux 未加权） | 首步绝对误差 | 200 步最大绝对误差 | 最大误差步 | 超过 0.005 步数 |
|---|---:|---:|---:|---:|
| 总 loss | 9.5367431640625e-7 | 1.9073486328125e-6 | 210 | 0/200 |
| MTP loss | 7.947285975973273e-7 | 3.1789143886129523e-6 | 254 | 0/200 |
| aux loss | 5.8673322200775146e-8 | 1.2973323464393616e-6 | 290 | 0/200 |

首步指续训第 201 步；总 loss 门槛 0.0005，MTP/aux 各 0.001，三项每一步门槛均为 0.005。覆盖完整 201–400，退出码 0，不平均坏步、不只检查末步。MTP 从记录加权值除以 0.3，HP aux 除以 1e-4，MF aux 已为未加权值；各步同时核对分项重组 total，容差 2e-5。

[完整严格审计](https://github.com/xingleixu123/hyper-parallel/blob/3f1afbb1ff6b8a4d4c36cd1880a64d1e0b982410/evidence/clean-final-validation.json) · [数值与输入审计](https://github.com/xingleixu123/hyper-parallel/blob/3f1afbb1ff6b8a4d4c36cd1880a64d1e0b982410/evidence/hp-archive200-r2-numerical-audit.json) · [新 HP 指标](https://github.com/xingleixu123/hyper-parallel/blob/3f1afbb1ff6b8a4d4c36cd1880a64d1e0b982410/runs/hp-archive200-r2/metrics.jsonl) · [新 MF 指标](https://github.com/xingleixu123/hyper-parallel/blob/3f1afbb1ff6b8a4d4c36cd1880a64d1e0b982410/runs/mf-reference200/metrics.jsonl) · [2400 条对照误差 CSV](https://github.com/xingleixu123/hyper-parallel/blob/3f1afbb1ff6b8a4d4c36cd1880a64d1e0b982410/evidence/clean-200-loss-errors.csv)

![四个完整对照与同一次 MF 参考](https://raw.githubusercontent.com/xingleixu123/hyper-parallel/3f1afbb1ff6b8a4d4c36cd1880a64d1e0b982410/evidence/clean-200-loss-errors.png)

## 不依赖历史实验补丁的证据

训练冻结提交为 `dcfdf9b03cd353f79af840f9c65436eecefc3c87`。完成后再从最新 JT_dev 只依次 cherry-pick 已提交 PR #213、#243、#245，得到 `fe16c866cba6f40e5fea432e8147e7446fbf1e7f`。两者 **Git tree 都为 `05d9494e1d9851af165a3a181d546b3542051ab9`，全仓库文件树完全相同**，没有必须另补的隐藏实验改动。

[PR 组合与实测树等价回执](https://github.com/xingleixu123/hyper-parallel/blob/3f1afbb1ff6b8a4d4c36cd1880a64d1e0b982410/evidence/pr-series-tree-equivalence.json) · [实际测试源码清单](https://github.com/xingleixu123/hyper-parallel/blob/3f1afbb1ff6b8a4d4c36cd1880a64d1e0b982410/evidence/source-archive-r2-manifest.json) · [兼容 PR 14 文件一致性](https://github.com/xingleixu123/hyper-parallel/blob/3f1afbb1ff6b8a4d4c36cd1880a64d1e0b982410/evidence/profile-pr-source-identity.json)

训练后重新检查 1962 个冻结文件哈希，全部一致；339 个实际 HyperParallel 模块全部来自本次独立 editable 源码。正式模块没有运行时字符串改源码、全局 monkeypatch、对象换类或旧 phase12 实验脚本导入，没有未来参考值、步号或坐标修正。训练观测器只记录输入、恢复状态和标量；正式入口与观测入口调用同一个 `create_archived_trainer`。

从同一次 MF200 完整锚点恢复；第一次前向前核对 48 个模型数组、28 个 Muon 动量、18 个 AdamW 一阶矩、18 个二阶矩，shape、dtype、值均精确相同。训练步、optimizer step、scheduler 和数据游标均为 200。[实际恢复记录](https://github.com/xingleixu123/hyper-parallel/blob/3f1afbb1ff6b8a4d4c36cd1880a64d1e0b982410/runs/hp-archive200-r2/RESTORE.json)保留逐数组哈希；本轮不是只恢复模型，也不是优化器零状态重启。

本次 MF 独立新跑 200 步与历史 MF 三项逐步完全相等。新 HP 与旧已达标实验的 loss 和 grad_norm 全 200 步相同仅作附加验证，正式验收始终使用**本次新 MF**。

## 按影响与因果证据排序

先保证状态和训练语义可比较，再按会改变验收结论的因素定位；不把不同背景的最大误差变化相加为贡献率。

| 优先级 / 因素 | 受控证据与影响 | 结论及动作 |
|---|---|---|
| P0：完整恢复与实际输入 | 48+64 数组精确；200 条 tokens、labels、mask、文档边界逐步相同 | 本轮排除恢复丢失，不以“只载模型”解释误差 |
| P0：兼容迁移时 expert norm 提前 BF16 舍入 | 首版与修正版唯一算术增量是 FP32 `norm+eps` 后统一转 BF16；首差在 220，完整 total/MTP/aux 最大误差从 0.379674/0.170532/1.544800 降至约 2e-6/3e-6/1e-6 | `reductions.py::normalize_ns` 非特殊 expert 分支；最小用例 4 个元素全错、最大差 0.00390625，修正后通过；已随 #245 修正。这是未发布候选的迁移错误，不伪装为 JT 旧缺陷 |
| P1：融合存储改变 Muon 逻辑矩阵边界 | 原生基线与只加 #243 的同锚点 200 步：total 最大 0.551411→0.125125，MTP 0.273798→0.177481，aux 1.237740→0.991819；首步不变 | 真正产品缺陷，#242 / PR #243；独立参数和融合参数更新等价性为直接判据。单项仍不达标 |
| P1：旧目标和混合精度配方 | 原生两个对照均使用重置位置，而归档实际用绝对位置；旧 packed MTP、Q-up scale、精度/归约共同构成配方 | 语义差异必须先明确，不把组合收益冒称单因素收益；显式 #244 / PR #245，默认不启用 |
| P2：共享专家 down 归约 | 历史 #99→#100 唯一增量让 MTP 最大误差 0.0096702576→3.1789144e-6，超限 1→0；最新配方保留后完整验证通过 | #240 / PR #241 提供诊断与 FP64 正确性分析；属于旧算术兼容，不是已证明的通用 norm 数学缺陷 |
| P3：当前剩余 FP32 记录/舍入残差 | 三项最大约 1e-6，全部远低于要求 | 不继续为不影响验收的残差扩大 dump 或追加改动 |

历史输出头 GEMM、CE、加权专家梯度、Muon/AdamW 顺序等逐项打点与消融保留在[100 组实验记录](https://github.com/xingleixu123/hyper-parallel/blob/280d7d9a2e70575e90d96b4e164d89923b6394be/evidence/completed-cann92-inventory.csv)和[完整因果链](https://github.com/xingleixu123/hyper-parallel/blob/280d7d9a2e70575e90d96b4e164d89923b6394be/evidence/frontier396-causal-chain.json)。这些历史实验的基线与本次不同，不能冒记为全部在最新 JT_dev 上重新做过的独立消融。

本次四组均为完整 200 步，所有失败结果保留：

| 运行 | total 最大误差 / 超限步数 | MTP 最大误差 / 超限步数 | aux 最大误差 / 超限步数 |
|---|---:|---:|---:|
| 原生 JT + #213 | 0.5514106750 / 143 | 0.2737975121 / 177 | 1.2377402866 / 168 |
| 再加 #243 融合拆分 | 0.1251254082 / 138 | 0.1774811745 / 149 | 0.9918190278 / 170 |
| 显式兼容首版 87bdf43f | 0.3796739578 / 86 | 0.1705320676 / 102 | 1.5447999593 / 128 |
| 显式兼容修正版 dcfdf9b0 | 1.9073486e-6 / 0 | 3.1789144e-6 / 0 | 1.2973323e-6 / 0 |

## 正确性与旧 MF 兼容的边界

- **产品缺陷**：单分片 FP32 写回及独立逻辑投影的存储等价性分别以 #213、#243 修复。PR #180 已合入，当前 #232 已解除确定性设置对 K-shift 的接管，不再次叠加旧 #192 方案。
- **packed MTP**：独立文档应按文档边界移位；本配方为旧锚点保留跨文档目标。Megatron 当前 packed MTP 显式尊重文档边界。本次读取的 MF master PyNative MTP 调用仍是普通 roll，不能声称“最新 MF 已修复”而不核查调用。
- **位置输入**：104 条 MF 批次带文档重置后的 position_ids，但归档运行时 `use_eod_reset=False`，GPT 丢弃该字段，RoPE 实际使用绝对 arange。修正版明确 `reset_position_ids: false`；审计保留 104 条元数据差异，并按源代码与运行配置核对**有效位置**，200 条实际位置一致。原生两组有效位置不一致的 104 步也在各自审计中保留。
- **Q-up 旧 scale**：现代逐矩阵更新应按各自逻辑 shape 缩放；旧归档共用末分片 scale 仅为续训复现保留，不作为新训练默认。
- **dtype 和归约**：FP32 主参数、BF16 计算及梯度合并顺序是精度配方；CANN 固定 tile/core、epsilon 舍入和输出头 MatMulV2/NZ 是归档算术兼容。共享 down 实测 MF norm 标量更近 FP64，但不同的 92,955 个 BF16 normalized 元素全部是普通 HP 更近 FP64，因此接近 MF loss 不能替代数学正确性判断。

当前官方参考固定到实际复核的版本：[MF PyNative schema](https://gitcode.com/mindspore/mindformers/blob/1c9258ca845e21ddd66fde83e262d5635359bd30/mindformers/pynative/base_models/gpt/gpt_model.py#L798)、[MF PyNative Muon](https://gitcode.com/mindspore/mindformers/blob/1c9258ca845e21ddd66fde83e262d5635359bd30/mindformers/pynative/optimizer/muon.py)、[MF PyNative MTP](https://gitcode.com/mindspore/mindformers/blob/1c9258ca845e21ddd66fde83e262d5635359bd30/mindformers/pynative/transformers/multi_token_prediction.py#L460)、[Megatron Muon 接口](https://github.com/NVIDIA/Megatron-LM/blob/c6e38c6c3fc308108486add972974c4c74d1d860/megatron/core/optimizer/emerging_optimizers.py#L458)、[Megatron packed MTP](https://github.com/NVIDIA/Megatron-LM/blob/c6e38c6c3fc308108486add972974c4c74d1d860/megatron/core/transformer/multi_token_prediction.py#L202)。当前 MF/现代 Muon 的 FP32 norm 与 clamp 定义不要求旧 DVM 固定归约；`clamp(norm,eps)` 和 `norm+eps` 也不是同一算法定义。

## 配置、补丁与复现

实际设备为 125 的单张 **Ascend 910B3**。HP：Torch/TorchNPU 2.12，Python 3.12，CANN 9.2.0-beta.2；MF：用户固定 r1.8.0 归档，MindSpore 2.7.2，CANN 8.5.0，源码不改。没有 910C、多卡、其它形状和性能验证。

4K Arrow / 4710 条，batch=1，seed=42，全并行度 1；主干两层加一个 MTP，hidden5120、vocab152064、8 routed、2 shared、top2。FP32 主参数、BF16 计算。当前承接的配置为 **Muon＋AdamW 混合配方**，不能写成早期全 AdamW 阶段。

Muon：lr1e-4、wd0.1、momentum0.95、Nesterov、legacy NS5、eps1e-7、matched_adamw_rms0.2、QKclip100。AdamW：lr1e-4、wd0.1、betas(0.9,0.95)、eps1e-8，norm 不衰减。环境 `CLOSE_MATMUL_K_SHIFT=0`，保留 CANN set_env.sh 的 Python 编译组件路径。关闭大 checkpoint 输出，不修改原 MF 归档。

```bash
# 选择一个全新且由本任务使用的 checkout；已取得对应 fork 提交对象后：
git switch -c your-clean-rerun a1e6c222eab097c9609d6e744a3c5f98aeed6bdc
git cherry-pick 6f04715a9be134b1e544dcf3c8487bbd8051ffcf 22c2e70969af168d594c5302dbbe200a1b85a173 4ccb63a84304b4d968fb483f5d86b4a385be23df
pip install -e .
python -c 'import hyper_parallel; print(hyper_parallel.__file__)'
source /usr/local/Ascend/cann/set_env.sh
export CLOSE_MATMUL_K_SHIFT=0
export ASCEND_RT_VISIBLE_DEVICES=0
python -m torch.distributed.run --nproc_per_node=1 --master_port=31822 \
  -m hyper_parallel.models.jt_deepseek_v3.mf_archive.run your_complete_recipe.yaml
```

先替换路径、准备完整转换后的 MF200/HP200 锚点和原始数据；单有源码不能重建用户数据。标准入口用[实际 HP 完整 YAML](https://github.com/xingleixu123/hyper-parallel/blob/3f1afbb1ff6b8a4d4c36cd1880a64d1e0b982410/configs/hp-archive200-r2.yaml)和[MF 完整 YAML](https://github.com/xingleixu123/hyper-parallel/blob/3f1afbb1ff6b8a4d4c36cd1880a64d1e0b982410/configs/mf-reference200.yaml)。实际观测启动命令见 [HP 启动脚本](https://github.com/xingleixu123/hyper-parallel/blob/3f1afbb1ff6b8a4d4c36cd1880a64d1e0b982410/run-hp-archive200-r2.sh)、[MF 启动脚本](https://github.com/xingleixu123/hyper-parallel/blob/3f1afbb1ff6b8a4d4c36cd1880a64d1e0b982410/run-mf-reference200.sh)。`restore_optimizer/restore_train_state/restore_dataloader_state` 都为 true，`train_iters=400`。模型、优化器和 MLA 显式选择 `mf_archive` 类，标准入口负责归档裁剪阶段。观察入口记录 200 条实际输入后运行 `audit_clean_runs.py` 和 `verify_clean_final.py`。

原生 #243 修复仅处理 Q/latent-KV 与 expert gate/up 的存储融合边界；归档优化器自己实现更宽的旧 MF schema，不通过该原生工厂，因此不能将其长程组合结果都归给 #243。

## 独立 issue / PR、测试和剩余事项

| 独立问题 | PR | 本次核实状态与范围 |
|---|---|---|
| [#216 token 恒等权重](https://github.com/hf1s/hyper-parallel/issues/216) | [#180](https://github.com/hf1s/hyper-parallel/pull/180) | 已合入 JT_dev，本次包含 |
| [#187 K-shift 配置接管](https://github.com/hf1s/hyper-parallel/issues/187) | [#192](https://github.com/hf1s/hyper-parallel/pull/192)，上游采用 [#232](https://github.com/hf1s/hyper-parallel/pull/232) | #232 已合入，#192 仍开放；方案不同，不重复叠加 |
| [#215 FP32 单分片写回](https://github.com/hf1s/hyper-parallel/issues/215) | [#213](https://github.com/hf1s/hyper-parallel/pull/213) | 开放，实测头 6f04715a |
| [#242 融合矩阵边界](https://github.com/hf1s/hyper-parallel/issues/242) | [#243](https://github.com/hf1s/hyper-parallel/pull/243) | 新增独立产品修复，1 commit / 3 files，头 22c2e709；独立 14 UT 通过 |
| [#244 固定归档显式兼容能力](https://github.com/hf1s/hyper-parallel/issues/244) | [#245](https://github.com/hf1s/hyper-parallel/pull/245) | 新增显式配方，1 commit / 14 files，头 4ccb63a8；独立 18 UT 通过 |
| [#240 共享 down 归约兼容定位](https://github.com/hf1s/hyper-parallel/issues/240) | [#241](https://github.com/hf1s/hyper-parallel/pull/241) | 独立诊断和正确性说明；不是默认 optimizer 修复 |

集成源码 33 项 UT 通过，静态检查通过；输出头小矩阵 NPU 测试覆盖计算与设置恢复。独立分支和集成分支测试数不同，原因是 #213 和 #243 的额外测试，不能混报。pytest 的 7 个未注册项目 mark 警告保留。

[#213/#243/#245 当前头的远端 check-runs/status](https://github.com/xingleixu123/hyper-parallel/blob/3f1afbb1ff6b8a4d4c36cd1880a64d1e0b982410/evidence/final-pr-states.json)仍无可执行通过记录，status 为 pending；本地/NPU 验证通过不等于远端 CI 绿色。维护者设计/API 评审和 PR 合入尚未完成，总 issue 保持开放，不自动关闭已有问题。

更早差异分类继续引用 [#190](https://github.com/hf1s/hyper-parallel/issues/190)、#149、#150、#152、#153、#154、#155、#156、#210。#190 是历史 301–600 窗口；本次验收是 201–400，不能混用。PR #137 属于旧基线的较宽 Draft，本次重新实现与验证，不冒记为已验证本次代码。

## 留存与清理

本阶段先清理重复传输包 155,381,760 字节，最终再清理 175,823,392 字节，共 **331,205,152 字节（约 316 MiB）**。删除前核实精确路径、归属、无打开引用；保留 Git 提交、冻结源码、完整指标、输入审计、配置、恢复证据以及 MF200/HP200 锚点。临时渲染字体副本也已删除，未删本机原字体。

[前期清理回执](https://github.com/xingleixu123/hyper-parallel/blob/3f1afbb1ff6b8a4d4c36cd1880a64d1e0b982410/evidence/cleanup-transfer-archives.json) · [收尾清理回执](https://github.com/xingleixu123/hyper-parallel/blob/3f1afbb1ff6b8a4d4c36cd1880a64d1e0b982410/evidence/cleanup-final.json)。没有新建全量大张量 dump，没有改 zth 目录，没有停止 xxl2 的任务。实际输入批次留在 125 用于复查，公开证据包不包含原始样本和 checkpoint 张量。

本地证据根：`D:\Obsidian Vault D\hyperllal\MF-HP-JT库配置-125-20261014\clean-rerun-20261017`。远端任务根：`/home/xxl/jt-adamw-align-20261013/phase13`；状态环境：`/dev/shm/jt-adamw-align-20261013/state/jt-clean-archive-r2-20261017`。原锚点目录和共享数据保持保留。


本次证据提交 `3f1afbb1ff6b8a4d4c36cd1880a64d1e0b982410`，70 个文件，逐文件字节数与 SHA-256 见 [MANIFEST.json](https://github.com/xingleixu123/hyper-parallel/blob/3f1afbb1ff6b8a4d4c36cd1880a64d1e0b982410/MANIFEST.json)。所有新 PR 尚待评审与 CI，不随数值通过自动关闭总问题。


</details>
