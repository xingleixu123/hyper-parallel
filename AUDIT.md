# MF / HP 对齐：正确性与产品修改必要性复核

复核日期：2026-10-18。本轮纠正了两个范围结论，并调整了一个 PR 的合入状态：**PR #213 只在实际收到已有 FP32 update 时保护精度；PR #192 已被上游 #232 另一方案取代；旧 MF 归档复现 PR #245 已转 Draft，当前不推荐作为通用产品改动合入。** PR #180 和 #243 有独立于 MF 轨迹的正确性依据，PR #241 可保留为只读诊断工具。

前一轮把约 2309 行归档兼容包作为可供产品合入的正式候选，范围偏宽。本轮据实际代码与接口反例收紧该结论。旧归档配方的 200 步复现通过仍成立；“提交能重建实测代码树”证明的是可复现性，不能证明所有改动都必要。此次没有新训练，不能冒记为对最新 MF PyNative 的 200 步实测。

## 本次实际核查的官方源码

| 对象 | 固定提交 / 来源 | 具体核查范围 |
|---|---|---|
| MF GitCode master | `409019aac0c6579bb4fb8a9b04a75c259d98a96e`，官方分支 API 查询 | 只使用 `mindformers/pynative/`：Muon、GPT schema、MTP、router、RMSNorm、embedding、expert/MLP、RoPE |
| Megatron-LM main | `1effc75a46b26c6b93478355ff9fc3bb8ac1fafc`，官方 GitHub API 查询 | emerging optimizer、token 梯度归一化、MTP、router、Norm、embedding、MLP、RoPE |
| Megatron 声明的 EO 开发依赖 | `v0.3.0` → `b309e2f01cda75dc96a6dc1a2355a7b3b64b5e16` | `pyproject.toml` 声明与该 tag 的 Muon/normalize；不将 EO 任意 main 当成所有 Megatron 部署实际安装版本 |
| HP 当前 JT_dev | `a1e6c222eab097c9609d6e744a3c5f98aeed6bdc` | 每个 PR 按自己的固定 base/head diff 审查，未把继承代码当新增问题 |

[来源与文件 SHA-256 清单](source-catalog.json)。第三方原始源码快照保留在本地 `industry/`；公开报告直接链接官方固定提交，避免只引用旧训练图或记忆中的 API。

重要的具体源码位置：

- [MF schema：gpt_model.py 798–867](https://gitcode.com/mindspore/mindformers/blob/409019aac0c6579bb4fb8a9b04a75c259d98a96e/mindformers/pynative/base_models/gpt/gpt_model.py#L798)：显式定义模型逻辑矩阵；分组比 PR #243 更宽，不能全部混入同一存储修复。
- [MF NS：muon.py 1195–1226](https://gitcode.com/mindspore/mindformers/blob/409019aac0c6579bb4fb8a9b04a75c259d98a96e/mindformers/pynative/optimizer/muon.py#L1195)：FP32 norm/clamp 后进入 BF16 迭代；不是旧 `norm + eps` 定义。
- [MF scale：muon.py 1238–1248](https://gitcode.com/mindspore/mindformers/blob/409019aac0c6579bb4fb8a9b04a75c259d98a96e/mindformers/pynative/optimizer/muon.py#L1238)：逐 piece 使用各自 shape。旧 Q-up 共用最后分片 scale 的问题不能写成最新 master 现状。
- [Megatron Muon：emerging_optimizers.py 458–510](https://github.com/NVIDIA/Megatron-LM/blob/1effc75a46b26c6b93478355ff9fc3bb8ac1fafc/megatron/core/optimizer/emerging_optimizers.py#L458)：配置启用且模式支持时才拆 QKV，默认 `split_qkv=False`，部分 GTP 模式会 fallback 整体更新。行业并没有“任何融合整体 NS 必错”的普遍规则。
- [Megatron token 归一化：finalize_model_grads.py 738–757](https://github.com/NVIDIA/Megatron-LM/blob/1effc75a46b26c6b93478355ff9fc3bb8ac1fafc/megatron/core/distributed/finalize_model_grads.py#L738)：累计总和梯度后按全局 token 数缩放。HP 是局部 mean，比较完整链路，不能移植单行公式。
- [MF PyNative MTP：651–661](https://gitcode.com/mindspore/mindformers/blob/409019aac0c6579bb4fb8a9b04a75c259d98a96e/mindformers/pynative/transformers/multi_token_prediction.py#L651) 和 [Megatron packed MTP：219–275](https://github.com/NVIDIA/Megatron-LM/blob/1effc75a46b26c6b93478355ff9fc3bb8ac1fafc/megatron/core/transformer/multi_token_prediction.py#L219)：分别核查实际 shift 调用及文档边界参数，不能凭框架版本名称假定语义。

## PR 判断与必要修改

| PR / issue | 当前处理 | 独立正确性依据、触发条件和限制 |
|---|---|---|
| #180 / #216 | 已合入，保留 | 局部 mean 应乘 `n/N × DP × CP`，使平均后的梯度等于全局 token mean。单 rank 单微批应为严格恒等比例，与 MF 算术顺序无关 |
| #213 / #215 | 保留条件修复，修正文案 | 单分片收到 FP32 computed update 时，无需 BF16 通信 pack。默认 Muon 更新通常已 BF16；归档覆写动量/NS 的真实收益不能泛化到默认 JT |
| #243 / #242 | 保留 JT 存储契约修复 | 原 q_a/latent-KV、expert gate/up 仅融合存储时保持逐矩阵更新等价；原始投影与独立/融合更新测试是依据。旧整体优化是另一明确配方，需说明默认迁移与 false 退出 |
| #192 / #187 | 保持 Draft，标记被另一方案取代 | #232 已移除确定性设置对 K-shift 的写入；#192 默认 true 若原样合入会重新覆盖调用者环境。开关不同不等于哪个 kernel 数学上错 |
| #241 / #240 | 保留 examples 只读诊断 | 同时报普通、归档、FP64 数值与 hash；不替换通用 optimizer，不将 CANN 旧归约当作必需默认 |
| #245 / #244 | **已从 Ready 转 Draft** | 跨文档 MTP、末片 scale、固定 tile/core 和旧 GEMM 是复现选择。合入必要性不足，并有下述接口/支持范围缺口；当前不推荐产品合入 |

没有修改上述 PR 的代码 head，没有重写提交；只更新结论、标题、正文和 #245 Draft 状态。已合入的 #180、#232 不重新修改。所有问题单保持 OPEN，不以审查或复现通过自动关闭。

## Standards 轴：PR #245 的三个具体缺口

1. **配置被接受但未执行**：`mf_archive/optimizer.py:107–114,128–176,242–244` 接受公共 Muon 参数，却覆盖执行方法后忽略原生 callbacks 及部分 `zero_rms_scale_mode` 语义。固定配方可拒绝不支持选项，不能静默接受；恢复参数组时也需校验。这属于候选接口问题，不是已有默认 JT 缺陷。
2. **新增子类入口绕过初始化**：`trainer/text_trainer.py:40–54` 仅做类型检查，再调用子类 `__new__`，未执行 `__init__`。带实例状态的合法子类通过类型检查后会在 `_setup` 报错。已用固定提交的实际构造方法 AST 在无后端环境复现：[探针](probe_text_trainer_contract.py)、[结果](text-trainer-contract-result.json)。应收窄或定义初始化协议，不能仅用无状态子类测试。
3. **安装目录文件不等于实际 runtime**：`mf_archive/model.py:228–230` 读固定路径版本文件，`legacy_gemm.py:30` 则由动态装载器选库。多版本 CANN 环境可能误拒绝/误放行，需核验实际库与 runtime。此项是支持边界风险，不表示旧实测环境错误。

已排除两个不应误报的候选：factory 的全局 JIT/determinism 设置来自基线；head GEMM 确有 `try/finally` 恢复编译/格式开关，且单线程限制已写明。固定归约、CPU 同步本身不能自动判为数学 bug。大量平行实现是维护风险，仍需支持寿命和责任评审。

[Standards 详细审查](standards-review.md)。本轴有 2 项 P2 契约问题、1 项 P2 支持范围风险；没有把命名和纯格式噪声列为阻塞因素。

## Spec / 算法轴：判据须先于对齐结果

- **token 目标**：以全局 token 平均梯度推导，不以另一框架的打印值当期望。
- **Muon dtype**：分开检查动量、预归一化、NS 迭代、scale、relay、最终写回。FP32 主参数不意味着所有 update 都是 FP32；完整 step 的实际 dtype 才能支持默认影响声明。
- **Muon 矩阵**：先定义逻辑矩阵与存储融合承诺。PR #243 的 JT 特定契约成立；Megatron 可选拆分、MF 更宽 schema 不能替代这个契约。
- **packed MTP**：独立文档需要文档内 shift；若外部 mask 已排除所有边界监督，需按实际流水线判断。没有对最新 MF 全量训练，不能声称所有配置都出错。
- **归一化与舍入**：旧共享 down 样本的 MF norm 标量更近 FP64，而不同 BF16 normalized 值全部是普通 HP 更近 FP64。更贴近归档 loss 不能证明每个算子更正确。

[Spec / 算法详细审查](spec-algorithm-review.md)。未发现 #180/#213/#243 固定 diff 引入新的前向、反向、状态或通信错误；#213 默认触发描述和 #192 当前必要性结论需要收紧。单卡证据不能扩展为所有 TP/EP 组合已验证。

## 每个问题的具体处置

每个更新正文都包含：当前版本与触发条件、独立期望、最新官方源码、真正建议的动作、验收和测试边界；原有实验/作者记录折叠保留并明确旧基线。

| issue | 当前准确判断 |
|---|---|
| #149 | 逻辑存储分组、旧 shape scale、条件 FP32 写回分别判断；当前 MF 每 piece scale 已正确，不把旧缺陷泛化到 master |
| #150 | 独立 packed 文档内 shift 正确；最新 MF 普通 roll 的风险需结合上层 mask，归档目标不作为 HP 默认 |
| #152 | router FP32 GEMM 已由 #148 合入；权重副本 dtype 是独立精度选择 |
| #153 | 当前 MF FusedRMSNorm 直接调用底层 rms_norm，不能用旧 training_graph 的 Python casts 描述它；内部精度需算子证据 |
| #154 | embedding 参数、输出、梯度累加和通信 dtype 分开；Megatron 的 FP32 residual 输出选项不证明表梯度精度 |
| #155 | MF 与 Megatron 都有融合/分离路径；不同 kernel 本身不是 bug，须用同定义与独立 forward/backward 容差判定 |
| #156 | 当前 JT 已有 FP32 rotary；标准 RoPE 共同位置偏移等价有前提，不推广到任意位置规则/缓存 |
| #187 | 当前 #232 已取消 K-shift 环境接管；#192 Draft 历史方案不再是必要修复；有限复跑不能证明普遍零代价 |
| #190 | 历史 301–600 与后续 201–400 不混用；不同背景消融收益不当作全局贡献率 |
| #210 | 可配置、可恢复的 NS 精度能力提案；不能据行业 dtype 选择认定 BF16 默认错误；不改其其它八卡工作流 |
| #215 | 触发限定为已有 FP32 update 进入单分片写回；显式写清覆写动量/NS 的来源 |
| #216 | 全局 token 平均目标推导的恒等梯度修复，已合入 |
| #217 | 总结产品修复与归档复现，纠正“正式提交可复现 ⇒ 都应合入”的结论 |
| #240 | 普通/归档/FP64 三方诊断，不把固定 DVM 调度推荐为默认 |
| #242 | JT 特定存储等价性依据；更宽 MF schema 和所有 TP/EP 验证均不夹带 |
| #244 | 旧归档复现 Draft，记录合入必要性和候选接口缺口 |

## 验证和下一步边界

本轮实际执行：官方 API 查询与固定源码读取、逐文件 SHA-256、固定 PR diff、两轴独立审查、无设备构造契约反例。现有 212/14/18/33 项测试及旧 200 步是历史已执行证据，未重新运行，不改写其基线、输入或结果。

本轮不为已达阈值的归档残差继续跑大实验，不为贴近旧 MF 额外改产品默认。未来新产品修改须先有明确契约或独立反例，再做最小修复与相应测试；候选 #245 在范围和接口问题解决前保持 Draft。CI、维护者评审和合入是独立状态，不以本报告冒充远端检查通过。

没有启动设备任务，没有停止任何任务，没有接触 zth 目录。仅保存少量源代码、metadata、审查文档和最小反例，没有新全量 dump 或 checkpoint。
