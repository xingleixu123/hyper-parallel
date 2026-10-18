# Spec / 算法正确性只读复核（2026-10-18）

## 主结论

- **#180 保留**：局部 mean 的正确权重是 `n_micro / N_global * DP * CP`。先算比例可恢复单 rank、单微批的严格恒等梯度；数学依据独立于 MF。Megatron 从 token loss 总和累计梯度再除全局 token 数，不能直接移植它的一行归一化。
- **#213 改动方向可接受，但 #215 的触发范围与优先级证据需收紧**：跳过单分片 BF16 通信打包能保护已有 FP32 update；原生默认 Muon 的输入与返回更新已是 BF16，不能将归档/覆写路径的实测收益写成默认 JT 的普遍缺陷。
- **#243 可保留为 JT 存储等价性修复**：原模型明确把 Q 与 latent-KV、expert gate/up 融合存储，拆回原逻辑矩阵符合该契约。该结论不是“Megatron 默认如此”或“所有 fused Muon 都错”。仍需明确默认值迁移与单卡验证范围。
- **#192 不应原样合入**：#232 已取消 K-shift 环境接管，#192 默认 true 会重新覆盖调用者环境；应按已被另方案取代处理，并更新 #187 的现状。
- **#245 仅宜作为显式归档复现/诊断候选**：已知保留跨文档 MTP 与旧 Q-up scale，不构成通用正确性修复。支持降为 Draft，避免把旧 MF 轨迹达标等同于面向新训练的产品必要性。

## 审查边界和固定版本

只读审查，不修改仓库、issue 或 PR，不运行训练。只新增本报告。主工作区的无关修改未触碰。

| 对象 | 固定版本 / 来源 |
|---|---|
| HP 当前 JT_dev | `a1e6c222eab097c9609d6e744a3c5f98aeed6bdc` |
| PR180 | `3d381781589cd49680ab575535c6ccac30c81787` → `5de3cceab094bbed18915d1a9c9231ca220629ca` |
| PR213 | `521d3a8a9aa62353327093ab1258ca2c22006e8b` → `6f04715a9be134b1e544dcf3c8487bbd8051ffcf` |
| PR243 | `a1e6c222eab097c9609d6e744a3c5f98aeed6bdc` → `22c2e70969af168d594c5302dbbe200a1b85a173` |
| PR192 | 缓存固定 head `8e0722b5bfd209f5143d42afce1614f5f5c5b4b7` 的 GitHub files.patch；共享 Git 对象中该 head 不存在 |
| PR232 | 已合入，固定 head `d790be978d263b7f5ee31fab3c11937f8b9a5ea0` |
| MF 最新 GitCode master | `409019aac0c6579bb4fb8a9b04a75c259d98a96e`，本次根 agent 下载的 PyNative 源码 |
| Megatron 最新 main | `1effc75a46b26c6b93478355ff9fc3bb8ac1fafc`，本次根 agent 下载的源码 |

行业源码保存在本报告同级 `industry/`；PR/issue 正文和文件补丁在 `github/`。已阅读项目 code-style、distributed、project-overview 规则。本机 `python` 缺少 torch，本审查未重新执行 PyTorch UT；PR 正文中既有测试属于被审查证据，不冒记成本次执行结果。

## 180 / 216：token 恒等梯度

[固定修复](https://github.com/hf1s/hyper-parallel/blob/5de3cceab094bbed18915d1a9c9231ca220629ca/hyper_parallel/trainer/runtime/metrics.py#L79) 没有改变目标：设 rank r 的微批 b 包含 n_rb 个有效 token，局部 loss 为这些 token 的均值 l_rb；全局有效 token 总数为 N，FSDP 在 D=DP×CP 的域内平均梯度。应回传 `D * n_rb / N * l_rb`，从而平均后得到 `sum(n_rb * grad(l_rb)) / N`。单 rank 单微批时比值先计算为 1，避免旧反向链路先舍入 `1/n`、再乘 n。

日志仍通过 detached token 加权值归约；现有 SP 额外缩放、各目标 token 字段及 collective 次序保留。补丁没有引入新的 collective 或状态。现有 CPU 用例覆盖恒等、不等微批、DP×CP、SP 和命名目标；这不能替代真实多卡测试，但本局部变换有明确数学依据，未发现引入的算法错误。

[Megatron 最新 finalize_model_grads](https://github.com/NVIDIA/Megatron-LM/blob/1effc75a46b26c6b93478355ff9fc3bb8ac1fafc/megatron/core/distributed/finalize_model_grads.py) 的 per-token 分支在全局汇总 token 后用 `1 / safe_num_tokens` 缩放已累计的梯度。其入口为 token 总和，HP 入口为局部均值；两者需按完整链路比较。

## 213 / 215：保护 FP32 update 正确，但生产触发证据应补来源

[新增短路](https://github.com/hf1s/hyper-parallel/blob/6f04715a9be134b1e544dcf3c8487bbd8051ffcf/hyper_parallel/core/optimizer/muon.py#L1072) 在 `total_shard_size == 1` 时保留 `chunk_update_by_layout`、主参数 shape 映射、权重衰减和 LR 折叠，再复用相同 foreach 写回；多分片分支维持原有 BF16 pack/broadcast。单分片中所有 shard coord 都为零，生产调度会在本地计算 owned params 的更新，短路不会改变复制组广播的外层责任。未发现此改动引入的所有权、通信、状态或梯度问题。

但必须区分“写回函数接到 FP32 时会降精度”和“原生默认训练会接到 FP32”：

1. 同一固定 head 的 `muon.py:529` 将自定义 momentum 回调的返回值转 BF16；`546–547` 也将默认动量结果转 BF16。
2. `_compute_batched_ns_updates_fast` 在 `974–980` 写回并返回 `ns_inputs` 的 view；transform 分支在 `1050–1059` 同样恢复到原输入。即使正交化回调内部用了 FP32，也会被写回 BF16 输入。没有发现其它原生生产入口能绕开这条链路。
3. 本次集成源码中，明确产生 FP32 update 的入口是 PR245 的 `mf_archive/optimizer.py:128–176`：它覆写动量和 NS 方法，`archived_ns` 返回 FP32，最终 restored update 也为 FP32。
4. 新增 UT 手工向私有写回方法提供 FP32 update，能隔离证明本地写回额外 cast，但没有走完整原生 `Muon.step`。真实首步和 200 步实验若使用旧适配器/方法覆写，应明确列出；当前 #215 的“真实训练 24 份二维 FP32 update”未在该段标注产生它们的实现。

因此推荐保留这个小而合理的本地写回改进，但先将 issue/PR 的真实收益限定到可追踪的 FP32 更新来源，并调整其默认产品影响/P1 的论证。若坚持声明原生默认 JT 的 FP32 精度损失，应提供该固定版本完整 `step` 链路中产生 FP32 update 的可复现证据。不能把归档配方依赖直接说成原生默认普遍问题。

## 243 / 242：JT 特定存储边界契约成立

直接产品证据是 [JT 模型](https://github.com/hf1s/hyper-parallel/blob/a1e6c222eab097c9609d6e744a3c5f98aeed6bdc/hyper_parallel/models/jt_deepseek_v3/modeling_jt_deepseek_v3.py#L185)：类注释声明由原 `q_a_proj` 与 `kv_a_proj_with_mqa` 融合，203 行显式拼接两个原权重，216–220 行按同样边界拆分前向输出。expert gate/up 是 SwiGLU 中两个不同角色的投影，存为 `[E,2I,H]`。

[变换实现](https://github.com/hf1s/hyper-parallel/blob/22c2e70969af168d594c5302dbbe200a1b85a173/hyper_parallel/models/jt_deepseek_v3/adapter/runtime/muon_layout.py#L44) 仅对这两个已知名字切分，并在 shape 校验后拼回。NS 包含范数归一化及 Gram 多项式，对行拼接一般不满足可分配性：拼接后的 Gram 引入块间交叉项，整体范数与每块范数也不同。因此纯粹合并存储会改变优化器，独立参数与融合存储的逐矩阵更新等价是恰当验收条件。

动量本身逐元素线性，按原布局保存完整 momentum buffer 不妨碍上述拆分；此补丁不拆状态、不换参数、不重置 checkpoint。变换发生在 no_grad 的优化器内部，FSDP 需要时先 gather 完整矩阵再交变换；仅 expert 维分片时局部仍保有完整 gate/up 矩阵边界。未新增前向/反向运算和 collective，未发现新的流或可微通信问题。仍不能据静态推理和单卡 UT 声称全 TP/EP 组合已验证。

默认 true 会改变旧融合整体配方的下一次更新，现有明确 false 退出开关和回调冲突报错是必要的迁移说明。latent-KV 的 `[512+64,H]` 在原模型中属于同一个投影，保留它整体与本单契约一致；进一步拆成 latent/RoPE 或拆 Q/KV-up 都是另一个配方。

行业佐证必须保留限定：

- [MF 最新 PyNative schema](https://gitcode.com/mindspore/mindformers/blob/409019aac0c6579bb4fb8a9b04a75c259d98a96e/mindformers/pynative/base_models/gpt/gpt_model.py#L798) 在 823–867 行显式定义 expert gate/up、Q/KV/RoPE 和 Q/KV-up 分组，比本 PR 更宽。
- [Megatron 最新实现](https://github.com/NVIDIA/Megatron-LM/blob/1effc75a46b26c6b93478355ff9fc3bb8ac1fafc/megatron/core/optimizer/emerging_optimizers.py#L458) 先恢复未分片矩阵，再按配置拆 QKV，对每块独立调用 scaled orthogonalize，最后恢复布局。但 `split_qkv` 默认 false；GTP 下 blockwise/distributed 还有显式 unsplit fallback。这不能支持“所有融合整体 NS 都错”的普遍断言。
- [MF 最新 NS](https://gitcode.com/mindspore/mindformers/blob/409019aac0c6579bb4fb8a9b04a75c259d98a96e/mindformers/pynative/optimizer/muon.py#L1195) 是 FP32 norm/clamp 后进入 BF16 迭代、每 piece 按自身 shape scale。与 HP 或旧归档不同的 normalization、系数、算术次序并不自动构成错误；需分别声明优化配方。

## 192 / 187：当前默认已变化，原样合入会回退环境所有权

PR192 的固定补丁为 TrainingConfig 新增默认 true，并在 full determinism 时写 `CLOSE_MATMUL_K_SHIFT = '1' if flag else '0'`；连调用者已设值也会覆盖。

[已合入 PR232](https://github.com/hf1s/hyper-parallel/pull/232) 已移除该环境变量赋值。当前 JT_dev `trainer/runtime/random.py:57–65` 只设置确定性通信相关选项，明确把 K-shift 留给调用者。由此 PR192 的“默认保持原样”只适用于其旧 base；原样合入当前分支会恢复环境接管，与最新产品决策相冲突。

建议将 PR192 标记为被 PR232 取代，不把它纳入本次必要修复组合。#187 应在开头直接写出现状及替代 PR，并把旧逐位实验归入固定硬件/shape/版本的历史定位证据。当前正文仍包含“无条件设置”“清除开关代价为0”“换版本/宽度都不变”“建议10–30步替代长窗”等无范围断言；顶部的范围说明不足以让这些语句继续作为当前结论。有限复跑可证明当时样本可重复，不能证明所有平台普遍零代价，也不能替代用户既定200步验收。

## 245 的产品分类

本轴不把“对齐旧轨迹”作为数学正确性的裁判。PR245 自己说明保留旧 packed MTP 跨文档 shift、Q-up 最后分片 scale、固定 CANN 归约和 GEMM 舍入。它对研究历史状态迁移很有价值，但与修正独立文档训练目标/逐矩阵 shape scaling 的产品方向不同。支持将其保留为显式归档复现实验的 Draft；200步通过只限定该配方，不足以证明这2309行是通用训练产品必需改动。应独立审议维护范围，不为复现旧 MF 而恢复已修正的默认行为。