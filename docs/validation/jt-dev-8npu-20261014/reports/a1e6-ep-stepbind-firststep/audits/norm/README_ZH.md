# 原生梯度范数与 EP 首步差异：定向源码核对

结论：HP 的分组 FP32 归约可以产生正常的拓扑相关舍入差；现有 1 ULP 不能单独判为产品 bug，也不能解释全部首步差异。新同机 125 证据中，expert0 的 G0 已在裁剪前不同。优先验证这条上游计算路径，暂不为已可离线复算的 D/Q 裁剪边界另跑系数消融。

## 固定版本与计算路径

2026-10-18 14:56 UTC 经官方 HTTPS `git ls-remote` 重查：MF master 为 `409019aac0c6579bb4fb8a9b04a75c259d98a96e`；Megatron main 为 `62c063f2b731b57b1c2031b2b4f4c6a769271ccb`。HP 固定 `544526f7db8342c607244fd41009b2a74adb84c8`，clip_grad.py 完整原文 SHA256 为 `67a1e69bae376a75b7076c812df3691ab15edc4f5c3d36059f1384d422eb2b9b`。完整原文、Git blob SHA、函数行与取得方式在 [source-locations](source-locations-v1.json)；只重用已核缓存或获取缺失单文件，未更换 checkout/ref。

| 实现 | 梯度来源、计数与归约 | 范数、系数与应用 |
|---|---|---|
| [HP544 clip_grad.py](sources/hyperparallel/hyper_parallel/core/utils/clip_grad.py) L172–215、451–550、597–778 | 优先 main_grad，否则 grad；梯度 DTensor 布局优先，其次参数布局。Partial 先以临时 FP32 buffer 完成规定归约；Shard 按实际进程组签名合桶后 SUM，Replicate 不额外通信。不是按 expert 名称或一律 world SUM。 | L270–306 每 tensor norm 保持输入 dtype 路径；L538–548 把 scalar stack 转 FP32，求 norm 后取 p 次方，按组归约，再按桶顺序 FP32 相加。L789–813 计算并截断 `max_norm/(N+1e-6)`，转换到各梯度 dtype 后原位乘。 |
| [MF409 PyNative utils](https://gitcode.com/mindspore/mindformers/blob/409019aac0c6579bb4fb8a9b04a75c259d98a96e/mindformers/pynative/trainer/utils.py) L785–930 | main_grad 优先；每项平方和除以 Replicate 维大小及条件 PP 共享 embedding 计数，再在 enable_parallel 时做默认 world SUM。 | 每项显式 FP32 square/sum，堆叠相加后 sqrt；默认 max_norm=1、eps=1e-6。系数小于1才以其 item 值原位缩放。和 HP 的 norm→square、分桶/归约顺序不同。 |
| [Megatron62 optimizer](https://github.com/NVIDIA/Megatron-LM/blob/62c063f2b731b57b1c2031b2b4f4c6a769271ccb/megatron/core/optimizer/optimizer.py) L260–325、355–460；[clip_grads](https://github.com/NVIDIA/Megatron-LM/blob/62c063f2b731b57b1c2031b2b4f4c6a769271ccb/megatron/core/optimizer/clip_grads.py) L59–196 | 先过滤 shared、TP/ETP 及 GTP 复制份额；组由 optimizer 的 grad_stats_parallel_group 决定，DTensor 另有 DP 归约，不可泛称全部 world。 | p=2 走 TE/Apex/本地 fallback 选择的 multi-tensor norm，再平方、跨组 SUM、开根；系数同样有1e-6，但 tensor/fused 与 Python scalar 路径有别。源码符号存在不证明本次 PTA 二进制采用同一 kernel。 |

HP 的每 tensor norm 并没有统一先把所有输入转 FP32；不能只见外层 FP32 stack 就宣称 BF16 输入也全程 FP32。当前选中 G0 是 FP32，这一点不能外推全模型。相同数学 L2 需先满足唯一逻辑覆盖、归一化与 source layout 正确；即使满足，分片长度、根号再平方、foreach/本地实现、桶合并和 collective 树次序仍不保证跨拓扑 bitwise 相同。MF world SUM 的复制除权不能直接移植到 HP 的分组算法。

MTP 并非统一排除项：HP [trainer.prepare_optimizer_step L720–744](sources/hyperparallel/hyper_parallel/trainer/base.py) 传整个 model，按有效梯度参与；[apply_reduced_grad L459–515](sources/hyperparallel/hyper_parallel/core/fully_shard/hsdp_param.py) 保留归约后的 carrier/source layout。MF [apply_fsdp L1362–1655](https://gitcode.com/mindspore/mindformers/blob/409019aac0c6579bb4fb8a9b04a75c259d98a96e/mindformers/pynative/base_models/gpt/parallelize.py) 将普通/MTP dense 与 expert 用各自 fsdp/efsdp mesh，FP32 main_grad；共享 embedding 的 PP 同步/计数另有条件。Megatron 只有 `mtp_detach_heads=True` 时才给该分支打独立 norm-group 标签；[默认值 L98 为 False](sources/megatron/megatron/core/transformer/transformer_config.py)。不能拿可选组语义推断当前 JT MTP 被独立裁剪，也不能凭缺少同名过滤器认定 HP 重复计数。

## 已证范围与可证伪的下一步

旧 D 完整5120元素：FSDP N=22.473445892333984，EP N=22.473447799682617，相差1 FP32 ULP。各自 IEEE-FP32 系数分别为0.04449695348739624和0.04449694976210594；已有独审逐字节复算各自 G0→G1。它证明该 D 边界的缩放来源；D 首次更新权重仍完全相等，不能据此声称200步漂移已解释。

新同125 selected31：D5120、Q选中192×1536的 G0 exact；expert0 全7864320元素 G0 有7201234处差，max=1.1175870895385742e-8，故差异已早于裁剪。E0首次更新权重有636619处差，max=7.450580596923828e-9；Q选中片段首次更新11865处差，max=3.725290298461914e-9。Q不是完整矩阵，Muon更新依赖全矩阵，不能仅凭这片 G0 相等归罪 norm。证据输入 SHA 与选中边界列于 [evidence-boundaries](evidence-boundaries-v1.json)，本审未重复执行原生原件复算，未覆盖所有专家、MTP或全模型梯度。

下一条信息量更高的候选是：同一125初态/数据/配方，两侧只改变经官方支持范围核清的单一 matmul 内核开关，保留原 norm 与 clipping，首步先比 E0 G0 全元素，再比 G1/真实 optimizer 输入和更新。`CLOSE_MATMUL_K_SHIFT=1` 尚须核 CANN9.2 正式出处与实际 grouped matmul 路径覆盖；开关可设置、普通 matmul 有效，都不等于 `npu_grouped_matmul_add_` 已生效。若 E0 G0 完全不变，只能拒绝“该开关在本次有效路径足以消除差异”的假设；若变为 exact，仍须实际路径/覆盖证明和后续更新回归，不能直接称唯一根因。

系数固定1仅保留为较低优先级排除实验：先保留原 N2 全部计算和有限性检查，再只在首步用原 all_grads/foreach 顺序施加精确1，其他步原样；预期各自 G1=G0，D/Q片段跨拓扑 exact，而 E0 上游差保留。它不新增通信、不跨 run 借系数，但改变了训练配方，且大部分 D/Q 结论已能离线得到，不建议优先占卡。

独立 bug 的判据应是可证明的错 source/carrier、未规约 Partial、遗漏/重复逻辑梯度、错误组或系数应用；需完整每参数 dtype/布局/拥有者/归约组清单或原生证据，不能由1 ULP独断。125目前仅三步跨129参考 exact，尚无自己的正常无observer完整200 baseline；先建立同机基线，再谈因果臂和完整200×8验收。首轮三步与正常舍入均不豁免主LM最大差≤0.005要求，根因仍 UNRESOLVED。

此目录仅本机只读源码审计和新文档/原文快照；没有新增运行、请求、控制面或产品改动。
