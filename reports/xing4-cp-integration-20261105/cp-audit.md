# PR #1425 CP 整合只读审查

审查基点：#1425 head 2fde96553707b4a01b2d6a471650d98f2bc19f4c，merge-base d586665915709ef058de8d15e7b8db16e68caf60；#1439 head f96dd5b091f75ba4ec827f73e079b3b21e9174db；master 快照 c75f5f9f1e6d5ae157cbc7911a5beb893aaa2da4。GitHub files 快照确认 #1425 四文件。master 相对 #1439 base a9fa47d6 在两个公共文件没有差异。#1439 base 和 #1425 head 非祖先关系，不能用其直接差异代替 PR diff。

## 逐文件

- hyper_parallel/distributed/context_parallel/mla_context_parallel.py：必须移植可选 rotary/attention callbacks 和 TP-local head 元数据。当前 runtime 写死 RoPE 和普通 fused attention；同步、expanded overlap、latent overlap 三处 query 与 key 都要调用 rotary callback；attention 传 head_rank * compute_heads 和 local_heads，默认保持原行为。保留当前公开 apply_mla_rope。仅结构整理的 _attention_inputs 可以省略。
- hyper_parallel/models/deepseek_v3/adapter/distributed/context_parallel.py：必须增加并透传两个可选 callback。现存 wrapper 的投影及 mesh 校验和 forward rewrite 均通用；本轮保留现有路径以维持导入契约并避免无关搬迁，后续如迁公共路径可保留兼容 re-export。旧 PR 的 _make_cp_forward 仅结构整理，并非功能必需。
- 旧 xing40/adapter/distributed/context_parallel.py：不拷贝，依赖未合并 #1413 的旧模型。#1439 的 xing_4 adapter 已有 FP32 rotary、TND attention/maxima、Ulysses 头偏移和 CP wrapper；目前公共接口缺失会导致调用 TypeError。现有 wrapper 只暴露 ulysses_degree；完整覆盖 #1425 显式策略能力应透传 strategy、overlap、kv_exchange，保留 #1439 现有 Ulysses 默认 None，公共 MLACPPlan 已验证头数整除和 TP/hybrid/KV exchange 限制。
- 旧 xing40/adapter/distributed/moe_context_parallel.py：不拷贝，旧对象布局不兼容。现有 xing_4 wrapper 将 group 放到 gate.aux_loss_cp_group；Router 已归约 counts/tokens 和可微归约 affinity，覆盖原目的。

## 风险与验证

training.py 拒绝外部 attention mask 和 KV cache，随后生成本地 HF causal mask；CP wrapper 清除内部 mask 并由共享 CP 建立全局因果边界有依据，不应推断支持任意外部 mask 或 packed 输入。QK clip 须验证 TP×CP 下各 TP-local head maxima 经 CP MAX 合并。FP32 RoPE 需在 Q、K 和不同 overlap 路径对比非 CP 基准。实际 NPU TND 统计形状、Q/K 不等长和 packed 元数据需运行验证。

DS4 shared attention CP 包装模型逻辑并复用公共 collectives，Qwen3.5 wrapper 委托公共 GDN executor 并传算法选项；建议同样使用 Xing 薄适配和公共 MLA，不复制旧模型实现。CP shift token 与按 MTP depth 全局有效 token 归一化是 #1439 训练语义，#1425 未覆盖；需与 Trainer batch sharding、DP/CP 缩放端到端核对。

建议仓外验证：非 CP 与 CP2/TP1、CP2/TP2 logits/loss、逐头 maxima 和梯度；覆盖两种策略、overlap、合法 AllGather/hybrid/all-to-all-v 及非法配置，记录设备、seed、精度和容差。无 UT/ST 文件进入提交；三个官方 HF 文件保持字节不变。