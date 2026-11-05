# 最终代码审查

审查对象：#1439 原 HEAD f96dd5b091f75ba4ec827f73e079b3b21e9174db 上冻结的六文件工作树差异；#1425 head 2fde96553707b4a01b2d6a471650d98f2bc19f4c；master 快照 c75f5f9f1e6d5ae157cbc7911a5beb893aaa2da4。此审查只读，没有启动 NPU。

结论：对所审范围未发现需要阻止整合的静态代码问题。公共 runtime callback、Xing4 接入、Muon hook 顺序和明确的 swap 不兼容 guard 组合在设计上可接受；多卡和实际硬件结果仍由独立回归决定。

## 复核依据

- 公共 MLA runtime 的 Q 和 K 在同步与 overlap 路径均采用指定 rotary callback；不指定时保留原 callback。attention 以 module.scaling 显式传递原尺度，附加 TP-local head offset 和 local_heads；默认 attention 以 **kwargs 接收，实际 context 提取仅处理已知字段。Plan 持续校验 scale、并行度与通信组合。
- Xing4 wrapper 委托现有 DeepSeek 通用 wrapper，暴露 strategy、overlap、ulysses_degree 和 kv_exchange；原 Ulysses 默认不变，旧 xing40 类和 MoE adapter 不重复引入。CP rank 的头范围经 head_rank × compute_heads 定位，callback 用 -inf 填充未拥有部分，clip 的 MAX 汇总恢复每个 TP-local head。该推理依赖实际并行 mesh 和 QK clipping reducer 契约。
- Xing4 非因果 attention 新增显式拒绝；现有 training 入口拒绝外部 attention mask 和 KV cache，内部 HF 本地因果 mask 在 CP wrapper 中被清除，由共享 runtime 确定全局 causal 边界。此入口不应解读为任意外部 mask 支持。
- Muon post hook 注册在实际 Muon leaf：该 leaf step 在混合精度 wrapper 的 FP32 main 参数复制回模型之前执行，clip_qk 选择 main_param；ownership guard 要求两 up-projection 都可训练且归 Muon，避免其他 leaf 更新后漏裁剪。swap 构造 guard 在 leaf 已有 step hook 时拒绝；本次 Xing4 builder 先注册 hook，故该组合 fail-fast。构造 guard 不证明之后注册的 hook 或全局 hook 的生命周期均被拒绝。
- 两个公共文件相对 #1425 做的是功能移植；仓库已有公共 MLA 投影、计划与 collectives，没有复制另一套 CP 算法。wrapper 留在原路径维持现有调用方兼容；以后迁移时应另做兼容导出。
- 相对 master 快照 c75f5f9，base a9fa47d6 的上游差异仅 CODEOWNERS；本轮六文件均属预期 scope。三个官方 HF transformer 文件相对原 HEAD 无工作树差异，索引 blob 保持原值。git diff --check 无输出。

## 验证及剩余风险

主代理报告外部 CPU callback 7 项已通过；本审查未重复执行，亦不据此宣称 NPU 成功。实际 NPU 测试应覆盖 CP2/4、TP 与 CP 组合、expanded 和 latent 策略、overlap、AllGather/selective-prefix、完整 Xing4 训练与 QK clipping 的 logits maxima、输出和梯度。尤其需确认不同 CP rank 的头位置和 TP shard 的全局 MAX，及 FSDP 下 main weight 复制时机；现有独立 callback 测试和公共 MLA 测试不能单独替代完整端到端结果。clip_qk 还要求各参与全局 MAX 的 rank 使用相同模块顺序和全局头布局；不应无验证地推广到不满足此契约的 pipeline 分区。

本报告不构成对提交历史、远端 PR 状态或任何尚未运行硬件实验的独立确认。
