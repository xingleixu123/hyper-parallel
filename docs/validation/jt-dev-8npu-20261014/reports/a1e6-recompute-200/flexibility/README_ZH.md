**JT 重算灵活性与模块成本：固定源码只读复核及最小改进方案**

本报告建议先用一个问题、一个 commit 的 PR 补齐 **JT 的 `ModelAdapterSpec.recompute` provider**，复用已经存在的层选择和 checkpoint wrapper。首版只提供固定安全子模块集合及其层范围；任意模块路径、排除项、通信重算和性能最优策略仍需分别设计与验证。本文没有实现 provider，也没有运行新的 CPU 语义测试、容器或 NPU 实验。

报告整理时间为 2026-10-18 03:54:20 UTC。源码审阅固定在 `544526f7db8342c607244fd41009b2a74adb84c8`，审阅时 HEAD 与干净工作区已核对；[来源清单](sources.json)记录每个本地文件及官方文件的完整 SHA256、精确行号和链接。清单中的“最新”指本轮读取远端 HEAD 时的快照，不代表之后分支不会更新。

主任务传来的接续状态为 full200 已完整 PASS、selective200 当时进行中。本文不独立复核这些运行回执，也不以它们证明全部并行矩阵、以下建议 provider 或模块 ROI 已通过。历史 `18602c4` 的重算观测也不作为当前源码验收。

**当前入口的事实**

| 事实 | 已读源码位置 |
|---|---|
| JT 注册没有 `recompute`，选择 safe-regions 会在通用入口报缺少 provider | [JT 注册](https://github.com/xingleixu123/hyper-parallel/blob/544526f7db8342c607244fd41009b2a74adb84c8/hyper_parallel/models/jt_deepseek_v3/adapter/registration.py) 39–48；[选择实现](https://github.com/xingleixu123/hyper-parallel/blob/544526f7db8342c607244fd41009b2a74adb84c8/hyper_parallel/distributed/activation_checkpoint.py) 748–768 |
| 已有 `RecomputePolicy`，包含安全模式与 no-replay 模式；正常训练不应依赖 validation 导入 | [adapter 契约](https://github.com/xingleixu123/hyper-parallel/blob/544526f7db8342c607244fd41009b2a74adb84c8/hyper_parallel/models/adapter_spec.py) 36–47、90–111 |
| schema 已有 `layer_count`、`layer_indices`，仅用于 `source: model_adapter_safe_regions` | [配置 schema](https://github.com/xingleixu123/hyper-parallel/blob/544526f7db8342c607244fd41009b2a74adb84c8/hyper_parallel/trainer/config/parallelism.py) 54–115 |
| 层号来自发现容器按注册顺序展平后的零基索引；前 N 层与离散索引均已有校验 | [选择实现](https://github.com/xingleixu123/hyper-parallel/blob/544526f7db8342c607244fd41009b2a74adb84c8/hyper_parallel/distributed/activation_checkpoint.py) 772–819、943–962 |
| safe-regions 在 full/selective 模式分支之前返回，helper 不传 SAC context | 同文件 839、888、1114–1123 |
| 默认 selective 是算子级 SAC：eager 的 matmul 交替保存/重算，命中昂贵/通信集合的输出保存；compile 对命中 matmul 全保存 | 同文件 350–412 |
| no-replay 模式只检查当前选中目标是否冲突，不自动递归排除子树，也不安装 exclude wrapper | 同文件 842–858 |
| 构建顺序为 sharding → activation features → FSDP → materialize/load；新策略应沿现有顺序 | [builder](https://github.com/xingleixu123/hyper-parallel/blob/544526f7db8342c607244fd41009b2a74adb84c8/hyper_parallel/models/_transformers/model_builder.py) 497–539 |

因此，**safe-regions 下 full 和 selective 当前都走所选子模块的普通 checkpoint，不是子区域 SAC**。首版示例采用 `mode: full` 并明确说明该语义。不能把原有 SAC selective 的名字解释成“已经支持 MF 风格模块选择”，也不应在 provider PR 中顺带改变公共模式行为。普通 wrapper 调用关系见 [wrapper](https://github.com/xingleixu123/hyper-parallel/blob/544526f7db8342c607244fd41009b2a74adb84c8/hyper_parallel/core/activation_memory/wrapper.py) 485–511。

现有 JT CPU 测试明确检查当前两层主干加一层 MTP decoder 的发现结果：`model.layers.0`、`model.layers.1`、`mtp.layers.0.transformer_layer`，对应 0、1、2（[测试](https://github.com/xingleixu123/hyper-parallel/blob/544526f7db8342c607244fd41009b2a74adb84c8/tests/ut/models/jt_deepseek_v3/test_modeling_jt_deepseek_v3.py) 163–168）。这是该结构的已读测试契约，**实际索引仍由容器发现顺序生成**，不应把“最后一层永远是2”写入 provider。

**建议的首版安全集合与状态边界**

给 `model.layers.*` 和 `mtp.layers.*.transformer_layer` 两个前缀各声明以下三个后缀：

```text
input_layernorm
post_attention_layernorm
mlp.shared_experts
```

这些名称来自 [JT 模型](https://github.com/xingleixu123/hyper-parallel/blob/544526f7db8342c607244fd41009b2a74adb84c8/hyper_parallel/models/jt_deepseek_v3/modeling_jt_deepseek_v3.py) 439–440、577、595–596、620–631。两个 decoder norm 是计算子模块，shared-expert MLP 在 routed/EP 分支外单独调用。当前配方结构预期六处 decoder norm 和两处 shared-expert MLP，安装后的准确 FQN 与数量仍须测试和运行日志核验，本文没有安装它们。

首版 no-replay 声明可覆盖两个前缀下的 `self_attn`、`mlp`、`mlp.gate`、`mlp.experts`。这些声明表示本策略不选择相应整个区域；不是“所有这些计算绝不能重放”，也不是现有 full 模式错误的证据。现有 matcher 只对被选中的 FQN 做模式冲突检查，父级 `mlp` 声明不会自动禁止安全的 `mlp.shared_experts` 子项。

| 边界 | 源码事实与首版处理 |
|---|---|
| Router logits | JT 模型 401–422 保存并消费 `router_logits`。不要只跳过数学计算或用 wrapper 属性赋值代替 owner 的消费方法。 |
| Aux 与 token counts | 450–474 中 aux/counts 数学仍执行，副状态赋值有 `is_recomputing()` 保护。新边界必须验证没有重复累计或保留反向图。 |
| Expert bias 更新 | 508–523 修改 bias 后清空 counts；[optimizer hook](https://github.com/xingleixu123/hyper-parallel/blob/544526f7db8342c607244fd41009b2a74adb84c8/hyper_parallel/models/jt_deepseek_v3/adapter/runtime/jt_optimizer.py) 124–134 在 optimizer 后执行。保持每个 optimizer step 一次。 |
| EP 分发/合并 | [EP 接口](https://github.com/xingleixu123/hyper-parallel/blob/544526f7db8342c607244fd41009b2a74adb84c8/hyper_parallel/models/jt_deepseek_v3/adapter/distributed/ep_compute.py) 40–49 把 `ep_compute` 绑定到 executor。首版不包 routed/EP 整块，也不承诺通信可重放。 |
| QK 统计 | JT 模型 176–189 更新 `max_logits_val`；optimizer hook 106–120 消费并清零。相同值下 maximum 可能幂等，仍不能据此保证任意 replay 边界安全。首版避开整 attention。 |
| MTP 外围 | [MTP](https://github.com/xingleixu123/hyper-parallel/blob/544526f7db8342c607244fd41009b2a74adb84c8/hyper_parallel/components/modules/mtp.py) 64–74 的 `hnorm/enorm/eh_proj/final_layernorm` 在内部 decoder 外；当前 selector 选的是 `transformer_layer`。首版不宣称覆盖这些外围模块或整个 MTP。 |

整个 dense MLP 是后续候选，但不能用 `*.mlp` 冒充“仅 dense”：相同路径也匹配 MoE，当前 provider 又不接收模型实例。不要把样例第0层为 dense 的事实硬编码成通用策略。单独包 Linear 或 norm 也未必带来值得保留的显存收益；这正是必须先测模块集合 ROI 的原因。

另外，[chunked CE](https://github.com/xingleixu123/hyper-parallel/blob/544526f7db8342c607244fd41009b2a74adb84c8/hyper_parallel/components/losses/chunked_cross_entropy.py) 297 对 `project_and_score` 独立 checkpoint，因此 decoder `off` 不等于全训练图没有重算。比较模块策略时必须固定 `enable_chunk_loss` 和 chunk size。[HP checkpoint API](https://github.com/xingleixu123/hyper-parallel/blob/544526f7db8342c607244fd41009b2a74adb84c8/hyper_parallel/core/activation_memory/api.py) 297–305 默认 `early_stop=True`，进入某区域不保证其所有算子完整重放一次。

**官方参考与可借鉴范围**

先阅读了用户指定的 Ascend 软件来源 memory，MF 优先检查官方 GitCode。2026-10-18 本轮 `git ls-remote` 得到 MF master `409019aac0c6579bb4fb8a9b04a75c259d98a96e`、Megatron main `65ac8216e481b8a2e10d0f366865528fc010342a`；固定提交文件随后通过官方 GitHub 镜像/仓库的 HTTP 响应在内存读取并计算 SHA256。没有把旧参考 checkout HEAD 当作最新。

| 官方来源 | 与本任务直接相关的能力 |
|---|---|
| [MF PyNative RecomputeConfig](https://github.com/mindspore-ai/mindformers/blob/409019aac0c6579bb4fb8a9b04a75c259d98a96e/mindformers/pynative/config/config.py#L853)，853–908 | 模块路径×层范围、exclude、独立通信选择；MTP 接在全局层号尾部，内部重模块另有 `transformer_layer` 层级。 |
| [MF PyNative apply_recompute](https://github.com/mindspore-ai/mindformers/blob/409019aac0c6579bb4fb8a9b04a75c259d98a96e/mindformers/pynative/distributed/activation_checkpoint.py#L791)，562–637、791–869 | 对照模型白名单展开路径，处理父子重复，先 exclude 再重算；full 使用 `early_stop=False`。这是 MindSpore Cell 运行时，不直接等价于 JT Torch。 |
| [Megatron 配置](https://github.com/NVIDIA/Megatron-LM/blob/65ac8216e481b8a2e10d0f366865528fc010342a/megatron/core/transformer/transformer_config.py#L630)，630–684 | 区分模块 selective 与 full 的层块策略；部分模块使用丢弃输出型 checkpoint，MLP/MoE/shared experts 使用普通 checkpoint。 |
| [Megatron shared experts](https://github.com/NVIDIA/Megatron-LM/blob/65ac8216e481b8a2e10d0f366865528fc010342a/megatron/core/transformer/moe/moe_layer.py#L606)，273–282、606–612 | shared experts 可单独 checkpoint，给首版边界划分提供参考。 |

**HP 普通 norm wrapper 不等于 Megatron 丢弃输出型 norm checkpoint**。后者还有输出释放和反向触发机制（[transformer_layer](https://github.com/NVIDIA/Megatron-LM/blob/65ac8216e481b8a2e10d0f366865528fc010342a/megatron/core/transformer/transformer_layer.py#L1171) 1171–1185、1590–1595）；不能借其接口名称或论文预期给 HP norm 策略背书。MF 的灵活选择说明可改进的配置方向，但首个 provider PR 只完成固定安全集合×层范围，不宣称达到其全部功能。官方文件完整哈希见 sources.json；MF 两份已读文件与前次核对内容相同，Megatron 配置应以本次哈希为准。

**一个问题、一个 commit 的最小 PR 方案**

问题应准确表述为“JT 缺少重算 provider，不能使用已有 adapter-safe 层选择”，不要表述为“HP 不支持选择重算”。

建议单提交包含：

- 新增 `models/jt_deepseek_v3/adapter/policies/activation_checkpointing.py`，返回上述策略，避免 NPU 或 validation-only 导入；注册到 JT spec。已有 [V4.1 provider](https://github.com/xingleixu123/hyper-parallel/blob/544526f7db8342c607244fd41009b2a74adb84c8/hyper_parallel/models/deepseek_v41/adapter/policies/activation_checkpointing.py) 20–29 可参考数据声明形式，不能照抄其模块安全判断。
- 在现有 JT CPU 测试中通过真实公共 `_apply_activation_checkpointing` 验证安装；不 mock checkpoint、router 或整个 forward。只在 NPU kernel 边界使用现有可微 CPU grouped-SwiGLU reference。
- 补充中文配置说明和边界，不修改 checkpoint core、生产配方默认 mode、guard、runner 或已冻结请求。

正确的 CPU 测试 seam 见 [JT 测试](https://github.com/xingleixu123/hyper-parallel/blob/544526f7db8342c607244fd41009b2a74adb84c8/tests/ut/models/jt_deepseek_v3/test_modeling_jt_deepseek_v3.py) 51–91、113–150、153–195：真实小模型、CPU device type、实际 autograd、仅 grouped kernel 替身。新增测试应验证 `[0]`、`[1]`、`[2]`、`[0,2]`、`layer_count=0` 的 FQN、未选区域、MTP 覆盖及梯度；越界、重复层号、嵌套/冲突策略继续拒绝。

数值测试至少覆盖三 microbatch 累积，对比 LM/MTP/aux、全部参数梯度、token counts、aux/router引用释放，以及一次 bias 更新后的值。以 forward/recompute 标记核验 attention/router 未因本安全子区域策略额外执行；不要要求每个 post-hook 都执行两次，因为 early stop 可以截断 replay。已有 CPU 测试源码是测试设计依据，本文没有执行新增测试，更不能据此认定 NPU fused kernels 或 TP/CP/EP 已通过。

示例仅说明拟议 provider 完成后的接口，不是可用于当前544直接运行的配置：

```yaml
activation_checkpoint:
  mode: full
  swap_inputs: false
  selection:
    source: model_adapter_safe_regions
    layer_indices: [0, 2]
```

**模块集合 ROI 的实测方案**

先测边界与成本，再决定是否扩大安全集合。所有 A/B 固定源码、初态、数据、GBS/MBS、dtype、优化器、LR、aux/MTP系数、chunk size 和拓扑；仅改变重算选择。当前运行请求和源码不因此改动。新 provider 分支需要新的固定源与相应基线证据，不能把历史源或其它拓扑成绩外推。

先以短诊断比较 off、norm集合、shared-expert集合、二者组合，再与默认 full/SAC selective 对照；这些是待构造的诊断方案，本文没有生成实验请求。完整验收仍按主任务约定保存200步×8rank主 LM差不超过0.005的证据，MTP/aux观测不扩大为其专项或全矩阵通过。

已有 [module_profile.py](../module_profile.py) 55–105 使用 NPU Event，区分 forward/recompute，在短窗口收集 inclusive 耗时与 allocation delta。可复用采集方式，但解释必须保持准确：

| 指标 | 正确用途与限制 |
|---|---|
| 模块 inclusive NPU Event 时间 | 定位耗时区；父子模块时间不能直接相加。多 stream 或通信重叠还需短 trace 验证归因。 |
| 模块前后 allocation delta | 只是分配变化，不等于该模块保留激活，也不是其全局峰值贡献。 |
| 每步峰值 allocated/reserved | 记录每rank及最差rank；分清 allocator 保留与实际分配，统一测量窗口。 |
| 全步耗时、tokens/s | 在相同样本/有效token口径下比较，报告瓶颈rank与稳态分布；排除 warmup、checkpoint I/O。 |
| 边际收益 | 计算开关某集合后峰值减少的 GiB 与步时增加的 ms，报告 Pareto 候选，不预设哪个集合最优。 |

profiling 应为独立短跑，避免每步同步污染正式性能数据。正式性能用多次交错运行区分共享机噪声，不能靠单次不足1%的差异定优劣；200步精度证据与性能窗口分别说明。按模块数据支持层选择，实际保留哪些 norm/MLP 区域由测得收益决定。只保留必要 trace、摘要与复现证据，确认用途后清理无用中间物；本报告不执行这些清理。

**操作边界记录**

本次源码分析起初对已有 `blob:none` 官方参考仓执行了 `git cat-file -t` 查询新 HEAD。之后观察到参考仓 `.git/objects/pack` 中有本地时间 2026-10-18 09:25:43 的新 promisor pack，可能由 Git 自动补取对象产生；未独立追踪其因果归属。因此不能把整个参考读取过程断言为绝对零磁盘写入。发现后停止 Git 对象读取，官方文件改用内存 HTTP；没有继续 Git 补取或清理参考缓存。

本次获准写入仅为本目录的 README_ZH.md 和 sources.json。没有实现 provider、修改固定源码、guard、runner、请求或正在运行实验的输入，也没有新建容器或运行 NPU。本文的结论是可审查的源码事实、改进建议与测量设计，硬件效果仍待后续证据。
