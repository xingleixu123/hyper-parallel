# JT 重计算灵活度与模块成本复核 v2

检视时间：2026-10-16T04:59:28+08:00。独立 Standards／实现与证据轴；仅阅读固定源码和已有证据、查询官方远端 HEAD。没有运行容器、训练或设备实验，没有修改生产源码、原始记录、PR 或 issue。

当前 JT 已能直接配置 off／full／selective，且三组 FSDP8 完整 200 步的已记录 LM、MTP、aux（包括还原系数后）、total loss 与 grad norm 差值均为 0。**当前 YAML 不能直接表达 MF PyNative 的“模块路径 × 层范围 + exclude + 通信选择”组合，但底层重算和排除原语已经存在。** 优先复用现有配置及 wrapper；若要提供默认的 JT 安全子区域，最小产品改动是接入已有 adapter provider，不是重写 checkpoint 核心。

本结论限定于集成提交 `18602c4524dc664b81b2bad9a7b624455ce1494f`。FSDP8 的这三组观测不等价于 CP／TP／EP 各种拓扑的重算验收，也没有确定通用最优策略。full 比 selective 多省 0.482707977 GiB allocated，单次中位步时仅多 0.249083%（约 3.069 ms），不足以判定稳定速度差别。

## 1. 固定版本与最新官方源码

| 对象 | 本轮核对版本 | 读取方式与边界 |
|---|---|---|
| JT 实验源 | `18602c4524dc664b81b2bad9a7b624455ce1494f` | 本地 `fixes/latest-jt-validation-f637`，HEAD 与工作区干净状态均核对；不改变实验固定源 |
| MF PyNative | `5089da308ebcc146d0cbe32180c209c4f1d39d77` | 官方 GitHub `refs/heads/master` 的本轮 `git ls-remote` 结果；使用已抓取的 Git 对象读取固定文件 |
| Megatron Core | `3914483fd325e053797aff3fa431d4a3141139aa` | 官方 GitHub `refs/heads/main` 的本轮 `git ls-remote` 结果；使用已抓取的 Git 对象读取固定文件 |

本轮官方 HEAD 与先前保存的最新核对一致。参考 checkout 的工作区 HEAD 仍为 MF `302c939f1f6c3335c533f8c83cdce67db671f607`、Megatron `07046cc6b8b7479ba05e6b47ed25f3f8e6634573`；报告引用的新提交来自 `git show <固定提交>:<文件>`，没有将旧 checkout HEAD 冒称最新。MF 的 PyNative 重算实现文件两版本字节相同，配置文件仅修改数据均衡说明；Megatron 的 `recompute.py`、`transformer_layer.py`、`moe_layer.py` 字节相同，配置变化主要是 DSA／推理图约束，本报告重新核对了新提交的重算字段及校验。已有差异记录：[MF/Megatron 文件差异](<D:/Obsidian Vault D/hyperllal/jt-dev-8npu-20261014/evidence/recompute-reference-implementation-check.json>)、[Megatron 10 月 16 日实现复核](<D:/Obsidian Vault D/hyperllal/jt-dev-8npu-20261014/evidence/megatron-recompute-implementation-recheck-20261016.json>)。

官方依据限定在 `mindformers/pynative/distributed/activation_checkpoint.py`、`mindformers/pynative/config/config.py` 和 Megatron Core 实现。MF 示例里的 `select_recompute`、`mp_comm_recompute`、`parallel_optimizer_comm_recompute` 等旧静态图配置不能作为此处 PyNative 接口证据；MF PyNative 导入的是 MindSpore Cell 与当时 HyperParallel 的 `core.activation_checkpoint`，当前 JT 使用 Torch 的 `core.activation_memory`，运行时契约不能按名称直接互换。见 [MF PyNative 导入与 wrapper 依赖](https://github.com/mindspore-ai/mindformers/blob/5089da308ebcc146d0cbe32180c209c4f1d39d77/mindformers/pynative/distributed/activation_checkpoint.py#L15)。

## 2. 当前配置实际能做到什么

| 需求 | 当前 JT 普通训练 YAML | 已有底层／扩展机制 | 判断 |
|---|---|---|---|
| 关闭 decoder 重算 | `activation_checkpoint.mode: off` | builder 跳过 activation checkpoint 安装 | 已有，可直接使用；chunked head/CE 独立重算仍保留 |
| 所有发现的 decoder 全量重算 | `mode: full`、`selection.source: default` | 每个 block 安装 HP 非重入 wrapper | 已有，当前模型 3 处结构覆盖已核对 |
| selective 算子策略 | `mode: selective`、`selection.source: default` | 每个 block 内按 op 保存／重算 | 已有；不是按模块名称选择 |
| 前 N 层或离散层号 | schema 有 `layer_count`／`layer_indices` | 只能配合 `model_adapter_safe_regions`，且模型须声明 provider | 通用框架已有，当前 JT 尚不能直接启用 |
| 任意 FQN／通配模块、按层组合 | activation 配置无此字段 | 公开 wrapper 可对已解析模块使用；通用 plan override 有 FQN/glob 与自定义工厂扩展 | 无开箱即用的 JT 重算 YAML 映射；实验可复用原语，需要少量 Python 选择逻辑 |
| 排除某子模块重放 | activation 配置无用户 `exclude` 字段 | 公开 `checkpoint_exclude_wrapper` 已有 | 不能说“HP 不支持排除”；缺的是用户配置到模块／算子的映射与模型约束 |
| 用户指定保存／重算哪些通信 | 没有 `recompute_comm` 配置 | selective 有固定通信 op 集合，另有 FSDP 生命周期 op 忽略规则 | 不等价于 MF 的显式按通信路径选择 |
| activation 输入换出 | `swap_inputs: true` 可传入 full/selective | eager wrapper 支持；compile 路径告警禁用 | 已有的独立优化轴，本轮三组均关闭，不是重算性价比证据 |

配置入口与传递依据：[模式和选择 schema](<D:/Obsidian Vault D/hyperllal/jt-dev-8npu-20261014/fixes/latest-jt-validation-f637/hyper_parallel/trainer/config/parallelism.py:54>)、[builder 模式门禁](<D:/Obsidian Vault D/hyperllal/jt-dev-8npu-20261014/fixes/latest-jt-validation-f637/hyper_parallel/models/_transformers/model_builder.py:380>)、[Trainer 传递配置](<D:/Obsidian Vault D/hyperllal/jt-dev-8npu-20261014/fixes/latest-jt-validation-f637/hyper_parallel/trainer/base.py:300>)。保持当前实验口径的最小配置是：

```yaml
activation_checkpoint:
  mode: full  # 比较时分别改为 off / selective
  swap_inputs: false
  selection:
    source: default
compile:
  enabled: false
```

不要在当前 JT 配置中直接添加 `layer_indices` 并期望生效：`source: default` 会拒绝层选择字段，切到 `model_adapter_safe_regions` 又要求 `ModelAdapterSpec.recompute`；[JT adapter 注册](<D:/Obsidian Vault D/hyperllal/jt-dev-8npu-20261014/fixes/latest-jt-validation-f637/hyper_parallel/models/jt_deepseek_v3/adapter/registration.py:39>)没有该 provider。已有 [RecomputePolicy 领域声明](<D:/Obsidian Vault D/hyperllal/jt-dev-8npu-20261014/fixes/latest-jt-validation-f637/hyper_parallel/models/adapter_spec.py:36>) 与 [V4.1 的实际 provider](<D:/Obsidian Vault D/hyperllal/jt-dev-8npu-20261014/fixes/latest-jt-validation-f637/hyper_parallel/models/deepseek_v41/adapter/policies/activation_checkpointing.py:20>)可供接入参考，不能把 V4.1 的能力归为 JT 已完成。

层号是**发现容器按注册顺序展平后的零基索引**，不是不加说明就可套用的模型原始全局层号。当前 2 层主干 + 1 层 MTP decoder 的发现顺序为 0、1、2；未来模型深度、别名或容器结构改变时应输出实际 FQN 清单再解释索引。`layer_count` 是前缀，`layer_indices` 是显式集合；`layer_count=0` 合法但仍先解析 adapter。见 [层选择实现](<D:/Obsidian Vault D/hyperllal/jt-dev-8npu-20261014/fixes/latest-jt-validation-f637/hyper_parallel/distributed/activation_checkpoint.py:772>)。

通用 `plan_overrides.match` 本来支持 FQN/glob，`inner_wrapper` 也能指向用户的装饰工厂，[通用路径与工厂入口](<D:/Obsidian Vault D/hyperllal/jt-dev-8npu-20261014/fixes/latest-jt-validation-f637/hyper_parallel/trainer/config/parallelism.py:151>)；但当前内置注册项都是 CP 方案，[实际内置 wrapper 注册表](<D:/Obsidian Vault D/hyperllal/jt-dev-8npu-20261014/fixes/latest-jt-validation-f637/hyper_parallel/distributed/context_parallel/wrappers.py:1247>)没有现成的重算选择工厂。该机制说明无需另造所有匹配基础设施，不证明目前只填一个内置名字就能完成 JT 路径重算。不要为一次重算实验改动参数／I/O sharding contract 或换掉 CP wrapper；直接复用重算 API 的实验安装点更容易隔离变量。
## 3. full／selective 的真实实现与覆盖

两种模式在当前 JT 都进入 HP 非重入路径。JT 明确关闭 HF native gradient checkpointing，因为 router aux 通过副状态离开 decoder，HF reentrant 路径会破坏该目标的梯度；见 [JT 对 HF 路径的约束](<D:/Obsidian Vault D/hyperllal/jt-dev-8npu-20261014/fixes/latest-jt-validation-f637/hyper_parallel/models/jt_deepseek_v3/modeling_jt_deepseek_v3.py:653>)。full 为整个发现的 block 设置 checkpoint，selective 使用相同 block 边界但附加算子策略，见 [selective 安装](<D:/Obsidian Vault D/hyperllal/jt-dev-8npu-20261014/fixes/latest-jt-validation-f637/hyper_parallel/distributed/activation_checkpoint.py:966>)、[full 安装](<D:/Obsidian Vault D/hyperllal/jt-dev-8npu-20261014/fixes/latest-jt-validation-f637/hyper_parallel/distributed/activation_checkpoint.py:1028>)。

当前 eager selective 的实际规则是：每次 checkpoint 调用独立初始化前向／重放计数；命中的 matmul 类算子按调用序号交替保存和重算，命中的昂贵算子及通信输出必须保存，`topk` 强制重算，其余默认重算。compile selective 则是无状态策略，命中的 matmul 全保存。**eager 与 compile 的 `selective` 不是同一策略。** 源码：[算子清单](<D:/Obsidian Vault D/hyperllal/jt-dev-8npu-20261014/fixes/latest-jt-validation-f637/hyper_parallel/distributed/activation_checkpoint.py:75>)、[eager 策略](<D:/Obsidian Vault D/hyperllal/jt-dev-8npu-20261014/fixes/latest-jt-validation-f637/hyper_parallel/distributed/activation_checkpoint.py:377>)、[compile 策略](<D:/Obsidian Vault D/hyperllal/jt-dev-8npu-20261014/fixes/latest-jt-validation-f637/hyper_parallel/distributed/activation_checkpoint.py:350>)。

这些集合按实际存在的 Torch op overload 构建，部分来自 Torch 私有 compute-intensive 列表；不能把“源码列出昂贵 op”推成“当前 NPU 的所有昂贵 kernel 都已命中”。例如 JT 调用 `torch_npu.npu_fusion_attention`，配置集合写的是 `npu.npu_fusion_attention_v3`，必须由当前运行时分派确认两者的对应；只比较 Python 名字不能认定命中或认定 bug。本轮 200 步没有 op 级 save/replay trace，所以不据此解释全部性能差距。见 [JT 融合 attention 入口](<D:/Obsidian Vault D/hyperllal/jt-dev-8npu-20261014/fixes/latest-jt-validation-f637/hyper_parallel/models/jt_deepseek_v3/modeling_jt_deepseek_v3.py:175>)。

FSDP 的参数 all-gather、复制、临时分配等还被专门排除于 SAC 匹配账本，原因是前向预取与重放的参数准备位置可能不同。该“ignore”保留这些操作的正常执行，不是缓存其输出、也不是禁止通信重放。见 [FSDP SAC 生命周期规则](<D:/Obsidian Vault D/hyperllal/jt-dev-8npu-20261014/fixes/latest-jt-validation-f637/hyper_parallel/distributed/activation_checkpoint.py:325>)。因此任意 TP／CP／EP 通信选择都要独立证明 collective 次序、stream 完成及 backward 所需状态，不能把固定 MUST_SAVE 集合或 ignore 集合解释为统一的“通信全关闭”。

当前完整 200 步 full 与 selective 各有以下 3 个 wrapper 区域：

| FQN | 实际结构类型 | 包含／不包含的范围 |
|---|---|---|
| `model.layers.0` | `abc.HSDPCheckpointWrapper` | 主干第 0 个 decoder：norm、attention、MLP、残差 |
| `model.layers.1` | `abc.HSDPCheckpointWrapper` | 主干第 1 个 decoder：含 MoE |
| `mtp.layers.0.transformer_layer` | `hyper_parallel.core.activation_memory.wrapper.CheckpointWrapper` | MTP 内部 decoder；不等于整个外层 MTP 模块 |

8 个 rank 的安装日志均记为 3 处，结构快照也为 3 处。旧 `integration/check/activation_checkpoint.json` 只计到 1 是已知 issue229 的动态 FSDP 子类识别遗漏，不能当作只重算了 MTP。证据与各 rank 日志位置在 [三种模式资源与覆盖独立审计](<D:/Obsidian Vault D/hyperllal/jt-dev-8npu-20261014/evidence/f637-fsdp8-selective-r3-resource-independent-review.json>)。

“full”不是整个训练图每个算子都重复执行：embedding、主干末 norm、外层 MTP 的 `hnorm/enorm/eh_proj/final_layernorm` 不在上述 decoder wrapper 边界中，[MTP 外围与 decoder 的次序](<D:/Obsidian Vault D/hyperllal/jt-dev-8npu-20261014/fixes/latest-jt-validation-f637/hyper_parallel/components/modules/mtp.py:38>)。HP 非重入 checkpoint 默认 `early_stop=True`，拿到 backward 需要的张量后可以提前退出重放，[checkpoint API 的早停契约](<D:/Obsidian Vault D/hyperllal/jt-dev-8npu-20261014/fixes/latest-jt-validation-f637/hyper_parallel/core/activation_memory/api.py:297>)；结构覆盖与完整训练精度证据不等于每个 kernel 恰好重放一次。

本配方三组 `enable_chunk_loss=true`、`chunk_loss_size=4096`。head/CE 的 `projected_cross_entropy` 无条件按 chunk 设置独立 checkpoint，因此 off 的含义是关闭本节 decoder 重算，并未关闭所有重计算。见 [投影与 CE 独立 checkpoint](<D:/Obsidian Vault D/hyperllal/jt-dev-8npu-20261014/fixes/latest-jt-validation-f637/hyper_parallel/components/losses/chunked_cross_entropy.py:227>)。改变 chunk size 会同时改变峰值、反向调度和矩阵形状，不应混入模块重算 A/B。

router 的 aux 和 token-count 副状态已有重放标记保护，见 [重放保持数学但不重复累加副状态](<D:/Obsidian Vault D/hyperllal/jt-dev-8npu-20261014/fixes/latest-jt-validation-f637/hyper_parallel/models/jt_deepseek_v3/modeling_jt_deepseek_v3.py:450>)；任何新的 wrapper 边界仍需检查这类 once-only 状态及 QK clip 统计、RNG、共享参数、冻结参数和梯度累积，不能只检查前向 loss。
## 4. MF PyNative 与 Megatron 对照：可以借鉴的精确语义

**MF PyNative 当前实现。** [RecomputeConfig / RecomputeCommConfig](https://github.com/mindspore-ai/mindformers/blob/5089da308ebcc146d0cbe32180c209c4f1d39d77/mindformers/pynative/config/config.py#L853)提供 `mode: None/full/select`、`full_recompute_layer`、`select_module`、`exclude_op`，及独立 `recompute_comm.enable/select_module`。层范围为单层或闭区间，例如 `"0-1"`；校验要求范围合法、有序且不重叠。MTP 层号接在主干之后，但选 MTP 内部 attention 时路径须包含 `transformer_layer`。这些是 PyNative 实际 schema，不引用静态图字段。

`select_module` 与 `exclude_op` 将路径模式映射到层范围，对模型暴露的白名单展开；匹配使用带超时的 `regex.fullmatch`，不是 HP `fnmatch` glob。父模块覆盖子模块时去重，full 层也会覆盖其子模块选择。见 [路径正则匹配](https://github.com/mindspore-ai/mindformers/blob/5089da308ebcc146d0cbe32180c209c4f1d39d77/mindformers/pynative/distributed/activation_checkpoint.py#L450)、[白名单与展开](https://github.com/mindspore-ai/mindformers/blob/5089da308ebcc146d0cbe32180c209c4f1d39d77/mindformers/pynative/distributed/activation_checkpoint.py#L512)、[父子覆盖去重](https://github.com/mindspore-ai/mindformers/blob/5089da308ebcc146d0cbe32180c209c4f1d39d77/mindformers/pynative/distributed/activation_checkpoint.py#L588)。不匹配的模式会警告，不能把“配置存在”当成“目标确实安装”。

执行顺序为先 exclude wrapper，再 full、select、comm。full 的 `early_stop=False` 是因为整层包含通信及自定义 autograd 边界，副作用必须重放完整；选子 Cell 的 wrapper 默认仍可早停。见 [MF 实际安装顺序和 full 早停设置](https://github.com/mindspore-ai/mindformers/blob/5089da308ebcc146d0cbe32180c209c4f1d39d77/mindformers/pynative/distributed/activation_checkpoint.py#L791)。exclude 能处理子 Cell、登记的 `_comm_ops` 槽、可替换的 callable 属性或方法；这是**保存排除区域及复用结果**，不是简单不包一个子节点，见 [MF 排除 wrapper 的实际目标](https://github.com/mindspore-ai/mindformers/blob/5089da308ebcc146d0cbe32180c209c4f1d39d77/mindformers/pynative/distributed/activation_checkpoint.py#L706)。

通信配置也有边界：实现期望目标为可替换通信算子，若最终命中 Cell，会记录该通信配置不生效；独立配置项不能自动保证任意 collective 可安全重放。见 [通信选择实际安装](https://github.com/mindspore-ai/mindformers/blob/5089da308ebcc146d0cbe32180c209c4f1d39d77/mindformers/pynative/distributed/activation_checkpoint.py#L654)。MF 给算子传 `output_recompute=True`，当前 HP Torch checkpoint 没有该控制参数，未知 kwargs 会继续传给被包 callable，见 [当前 Torch checkpoint 参数与调用](<D:/Obsidian Vault D/hyperllal/jt-dev-8npu-20261014/fixes/latest-jt-validation-f637/hyper_parallel/core/activation_memory/checkpoint.py:578>)。不要把 MF 的该调用直接复制为当前 JT 配方。

**Megatron Core 当前实现。** [重算配置与模块清单](https://github.com/NVIDIA/Megatron-LM/blob/3914483fd325e053797aff3fa431d4a3141139aa/megatron/core/transformer/transformer_config.py#L630)支持 full 的 uniform 分组、block 指定每个 pipeline stage 的重算层数；实际执行在 [uniform/block 循环](https://github.com/NVIDIA/Megatron-LM/blob/3914483fd325e053797aff3fa431d4a3141139aa/megatron/core/recompute.py#L196)。配置文档虽然有 method 为 None 时的说明，当前校验对 full 实际要求 method 和层数，配置应服从 [配置校验](https://github.com/NVIDIA/Megatron-LM/blob/3914483fd325e053797aff3fa431d4a3141139aa/megatron/core/transformer/transformer_config.py#L2451)。

selective 的配置使用预定义模块类别，例如 `core_attn`、`layernorm`、`mla_up_proj`、`mlp`、`moe`、`moe_act`、`shared_experts`；默认是 `core_attn`，selective 不接受 `recompute_num_layers`。其中 `mlp` 对 dense MLP 生效，MoE 是另一类别，见 [dense MLP 的 gate](https://github.com/NVIDIA/Megatron-LM/blob/3914483fd325e053797aff3fa431d4a3141139aa/megatron/core/transformer/transformer_layer.py#L576)、[MoE 的 gate](https://github.com/NVIDIA/Megatron-LM/blob/3914483fd325e053797aff3fa431d4a3141139aa/megatron/core/transformer/moe/moe_layer.py#L263)。某些类别采用 discard-output checkpoint，另一些采用普通 checkpoint；不能只把同名模块包一层普通 wrapper 就宣称等价实现了所有优化。

当前公开全局 schema 中，Megatron 的模块类别选择不是 MF 的任意路径×任意层号映射；selective 按全局配置应用于适用层，full 的分组／计数另行控制。通信通常跟随模块算法、量化、SP、overlap 约束，不应归纳成一个通用的用户通信 include/exclude。它适合借鉴**按明确边界建类别、说明保存／丢弃哪段张量及兼容性**；MF 适合借鉴**可定位路径、层范围、显式排除及安装计划**。两者的 kernel、硬件、框架不同，本报告没有跨框架性能优胜结论。

## 5. 已有原语与实际缺口：最小建议

以下是接口能力判断和增量建议，不是本轮新代码的规范违规；本轮未修改代码。Fowler 启发式不应用来要求把成熟分布式边界抽象成一个万能模式。

1. **先复用已有功能。** 对全层 off/full/selective，现有 YAML 已够用，保持 `source: default` 即可。做模块成本诊断时，可用公开 [checkpoint / exclude wrapper](<D:/Obsidian Vault D/hyperllal/jt-dev-8npu-20261014/fixes/latest-jt-validation-f637/hyper_parallel/core/activation_memory/api.py:102>)，在单独实验安装层解析精确 FQN，不需要新增 checkpoint 核心 API。安装点必须与 builder 的“参数／激活布局 → activation wrapper → FSDP → materialize/compile”顺序一致，见 [实际构建顺序](<D:/Obsidian Vault D/hyperllal/jt-dev-8npu-20261014/fixes/latest-jt-validation-f637/hyper_parallel/models/_transformers/model_builder.py:497>)；在最终 FSDP 模型上任意换子模块会改变生命周期，不能当作同一实验。
2. **最小 JT 产品接入是 adapter 声明。** 可先声明经验证的 norm／MLP 子区域，并包含 MTP decoder 内相应路径，再用现有 `source: model_adapter_safe_regions` + `layer_count` 或 `layer_indices`。这能提供“固定安全模块集合 × 指定层”的能力，不需新增 selector schema。当前固定源尚未接入，因此下列仅是接入后的配置形状，不能立即用于当前验收：

   ```yaml
   activation_checkpoint:
     mode: full
     swap_inputs: false
     selection:
       source: model_adapter_safe_regions
       layer_indices: [0, 2]
   ```

   这里 `mode: full` 表示每个被选安全子区域使用普通 checkpoint，不是把第 0、2 个完整 decoder 重算。安全区域分支先于 full/selective 分派，内部未加 selective op policy；因此在该分支把 mode 改成 selective **不会**自动给这些子区域附加默认 SAC。见 [安全区域分支先返回](<D:/Obsidian Vault D/hyperllal/jt-dev-8npu-20261014/fixes/latest-jt-validation-f637/hyper_parallel/distributed/activation_checkpoint.py:1114>)、[安全区域 wrapper 安装](<D:/Obsidian Vault D/hyperllal/jt-dev-8npu-20261014/fixes/latest-jt-validation-f637/hyper_parallel/distributed/activation_checkpoint.py:822>)。必须在文档和执行计划中说明，避免同名 mode 隐含不同语义。
3. **真正按任意路径／层范围自由组合时，再补小型配置映射。** 优先复用已有层发现、FQN 匹配和 wrapper，不另建第二套模型遍历。应输出规范化后的安装清单、层号→FQN、实际命中数；对零命中、非法索引、重复／父子嵌套、共享实例和冲突目标明确处理。普通重算 wrapper、exclude wrapper、算子 SAC 是不同动作，不能用一个名称不明的 bool 表示全部。通用 `plan_overrides` 工厂也能作为外部扩展，但不要把 sharding 与重算配置强行耦合来回避清晰接口。
4. **`no_replay_module_patterns` 不是 MF `exclude_op` 的替代。** 当前逻辑只检查选中目标自身 FQN 是否命中 no-replay 并报错，没有安装缓存排除 wrapper，也不会自动排除选中父模块下的某个子孙。见 [安全目标／禁止目标检查](<D:/Obsidian Vault D/hyperllal/jt-dev-8npu-20261014/fixes/latest-jt-validation-f637/hyper_parallel/distributed/activation_checkpoint.py:842>)。例如将父 MLP 包为 checkpoint，并声明其子 gate 不可重放，并不会由该声明自动保护 gate。若新增按路径配置，必须先定义父子覆盖规则和排除边界，而不是把这个字段当通用黑名单。
5. **通信与输出丢弃后置。** 先验证不改变通信边界的 norm／dense MLP，再分开验证 routed MoE、共享专家、MLA、TP／CP／EP 通信。已有 exclude wrapper 只支持 eager、外层 HP 非重入 checkpoint，且不支持嵌套 exclude；`save_output=False` 只允许输出直接传给另一个 excluded region，见 [排除 wrapper 限制](<D:/Obsidian Vault D/hyperllal/jt-dev-8npu-20261014/fixes/latest-jt-validation-f637/hyper_parallel/core/activation_memory/wrapper.py:579>)。不要把它当通用输出释放开关，也不要直接照搬 MF 的 output_recompute 或 Megatron discard-output 功能。
6. **whole-decoder 按层选择也是单独的小缺口。** 当前 selector 附着在 adapter safe 子区域通道，不能不经声明就“只对第 N 个完整 decoder 使用默认 full/SAC”。若需求确实包括该操作，优先把层过滤复用到既有默认 wrapper 路径，并明确索引次序；不能将新增 JT safe provider 宣称为任意整层／任意模块组合已全部支持。

验证这些增量时，除前向数值外，应覆盖 aux-only 与 total 的全部参数梯度、router token-count／bias 更新、MTP 多深度、共享／冻结／未用参数、no-sync→sync、checkpoint 保存续训、空 token／空专家及实际 CP/TP/EP collective 次序。当前三组 FSDP8 精度通过是默认路径的证据，不是尚未安装的新子区域或通信方案的证据。
## 6. 当前 200 步资源结果与选择边界

本节引用已完成的独立资源复算和严格比较；本轮再次核对了报告字段与固定 source/config 身份，没有重新执行训练。精度证据：[full 对 off 严格比较](<D:/Obsidian Vault D/hyperllal/jt-dev-8npu-20261014/evidence/f637-fsdp8-full-200-strict-comparison.json>)、[selective 对 off 严格比较](<D:/Obsidian Vault D/hyperllal/jt-dev-8npu-20261014/evidence/f637-fsdp8-selective-r3-strict-comparison.json>)。两份都是 8 rank × step 1–200、weighted 与还原系数后的 LM/MTP/aux max_abs=0；MTP 系数 0.3，aux 系数 0.0001。这里比较的是采集标量及其系数还原，不声称原始所有张量位级一致。

| 模式 | 逐步 8 rank MAX 后，21–200 步中位时间（s） | 同窗口有效监督 token/s | 全 200 步截至采样累计 allocated 峰值（GiB） | reserved 峰值（GiB） |
|---|---:|---:|---:|---:|
| off | 1.155002654 | 16008.407 | 22.032196045 | 27.250000000 |
| full | 1.235179961 | 14969.751 | 19.392932892 | 24.320312500 |
| selective | 1.232110977 | 15011.921 | 19.875640869 | 24.789062500 |

每组计时窗口只计一次有效监督 token，总数 3,325,552；吞吐分母为逐步 rank MAX 时间之和，不是 180×中位时间，也不对 8 个 rank 的已归约 token 再求和。full 相对 off allocated 节省 2.639263153 GiB（11.9791%），中位步时增加 6.9417%；selective 节省 2.156555176 GiB（9.7882%），中位步时增加 6.6760%。

当前显存足够且目标是本配方吞吐时，off 是本次观测中最快的一组；需要更多显存余量时，full 是本次观测中节省 allocated 最多的一组。full 对 selective 的 3.069 ms 差值只有一次运行，不足以给两者稳定速度排位，更不能证明全局最优。三组都是相同观测工具和固定配置，含输入摘要采集开销；不是无插桩吞吐或性能 CI。

显存值包含初始化、warmup 和截至采样的训练累计峰值，不能叫作“保存激活占用”或总 HBM。off 在末步保存 checkpoint，full/selective 不保存；资源 callback 先采样，末次保存的时间和后续分配不在这组样本内，因此不要比较容器总时长，或宣称测得了保存过程的全生命周期峰值。详细口径见 [资源与测量边界原审计](<D:/Obsidian Vault D/hyperllal/jt-dev-8npu-20261014/evidence/f637-fsdp8-selective-r3-resource-independent-review.json>)。

## 7. 模块成本实验：需要保持或隔离的因素

现有 [历史模块成本报告](<D:/Obsidian Vault D/hyperllal/jt-dev-8npu-20261014/evidence/recompute-cost-analysis.md>)／[历史模块成本 JSON](<D:/Obsidian Vault D/hyperllal/jt-dev-8npu-20261014/evidence/recompute-cost.json>)基于 `ec2e16bce26e497519c573f0a8d87a538a329197`。40 步 A/B 当时显示只重算 MLP 节省约 0.8528 GiB、步时增加约 1.91%；norm 节省约 0.3126 GiB 而时间接近噪声；只包 MTP decoder 当时没有降低整步峰值。这些可用于选择诊断优先顺序，**不能搬进 18602c4 的 200 步验收或与上表相减**。旧 MLP+norm 的 200 步也是另一源及配方，不能替代当前复测。旧模块 events 是 inclusive 调用观察，不是完整的保存激活清单。

建议下一轮先用当前源进行独立小样本定位，再决定是否值得扩展到严格 200 步；下面是设计边界，不是已经启动或已经通过的实验：

| 维度 | 必须保持／独立记录的内容 | 避免的错误结论 |
|---|---|---|
| 模型与数学配方 | 固定源码/patch/image/CANN、初始权重、输入序列与 packed 边界、loss 分母/系数、NS dtype、expert FP32 wgrad、HF32 开关、RoPE/attention kernel、gradient clipping | 将精度、kernel 或优化器变化的影响计入重算收益 |
| 并行与 batch | 固定 FSDP/TP/SP/CP/EP/PP、微批大小、累积次数、有效 token 和 MoE routing 分布；一次只变一个模块策略 | CP/TP 降低 DP 后本地微批数改变，被误判为重算变慢或显存泄漏 |
| 安装区域 | 显式记录 FQN、dense／routed／shared expert 区别、MTP 内外层、真实动态 wrapper、父子和共享对象去重、安装顺序 | 同名 `.mlp` 混合不同计算，或 wrapper 改变 FSDP unit／通信边界 |
| head 与 loss | 固定 enable_chunk_loss、chunk_size=4096、TP loss/SP 输入布局，禁止顺带调整 head 策略 | 把巨大的 logits 峰值或 chunk 反向差异当成 attention/MLP 激活收益 |
| 重放语义 | 明确 full/SAC/exclude、每区域 early_stop、实际 op overload 命中、RNG和副状态、原前向与 replay 计数 | 将提前结束的 replay 当完整前向；把 ignore 当保存；漏掉 aux 梯度或重复 router counts |
| 计时 | 模块 events 单独运行，标记 forward/replay/完成与早停；end-to-end 使用同口径逐步全 rank MAX，warmup一致 | host hook 时长当 kernel 时间、包含异步等待的位置不同、嵌套 inclusive 时间相加 |
| 显存 | 同样采样／reset 边界；报告累计峰值和稳态峰值的区别；如统计 saved tensors，要按实际 storage 去重并排除参数/视图别名 | 进入/退出 allocated 差当保存激活；多模块单独收益相加；allocated 与 reserved 相互替代 |
| 比较置信度 | 同设备空闲条件，配对重复 off/candidate；报告完整分布与运行次数；只在额外时间显著且稳定时给 GiB/额外秒 | 用一次 3 ms 差值排速度冠军；delta-time 接近零却报告无限性价比 |
| 任务操作边界 | 遵守现有协调器、归属证据与 20 分钟冷却；外部占卡立即按既有规则让卡；诊断数据保留可核验摘要后按任务范围清理 | 把外部争用样本用于最终吞吐；删除他人输入或运行中的自有证据 |

优先候选是 norm、dense MLP、MoE/shared experts 分开，以及 MTP 内部 decoder。它们应先验证目标实际生效，再检查整体峰值差和完整梯度；attention／MLA 子投影／通信再单独展开。某模块局部激活少了却没降低整步峰值是可能的：峰值可能出现在 head、参数 unshard、其他层或 optimizer 阶段。要同时给出节省 GiB、额外秒、有效 token/s、峰值所在阶段；性能预算与显存预算不同，不强行用一个比值决定唯一方案。

## 8. 审查结论与证据索引

本轮确认能力边界，未发现需要否定既有默认 FSDP8 off/full/selective 200 步结论的新证据。当前已知缺口是 JT provider 接入、默认 whole-block 的层过滤、用户路径／exclude／通信配置映射及新边界实测；不是现有 checkpoint／exclude 底层能力缺失。上述建议按需实施，均未在本轮落地，也不构成新增 PR/issue 承诺。

已核对的本地源文件 SHA256（实际工作区字节，绑定 `18602c4524dc664b81b2bad9a7b624455ce1494f`）：

| 文件 | SHA256 |
|---|---|
| `hyper_parallel/components/losses/chunked_cross_entropy.py` | `eba4deaa90834afc2fba4a3c4dc550331e43ac51738badf2819c10a18931e41b` |
| `hyper_parallel/components/modules/mtp.py` | `a53caede963cbe4c6ef1711bfdfff7ef0a39a2393720188b11413b56dbead65c` |
| `hyper_parallel/core/activation_memory/api.py` | `7a0120c4c6f7ed423552c11f771a974cad0bfeab74ff23c62834d118adc67b34` |
| `hyper_parallel/core/activation_memory/checkpoint.py` | `ca83f01db99749fcd9176a74dac8bf9508ffc2fc7e0536d06d4a33a8bbbb1872` |
| `hyper_parallel/core/activation_memory/wrapper.py` | `3283c73da08a6342b86287305f5ef931d2dbd44af73b89a92c44a273d9bb89af` |
| `hyper_parallel/distributed/activation_checkpoint.py` | `ccb0cf8023569c4114fa567bd4c2bd3a54fd70b50b1a805efdd0410ca5ea326b` |
| `hyper_parallel/distributed/context_parallel/wrappers.py` | `360cd6c727795572f49a18704c8c1885cda6f21b9d81804fde6f90408f3c56d0` |
| `hyper_parallel/models/_transformers/model_builder.py` | `f5dc8a3bb02141cdc54316d0b0191b9eaa17da6ffaeed706c0147515fc886581` |
| `hyper_parallel/models/adapter_spec.py` | `492b4ae3af89b1a2768a69dfefbbbcfa37000dac9e5d00a454e6a3591fe638b9` |
| `hyper_parallel/models/deepseek_v41/adapter/policies/activation_checkpointing.py` | `334a6b3f3bb42e55f602322e71f8f950a448e9ca065d288e5bbaba306914a69b` |
| `hyper_parallel/models/jt_deepseek_v3/adapter/registration.py` | `fba0632ef7c8501ddcb081ad1a052081cf69a47272cb12a9f5e7594cdc7efba6` |
| `hyper_parallel/models/jt_deepseek_v3/modeling_jt_deepseek_v3.py` | `2ddd28910688710ea777f1724fb8aeb8821c6045a01f18f21eea4e92507fd906` |
| `hyper_parallel/trainer/base.py` | `eb0293f1c87c09895398e848cee62b39a2500484ab2a4f35ce9cfd7085b8a0c7` |
| `hyper_parallel/trainer/config/parallelism.py` | `bc916586d69bb75235dad53a204d93bc0a29a61c2475975645c0b64dd98ad2bc` |

已读取的报告／证据 SHA256：

| 文件 | SHA256 |
|---|---|
| `f637-fsdp8-full-200-strict-comparison.json` | `142513b5c3301bbebc6004c0aa0c1f1b4cc12ac7c54073eb8db407de66094e9b` |
| `f637-fsdp8-selective-r3-resource-independent-review.json` | `602ef366e804b60c6284a32ccf861400fb4ac579a35007176a7d0482b713f80d` |
| `f637-fsdp8-selective-r3-strict-comparison.json` | `118b0eb341ddbd8e6fbbb1db34a54fa4754e2b6d3ec566f06aa47e3ce0db1891` |
| `megatron-recompute-implementation-recheck-20261016.json` | `f8dfaacbc69cc3b5045e139fbfb9e73d60120115a0389e756c093785f7befd85` |
| `recompute-cost-analysis.md` | `56c0bc356cacdbe5a24e650578d032d3e2124180367f42607af5c6492733ad6d` |
| `recompute-cost.json` | `64a457fc601d90c24997d7ccf1444010aafb7bfca88e01fbcb9fad4f9883eaae` |
| `recompute-reference-implementation-check.json` | `ac0fe3b7dd8a49fbcc30096e2a94fa2660d32fe8fb43ec11699b9db86dbb66b8` |

官方固定提交所读关键文件 SHA256（`git show` 的 Git blob 内容）：

| 仓库与文件 | 固定提交 | SHA256 |
|---|---|---|
| `mindformers/mindformers/pynative/distributed/activation_checkpoint.py` | `5089da308ebcc146d0cbe32180c209c4f1d39d77` | `f7a12cd243559355b124f79b562ca7756c0010ae7047c97ad7e9b365b15d655d` |
| `mindformers/mindformers/pynative/config/config.py` | `5089da308ebcc146d0cbe32180c209c4f1d39d77` | `2e7b15b05daca4f05c1a32a38a913aa0811fbbf875ee9e21196604b9b80b9640` |
| `megatron/megatron/core/recompute.py` | `3914483fd325e053797aff3fa431d4a3141139aa` | `edd558802163d851f14a652ecd7d6aa840a1114705f181ca3afbbbeec0fc1502` |
| `megatron/megatron/core/transformer/transformer_config.py` | `3914483fd325e053797aff3fa431d4a3141139aa` | `0c5dffbca1add572204ed36de7ee0cd403b8a57ec22771e4011ab19a4513dca7` |
| `megatron/megatron/core/transformer/transformer_layer.py` | `3914483fd325e053797aff3fa431d4a3141139aa` | `d29e63063188fb1d4e69c0a01a3f5eded14d8b5ce3f05792d0d5d52c0abf7499` |
| `megatron/megatron/core/transformer/moe/moe_layer.py` | `3914483fd325e053797aff3fa431d4a3141139aa` | `b499e61fa5aa74994eb464210f0748cabc81f28efe6262cb3f032abe9122293d` |
