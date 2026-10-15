# 重计算配置能力与测量口径

核对日期：2026-10-15（北京时间）。以下是配置能力核对，性能判断需使用本任务 Ascend 实测。

| 能力 | 当前 JT_dev + 本任务修复 | MindFormers PyNative 固定源码 | Megatron Core 固定版本 |
|---|---|---|---|
| 完全重计算 | HP 非重入 wrapper，trunk 与独立 MTP decoder 全覆盖 | 按层列表/范围选择 full | full，uniform 分组或 block 层数 |
| 选择重计算 | 通用算子策略：保存部分昂贵算子，matmul 交替保存/重算 | 模块路径正则映射到层范围 | 预定义模块类别，如 core_attn、MLP、MoE、MLA up projection、norm |
| 任意模块和层组合 | 当前 YAML 不支持；adapter safe regions 需模型声明，JT 尚无该声明 | select_module 支持路径与层范围映射 | 通常按模块类别覆盖所有层，不等同于任意模块路径 |
| 排除重放 | 默认策略中的算子集合；无用户级逐层 exclude_op 映射 | exclude_op 可缓存模块/算子/通信输出 | 模块特定实现，部分支持丢弃输出的 checkpoint |
| 通信重计算 | 必须按具体 TP/EP/CP wrapper 验证 | 独立 recompute_comm 配置 | 与模块及通信 overlap 的兼容性相关 |

JT 的 `selective` 是算子选择策略，不能据名称认定为 MindFormers 的模块选择模式。若要支持类似灵活度，需要配置校验、模块匹配、嵌套区域冲突检测，以及副状态/collective 重放契约；本轮先以公开 checkpoint wrapper 做独立实验，不把实验插桩当成产品能力。

## 成本测量

- 200 步无模块插桩的 runs 用于精度和整体性能验收：排除前 20 步，报告中位步耗时、吞吐、最大 allocated/reserved。
- 单独的短诊断 runs 在少量稳定步骤记录模块 NPU events。耗时是模块调用的 inclusive 时间；重计算早停的未完成调用独立标注。
- 模块进入/退出 allocated 差额包含输出和临时分配，不能称为“保存激活大小”，也不能把嵌套模块时间相加。
- 真正的重计算收益通过同一配置的单模块重算 A/B 测量整体峰值差与步耗时差；收益比使用节省 GiB / 额外秒，并检查 loss 未偏离。
- 内存已经足够时，判断是否值得重算还需看吞吐变化。显存降低本身不能证明性能收益。

## 核对依据

- [MindFormers master 配置说明](https://www.mindspore.cn/mindformers/docs/en/master/feature/configuration.html)，2026-10-14 检索；文档为可变 master；实现已另行固定为下列源码提交。
- [MindFormers master 内存优化](https://www.mindspore.cn/mindformers/docs/en/master/feature/memory_optimization.html)，包含 select_module、exclude_op 和 recompute_comm 的接口。
- [Megatron TransformerConfig](https://github.com/NVIDIA/Megatron-LM/blob/0101df91e5f86acb26a03bd8ad2361f3113476a6/megatron/core/transformer/transformer_config.py#L630)，核对时 main 的提交为 `0101df91e5f86acb26a03bd8ad2361f3113476a6`（2026-10-15 北京时间）。
- [Megatron Bridge 重计算说明](https://docs.nvidia.com/nemo/megatron-bridge/latest/training/activation-recomputation.html)，用于交叉核对模块类别及配置含义。
- 本任务核对代码为最新上游 `f637ef7` 上的集成源 `18602c4524dc664b81b2bad9a7b624455ce1494f`：`hyper_parallel/distributed/activation_checkpoint.py`、`hyper_parallel/trainer/config/parallelism.py`、JT adapter registry。

## 最新 PyNative 实现细节

2026-10-15 16:50（北京时间），用户指定的 MindFormers 官方 GitCode 入口与官方 GitHub 镜像 master 均核对到 `5089da308ebcc146d0cbe32180c209c4f1d39d77`。[`apply_recompute`](https://github.com/mindspore-ai/mindformers/blob/5089da308ebcc146d0cbe32180c209c4f1d39d77/mindformers/pynative/distributed/activation_checkpoint.py#L791) 先安装 exclude wrapper，再应用 full、select、comm；按父模块覆盖关系去重。它会把 MTP 作为 trunk 后面的层号，并将完整层的 early_stop 设为false，以使通信及自定义反向边界重放完整。选择子模块仍可早停。这些实现契约比只支持一个模块名列表更完整，移植时需要逐个拓扑验证。

同一源码的 [`Muon`](https://github.com/mindspore-ai/mindformers/blob/5089da308ebcc146d0cbe32180c209c4f1d39d77/mindformers/pynative/optimizer/muon.py#L1195) 在FP32中做范数归一化，再进入BF16 NS；Addmm/Baddbmm通过cube_math_type=4保留FP32加法。JT当前使用独立matmul与BF16乘加，数值路径不同。本任务已核对安装的 torch_npu 存在该精度接口；这里是实现语义比较，不能直接把 MF 的实现收益当作 JT 的精度或性能实测。

## 最新提交的差异核验

MF 的重计算实现与 Muon 实现文件在上述新提交和前次 `302c939f` 之间 blob 完全相同。Megatron 的 `transformer_block.py`、`transformer_layer.py` 也未变化；配置文件本次差异位于 DSA indexer / MLA 量化能力，重计算字段与校验段保持原语义。因此上表的能力判断仍成立。新引用固定到当前已核对提交，后续上游变化不自动成为本任务证据。机器记录见 [引用核对](recompute-reference-head-check.json) 和 [实现文件比较](recompute-reference-implementation-check.json)。
