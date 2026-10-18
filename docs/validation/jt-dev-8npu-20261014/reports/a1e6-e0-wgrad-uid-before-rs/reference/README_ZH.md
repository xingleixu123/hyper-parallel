# E0 FP32 wgrad：官方源码窄范围核对

本次源码核对支持继续追踪「真实 FP32 梯度对象 → 局部 token 分组 → 通信前缩放 → 归约结果」，没有证据把普通 autograd 零返回、EP4 的 `SUM+1/8` 或跨 EP 的末位差异直接判为产品错误。本文没有执行框架、容器或 NPU，也不验收本轮三步探针或 200 步训练。

2026-10-18 20:41 UTC，官方 `git ls-remote` 实际返回 MF master **409019aac0c6579bb4fb8a9b04a75c259d98a96e**、Megatron main **ce852bfb93f364f64a556602fc71a5d9ed2c377e**。先前 Megatron 62c063 是历史快照；本次更新读取六个相关文件，其中已有的 `tensor_parallel/layers.py` 和 `extensions/transformer_engine.py` 与 62c063 快照逐字节相同，未据此声称整个分支不变。固定 JT 对象仍为 **544526f7db8342c607244fd41009b2a74adb84c8**。

## 1. 梯度对象的身份比变量名更重要

固定 JT 的 `_NPUGroupedMatmul.backward` 在融合分支调用 `_accumulate_fp32_weight_gradient`，随后返回普通权重形状的零张量。实际贡献由 `npu_grouped_matmul_add_` 写入 FP32 `main_grad`：输入和 dY 保持该路径的低精度，目的缓冲区为 FP32；这不能单凭 Python dtype 推断底层乘加的全部内部精度。调用是 dY 转置乘 X，按 `group_list` 分组累加到参数布局，而不是把普通低精度 wgrad 算完再转 FP32。详见固定源 `npu_grouped_swiglu.py:59–91, 96–130`，SHA `b477b1bb…852e`，完整 SHA 在 [sources.json](sources.json)。

`HSDPParamV2._accumulate_fused_grad_if_needed` 的真实分支是：尚无累计对象时，将原 FP32 对象交给 `unsharded_accumulated_grad`；已有累计对象时，原地加到它上面；随后清除参数的 `main_grad` 和 `grad_added_to_main_grad`。因此，清空 `main_grad` 不表示梯度丢失，通信前累计对象非空也不是错误。接下来的普通梯度仍可混合累加。`reduce_scatter_grad` 再选择累计对象，按实际 dtype、分片和 padding 构造通信视图，先缩放后通信。精确位置为固定源 `hsdp_param.py:825–856, 1153–1209`，SHA `b910d5e4…9ec0`。

Megatron 自身的 dense fused linear 也把真实贡献写入 `main_grad` 并设置标志，但 dummy 返回并非一律为零：取决于 `zero_out_wgrad` 和后端，可是零、复用 dummy、空张量或 None。DDP hook 根据标志决定是否再加普通 `.grad`，然后清 `.grad`。这说明不能把「ordinary grad 非真实梯度」误读成缺失梯度，也不能照搬 JT 的零返回断言去检查 Megatron。见 [linear backward，877–928 行](https://github.com/NVIDIA/Megatron-LM/blob/ce852bfb93f364f64a556602fc71a5d9ed2c377e/megatron/core/tensor_parallel/layers.py#L877) 和 [DDP hook，604–629 行](https://github.com/NVIDIA/Megatron-LM/blob/ce852bfb93f364f64a556602fc71a5d9ed2c377e/megatron/core/distributed/distributed_data_parallel.py#L604)。

## 2. 三种前端／后端不能混为一条实现

| 路径 | 本次源码能证明的边界 | 不能据此证明的内容 |
| --- | --- | --- |
| JT544 Ascend 融合专家 wgrad | 匹配的 FP16/BF16 输入和权重；FP32 目的缓冲区；GMMAdd 原地累加；HSDP 接管对象 | 具体 CANN kernel、内部乘加精度、跨不同 K／分组的 bitwise 一致 |
| MF409 当前 PyNative GroupedMLP | MindSpore `nn.Cell` 前端，权重使用 `compute_dtype`，两次 `GroupedMatmul(split_item=3, group_type=0)`；EP 分发由并行包装接管 | 这些 MF 前端代码没有给出 JT 式 FP32 `main_grad` 所有权协议；不能因此断言其实际 wgrad 累加精度 |
| Megatron ce852 分组专家 | `TEGroupedMLP`／`TEGroupedLinear` 委托 Transformer Engine，并传入 `fuse_wgrad_accumulation`；另有 MCore 自己的 dense fused backward | 本次未把 TE 实现版本、TE/CUDA 内核或 Megatron-FSDP 与本次 Ascend GMMAdd 等同 |

MF 的精确入口见 [experts.py，18–24、77–87、177–237 行](https://gitcode.com/mindspore/mindformers/blob/409019aac0c6579bb4fb8a9b04a75c259d98a96e/mindformers/pynative/transformers/moe/experts.py#L177)。这是当前 `mindformers/pynative`，不是历史 `experimental/parallel_core`。Megatron 的对应边界见 [TEGroupedMLP，263–290、830–844 行](https://github.com/NVIDIA/Megatron-LM/blob/ce852bfb93f364f64a556602fc71a5d9ed2c377e/megatron/core/transformer/moe/experts.py#L830) 和 [TEGroupedLinear，2646–2652、2791–2808 行](https://github.com/NVIDIA/Megatron-LM/blob/ce852bfb93f364f64a556602fc71a5d9ed2c377e/megatron/core/extensions/transformer_engine.py#L2791)。

Megatron 还允许局部累计缓冲区与通信缓冲区采用不同精度：必要时通信前复制进去，通信完成后再复制回供优化器读取。于是 `main_grad`、通信输入和优化器输入的 dtype／对象身份需要分别核实；「局部 FP32」不自动等于「通信全过程 FP32」。见 [param_and_grad_buffer.py，708–714、909–922、1481–1501 行](https://github.com/NVIDIA/Megatron-LM/blob/ce852bfb93f364f64a556602fc71a5d9ed2c377e/megatron/core/distributed/param_and_grad_buffer.py#L708)。

## 3. 分组顺序与缩放轴

固定 JT 的 EP 接收缓冲区按发送 rank 排列，`_generate_permute_indices` 将 `(sender, local_expert)` 块转为 `(local_expert, sender)`，保持块内顺序；counts 也在同一顺序下求和。见固定源 `expert_parallel.py:101–168, 570–605`，SHA `f042dc5a…f030`。Megatron AlltoAll dispatcher 同样先交换、再按本地专家重排；MF 无 EP 时用 stable argsort 或融合 permute，有 EP 时直接消费并行包装提供的顺序。见 [Megatron dispatcher，447–457、697–803 行](https://github.com/NVIDIA/Megatron-LM/blob/ce852bfb93f364f64a556602fc71a5d9ed2c377e/megatron/core/transformer/moe/token_dispatcher.py#L768) 和 [MF permute／construct，117–147、177–205 行](https://gitcode.com/mindspore/mindformers/blob/409019aac0c6579bb4fb8a9b04a75c259d98a96e/mindformers/pynative/transformers/moe/experts.py#L117)。

这些位置定义的是 token／expert 对应关系，没有建立不同 EP 拓扑、不同局部 K 长度及归约树的逐位一致合同。稳定排序只约束同一输入顺序下的排序，不能替代跨拓扑的 UID 对齐。离线核对仍须使用真实 DP slot、batch SHA、micro、item/token、top-k、全局 expert，并保留 padding；仅比较无序 global input hash 或同编号 rank 不够。

在本次 TP1/CP1、完整数据域为 8、EP4 的 EDP 为 2 的已知配置下，下面是应核验的数学目标，而非浮点逐位承诺：FSDP 是八份局部贡献的平均；EP 是两组已汇集四个 sender 的贡献，各乘 `1/8` 后求和。必须先证实 token 权重与贡献集合相同，才能讨论两式的数值关系。

Megatron DDP 对常规非 per-token、无 GTP correction 的情形明确给出同一目标：采用 AVG 时，专家梯度先乘 `EDP/DP` 再做 EDP-AVG；采用 SUM 时，先乘 `1/DP` 再在专家数据组求和。实际实现还区分 per-token loss、GTP/EGTP correction、DP×CP 组和通信配置，不能把特例公式无条件推广。见 [DDP 缩放，266–313 行](https://github.com/NVIDIA/Megatron-LM/blob/ce852bfb93f364f64a556602fc71a5d9ed2c377e/megatron/core/distributed/distributed_data_parallel.py#L266) 和 [通信前缩放及 SUM/AVG 选择，722–733 行](https://github.com/NVIDIA/Megatron-LM/blob/ce852bfb93f364f64a556602fc71a5d9ed2c377e/megatron/core/distributed/param_and_grad_buffer.py#L722)。这支持本次 `SUM+1/8` 的缩放轴解释，不能单独证明它的实际执行已正确。

MF 当前默认关闭 FSDP 自带除法，改为 SUM；loss sense 使用 `1/(world/PP)`，并另行处理梯度累积。per-token 分支还在完整梯度上除以实际全局 token 数对应的均值。应连同 loss 构造和 TP/CP 复制关系理解，不能把 MF 的一个系数孤立移植到 JT。见 [MF parallelize，1636–1638 行](https://gitcode.com/mindspore/mindformers/blob/409019aac0c6579bb4fb8a9b04a75c259d98a96e/mindformers/pynative/base_models/gpt/parallelize.py#L1636)、[get_loss_sense，126–166 行](https://gitcode.com/mindspore/mindformers/blob/409019aac0c6579bb4fb8a9b04a75c259d98a96e/mindformers/pynative/distributed/utils.py#L126)、[trainer，2009–2038、2095–2132 行](https://gitcode.com/mindspore/mindformers/blob/409019aac0c6579bb4fb8a9b04a75c259d98a96e/mindformers/pynative/trainer/trainer.py#L2095)。

## 4. 对当前定位的具体约束

下一次解释 E0 原文时，应先串起实际 GMM 输入对应关系、FP32 main carrier、移交后的 accumulator、通信前 flat view／缩放，以及通信后所选坐标。若 UID 或输入编码不同，不能先归咎归约；若输入相同而局部贡献不同，只能先圈定局部分组计算边界；若局部贡献与缩放都相同、通信后才不同，才有依据进一步查归约。离线 FP64 乘积和改序求和可作诊断参照，不能冒充 NPU 的真实加法顺序或 kernel 身份。

本报告仅参考源码，不新增数学干预，不证明整个梯度相同，不确认问题根因，不提出已验证的产品修复，也不把三步证据扩展为 200 步精度／性能验收。

取证范围与完整 SHA、git blob、行锚在 [sources.json](sources.json)。官方 refs 原文保存在 [official-ref-readback-v1.json](official-ref-readback-v1.json)；来源明细见 [source-readback-v1.json](source-readback-v1.json) 和 [additional-source-readback-v1.json](additional-source-readback-v1.json)。MF 优先遵循用户指定 GitCode 官方入口；复用现有对象和既存官方快照，并对当前 commit tree 的 blob SHA 校验。禁用了 Git lazy fetch，未 clone/fetch/checkout；Megatron 仅通过官方 GitHub contents API 读取六个固定 commit 文件并验证 git blob SHA。首次本地 MF trainer blob 缺失时安全停止，随后复用已保存且 blob 相符的官方原文，没有隐式补取。所有新文件仅在本报告目录内产生。
