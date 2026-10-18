<!-- correctness-review-20261018 -->
## 2026-10-18 正确性复核：Embedding：参数副本精度与梯度累加策略分开验收

**当前判断**：历史实验的前向一致不保证反向同精度。重复 token 的梯度是各位置上游梯度之和；若先写 BF16 表梯度、再转 FP32 归约，较早舍入不能被 FP32 归约恢复。该事实不等于默认 BF16 计算副本是代码错误。

**最新源码**：[MF embedding 精度字段](https://gitcode.com/mindspore/mindformers/blob/409019aac0c6579bb4fb8a9b04a75c259d98a96e/mindformers/pynative/base_models/common/embeddings/language_model_embedding.py#L62) 将 `embedding_params_dtype` 与 `compute_dtype` 分开配置。[Megatron FP32 residual 输出选项](https://github.com/NVIDIA/Megatron-LM/blob/1effc75a46b26c6b93478355ff9fc3bb8ac1fafc/megatron/core/models/common/embeddings/language_model_embedding.py#L138) 的 `fp32_residual_connection` 是输出转换，不能拿这一字段证明其 embedding 参数或梯度累加必为 FP32。需继续跟实际参数/梯度缓冲与所选后端。

**修改方向**：若选择 FP32 embedding 参数/梯度策略，应完整实现保存、恢复、通信 dtype 与累加，不以末尾 `.float()` 假装更高精度。是否作为产品选项需有准确性和内存/性能证据，不能仅为贴近归档修改全局默认。

**验收**：重复 token、正负抵消、非零 padding、FP32/BF16 上游梯度与 checkpoint 恢复的独立累加参考；明确输出、表参数和梯度 dtype。本轮仅核对源码/契约，没有重新执行最新 MF 的反向训练，不冒记为新算子实测。


本次固定参考：MF GitCode master `409019aac0c6579bb4fb8a9b04a75c259d98a96e`（只核查 `mindformers/pynative/`，未用旧 training_graph 冒充），Megatron main `1effc75a46b26c6b93478355ff9fc3bb8ac1fafc`，当前 JT_dev `a1e6c222eab097c9609d6e744a3c5f98aeed6bdc`。源码路径/行号/SHA-256 见[完整正确性复核](https://github.com/xingleixu123/hyper-parallel/blob/xxl-dev/jt-correctness-review-20261018/AUDIT.md)与[来源清单](https://github.com/xingleixu123/hyper-parallel/blob/xxl-dev/jt-correctness-review-20261018/source-catalog.json)。本轮不新增训练结果，不以 MF loss 距离代替独立正确性判据。


<details><summary>原问题与历史实验记录（适用旧基线；当前判断以上方复核为准）</summary>

两边 embedding 的前向输出逐元素一致：在 fp32 表上 gather 后转成 bf16，与在舍入成 bf16 的表上 gather，结果相同（第 301 步逐层 dump 中 embedding 的 relative L2 为 0）。不一致的只有一处：

| 不一致 | 实验编号 | 判定 |
|---|---|---|
| 反向梯度的精度 | 8 | 框架差异，没有对错 |

文中链接：

- HP 指向 [hf1s/hyper-parallel@4fb7739c](https://github.com/hf1s/hyper-parallel/tree/4fb7739cf0216d54c799d29e13e2332bde8b9b7c)，与实验所用代码相同。
- MF 指向 [mindspore-ai/mindformers r2.1.0](https://github.com/mindspore-ai/mindformers/tree/c9a1697718614b4a63aec1379681ab5ca413c2d5)（`c9a16977`），引用的逻辑与实验所用版本相同。

## 1. 梯度精度：MF 在 fp32 里累加，HP 的表梯度先存成 bf16

反向时，两边传到 embedding 的上游梯度（每个位置一行，形状 [序列长度, hidden]）都是 bf16。embedding 的反向把同一个 token 所有出现位置的梯度行相加，得到表中这一行的梯度。两边的差别只在这个和用什么精度存。

**MF**：

- embedding 表以 fp32 存储（训练日志中 `embedding.word_embeddings.weight` 为 Float32）。
- 前向先在 fp32 表上 gather（[`language_model_embedding.py:132`](https://github.com/mindspore-ai/mindformers/blob/c9a1697718614b4a63aec1379681ab5ca413c2d5/mindformers/parallel_core/training_graph/base_models/common/embeddings/language_model_embedding.py#L132)），再转成计算 dtype（[`language_model_embedding.py:158`](https://github.com/mindspore-ai/mindformers/blob/c9a1697718614b4a63aec1379681ab5ca413c2d5/mindformers/parallel_core/training_graph/base_models/common/embeddings/language_model_embedding.py#L158)）。
- 反向时，上游梯度经这次 cast 转成 fp32，再按 token 累加到 fp32 的表梯度上，中间没有 bf16 舍入（按算子语义推断）。

**HP**：

- 前向用的是 FSDP 生成的 bf16 计算副本，gather 在 bf16 表上做：[`hsdp_param.py:665-673`](https://github.com/hf1s/hyper-parallel/blob/4fb7739cf0216d54c799d29e13e2332bde8b9b7c/hyper_parallel/core/fully_shard/hsdp_param.py#L665-L673)、[`hsdp_param.py:858-859`](https://github.com/hf1s/hyper-parallel/blob/4fb7739cf0216d54c799d29e13e2332bde8b9b7c/hyper_parallel/core/fully_shard/hsdp_param.py#L858-L859)。
- 参数的梯度与参数同 dtype，所以 autograd 给出的表梯度是 bf16：每个 token 的梯度和以 bf16 存下来。
- FSDP 归约时取的就是这份计算副本的梯度（[`param_group.py:614-618`](https://github.com/hf1s/hyper-parallel/blob/4fb7739cf0216d54c799d29e13e2332bde8b9b7c/hyper_parallel/core/fully_shard/param_group.py#L614-L618)），再拷进 `reduce_dtype`（实验配置为 float32）的缓冲区归约（[`param_group.py:625`](https://github.com/hf1s/hyper-parallel/blob/4fb7739cf0216d54c799d29e13e2332bde8b9b7c/hyper_parallel/core/fully_shard/param_group.py#L625)、[`hsdp_param.py:407-408`](https://github.com/hf1s/hyper-parallel/blob/4fb7739cf0216d54c799d29e13e2332bde8b9b7c/hyper_parallel/core/fully_shard/hsdp_param.py#L407-L408)、[`param_group.py:728-737`](https://github.com/hf1s/hyper-parallel/blob/4fb7739cf0216d54c799d29e13e2332bde8b9b7c/hyper_parallel/core/fully_shard/param_group.py#L728-L737)）。bf16 转 fp32 本身不损失精度，但前面那次 bf16 舍入已经发生，无法恢复。

**差别有多大**：同一个梯度，HP 多一次 bf16 舍入。

- 如果 NPU 的 embedding 反向内部用 fp32 累加、最后舍入一次，逐元素的相对误差最多约 2⁻⁹（0.2%）。
- 如果在 bf16 里逐次累加，窗口内出现次数多的 token 每加一次都要舍入，误差会更大。NPU kernel 内部用哪种方式没有核实。

**判定**：框架差异，没有对错。与 RMSNorm、router 权重是同一个机制：HP 的参数以 bf16 副本参与计算，MF 把存储精度和计算精度分开设置。前向不受影响，只有 embedding 梯度多一次舍入。


</details>
