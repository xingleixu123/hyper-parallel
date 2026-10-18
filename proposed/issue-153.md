<!-- correctness-review-20261018 -->
## 2026-10-18 正确性复核：RMSNorm：核对当前实现，不把旧训练图写成 PyNative 现状

**当前判断**：在相同 RMSNorm 定义和 epsilon 下，参数计算副本 dtype、归一化内部累加、乘 gamma 与最终 cast 的顺序都是需要明示的数值策略。差异本身不足以判任一实现算法错误。

**最新源码**：[MF PyNative FusedRMSNorm](https://gitcode.com/mindspore/mindformers/blob/409019aac0c6579bb4fb8a9b04a75c259d98a96e/mindformers/pynative/layers/layer_norm.py#L141) 以 `compute_dtype` 建 weight，并直接调用底层 `rms_norm(x, local_weight, eps)`。它不是旧训练图“Python 先把 x cast 为 FP32、再在末尾 cast 回”的逐字实现；底层 kernel 的内部累加需要算子证据，不能从包装层推断。[Megatron TorchNorm 选择](https://github.com/NVIDIA/Megatron-LM/blob/1effc75a46b26c6b93478355ff9fc3bb8ac1fafc/megatron/core/transformer/torch_norm.py#L58) 可选择原生 Torch RMSNorm，另有 Transformer Engine 路径，并无全框架单一舍入顺序。

**修改方向**：若框架违反自己承诺的 dtype 或导数契约，应修复；如果只是选择不同的合法精度/融合策略，先作为配置和数值差异分析。FP32 gamma 或较晚 cast 可以研究，但“更像 MF”不构成修改默认的充分理由。

**验收**：统一 epsilon、gamma、输入和计算策略，用独立 FP64 前向及解析/数值导数比较，并报告 BF16/FP32 分别误差与成本。不能由“少一次舍入”推出任何输入都更准确。本轮未新增 RMSNorm 产品修改。


本次固定参考：MF GitCode master `409019aac0c6579bb4fb8a9b04a75c259d98a96e`（只核查 `mindformers/pynative/`，未用旧 training_graph 冒充），Megatron main `1effc75a46b26c6b93478355ff9fc3bb8ac1fafc`，当前 JT_dev `a1e6c222eab097c9609d6e744a3c5f98aeed6bdc`。源码路径/行号/SHA-256 见[完整正确性复核](https://github.com/xingleixu123/hyper-parallel/blob/xxl-dev/jt-correctness-review-20261018/AUDIT.md)与[来源清单](https://github.com/xingleixu123/hyper-parallel/blob/xxl-dev/jt-correctness-review-20261018/source-catalog.json)。本轮不新增训练结果，不以 MF loss 距离代替独立正确性判据。


<details><summary>原问题与历史实验记录（适用旧基线；当前判断以上方复核为准）</summary>

两边都是 RMSNorm，eps = 1e-6，位置相同，共 16 个：两层 decoder 各 4 个（input、q_a、kv_a、post），final norm，以及 MTP 的 enorm、hnorm、层内 4 个和 final。不一致的有两处，都是精度：

| 不一致 | 实验编号 | 判定 |
|---|---|---|
| 1. 权重精度 | 5 | 框架差异，没有对错 |
| 2. 计算公式 | 6 | 框架差异，没有对错 |

文中链接：

- HP 指向 [hf1s/hyper-parallel@4fb7739c](https://github.com/hf1s/hyper-parallel/tree/4fb7739cf0216d54c799d29e13e2332bde8b9b7c)，与实验所用代码相同。
- MF 指向 [mindspore-ai/mindformers r2.1.0](https://github.com/mindspore-ai/mindformers/tree/c9a1697718614b4a63aec1379681ab5ca413c2d5)（`c9a16977`），引用的逻辑与实验所用版本相同。
- HF 指向 [huggingface/transformers v5.13.1](https://github.com/huggingface/transformers/tree/v5.13.1)，即实验中 HP 使用的版本。

HP 的 16 个位置都用 HF 的 `DeepseekV32RMSNorm`：[`modeling_jt_deepseek_v3.py:277`](https://github.com/hf1s/hyper-parallel/blob/4fb7739cf0216d54c799d29e13e2332bde8b9b7c/hyper_parallel/models/jt_deepseek_v3/modeling_jt_deepseek_v3.py#L277)、[`:280`](https://github.com/hf1s/hyper-parallel/blob/4fb7739cf0216d54c799d29e13e2332bde8b9b7c/hyper_parallel/models/jt_deepseek_v3/modeling_jt_deepseek_v3.py#L280)（q_a、kv_a），[`:523-524`](https://github.com/hf1s/hyper-parallel/blob/4fb7739cf0216d54c799d29e13e2332bde8b9b7c/hyper_parallel/models/jt_deepseek_v3/modeling_jt_deepseek_v3.py#L523-L524)（input、post），[`:575`](https://github.com/hf1s/hyper-parallel/blob/4fb7739cf0216d54c799d29e13e2332bde8b9b7c/hyper_parallel/models/jt_deepseek_v3/modeling_jt_deepseek_v3.py#L575)（final norm），[`:612-613`](https://github.com/hf1s/hyper-parallel/blob/4fb7739cf0216d54c799d29e13e2332bde8b9b7c/hyper_parallel/models/jt_deepseek_v3/modeling_jt_deepseek_v3.py#L612-L613)（MTP）。MF 在实验配置下用 `FusedRMSNorm`（训练日志 `fused_norm: True`）。

## 1. 权重精度：HP 用 bf16 计算副本，MF 用 fp32 原值

**MF**：权重以 `params_dtype`（fp32）创建（[`norm.py:336`](https://github.com/mindspore-ai/mindformers/blob/c9a1697718614b4a63aec1379681ab5ca413c2d5/mindformers/parallel_core/training_graph/transformer/norm.py#L336)），计算时直接以 fp32 参与（下面的 `norm.py:346-351`）。

**HP**：FSDP 混合精度按单元把参数转成 `param_dtype`（实验配置为 bfloat16），norm 权重也不例外：[`hsdp_param.py:665-673`](https://github.com/hf1s/hyper-parallel/blob/4fb7739cf0216d54c799d29e13e2332bde8b9b7c/hyper_parallel/core/fully_shard/hsdp_param.py#L665-L673)、[`hsdp_param.py:858-859`](https://github.com/hf1s/hyper-parallel/blob/4fb7739cf0216d54c799d29e13e2332bde8b9b7c/hyper_parallel/core/fully_shard/hsdp_param.py#L858-L859)。

**判定**：框架差异，没有对错。与 router 权重是同一个机制：HF 模型用 FSDP 混合精度训练时，所有参数都以 bf16 副本参与计算；MF 把参数的存储精度和计算精度分开设置。

## 2. 计算公式：HP 先把归一化结果转回激活 dtype 再乘权重，MF 全程 fp32、最后只转一次

**MF**：`FusedRMSNorm` 把输入转成 `layernorm_compute_dtype`，与 fp32 权重一起进 `RmsNorm`，结果再转回输入 dtype：[`norm.py:346-351`](https://github.com/mindspore-ai/mindformers/blob/c9a1697718614b4a63aec1379681ab5ca413c2d5/mindformers/parallel_core/training_graph/transformer/norm.py#L346-L351)

```python
original_type = x.dtype
output = self.norm(self.cast(x, self.compute_type), self.weight)[0]
output = self.cast(output, original_type)
```

- `layernorm_compute_dtype` 默认为 fp32：[`configuration_deepseek_v3.py:142`](https://github.com/mindspore-ai/mindformers/blob/c9a1697718614b4a63aec1379681ab5ca413c2d5/mindformers/models/deepseek3/configuration_deepseek_v3.py#L142)
- 不融合的 `RMSNorm` 同样全程用 fp32：[`norm.py:266-276`](https://github.com/mindspore-ai/mindformers/blob/c9a1697718614b4a63aec1379681ab5ca413c2d5/mindformers/parallel_core/training_graph/transformer/norm.py#L266-L276)

**HP**：HF 的 `DeepseekV32RMSNorm.forward`：[`modeling_deepseek_v32.py:57-62`](https://github.com/huggingface/transformers/blob/v5.13.1/src/transformers/models/deepseek_v32/modeling_deepseek_v32.py#L57-L62)

```python
input_dtype = hidden_states.dtype
hidden_states = hidden_states.to(torch.float32)
variance = hidden_states.pow(2).mean(-1, keepdim=True)
hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
return self.weight * hidden_states.to(input_dtype)
```

- variance 和归一化在 fp32 里算，但归一化结果先转回激活 dtype（bf16），再与权重相乘。
- 相乘的结果跟随权重的 dtype：权重是 bf16 副本时，还要再舍入一次。

**判定**：框架差异，没有对错。HP 的写法与 HF 上游一致；MF 只在最后舍入一次，每个 RMSNorm 少一到两次 bf16 舍入，精度更高。

</details>
