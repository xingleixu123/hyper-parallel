<!-- correctness-review-20261018 -->
## 2026-10-18 正确性复核：RoPE：历史 BF16 差异已不能代表当前 JT

**当前状态**：[当前 JT FP32 rotary 契约](https://github.com/hf1s/hyper-parallel/blob/a1e6c222eab097c9609d6e744a3c5f98aeed6bdc/hyper_parallel/models/jt_deepseek_v3/modeling_jt_deepseek_v3.py#L106) 已明确先将 channels/tables 转 FP32，完成旋转后转回原激活 dtype。旧正文中 HP 在 BF16 中完成旋转的代码是旧提交，不能继续当作当前 JT 缺陷。

**最新源码**：[MF PyNative RoPE](https://gitcode.com/mindspore/mindformers/blob/409019aac0c6579bb4fb8a9b04a75c259d98a96e/mindformers/pynative/base_models/common/embeddings/rope_utils.py#L148) 依据 `rotary_dtype` 和融合配置选择路径；[Megatron 未融合 RoPE](https://github.com/NVIDIA/Megatron-LM/blob/1effc75a46b26c6b93478355ff9fc3bb8ac1fafc/megatron/core/models/common/embeddings/rope_utils.py#L140) 的未融合路径会把 cos/sin 转到激活 dtype。行业存在多种合法精度策略，不能只为 MF 逐位结果强制采用某一路线。

**位置编号判断的条件**：在文档 attention 完全隔离、每个文档只差一个共同位置偏移、频率不随位置/有效长度改变的标准 RoPE 中，相对旋转保持不变；有限精度下仍可不同。不能把此性质推广到任意位置缩放、跨文档 attention、非线性位置规则或 cache 偏移。

**动作与验收**：保留当前 FP32 rotary 契约，位置重置作为明确配方。核对实际使用的位置，而不只比较数据字段；以解析二维旋转、逆变换、通道布局和 backward 验收。旧 MF `use_eod_reset=False` 的绝对位置只属于归档复现，不足以要求改变 HP 默认。


本次固定参考：MF GitCode master `409019aac0c6579bb4fb8a9b04a75c259d98a96e`（只核查 `mindformers/pynative/`，未用旧 training_graph 冒充），Megatron main `1effc75a46b26c6b93478355ff9fc3bb8ac1fafc`，当前 JT_dev `a1e6c222eab097c9609d6e744a3c5f98aeed6bdc`。源码路径/行号/SHA-256 见[完整正确性复核](https://github.com/xingleixu123/hyper-parallel/blob/xxl-dev/jt-correctness-review-20261018/AUDIT.md)与[来源清单](https://github.com/xingleixu123/hyper-parallel/blob/xxl-dev/jt-correctness-review-20261018/source-catalog.json)。本轮不新增训练结果，不以 MF loss 距离代替独立正确性判据。


<details><summary>原问题与历史实验记录（适用旧基线；当前判断以上方复核为准）</summary>

两边的旋转频率相同：rope_theta = 5e6，作用在 q、k 的 64 个 rope 维度上；HP 用 NumPy 的 fp32 计算逆频率，与 MF 逐位一致（[`modeling_jt_deepseek_v3.py:738-741`](https://github.com/hf1s/hyper-parallel/blob/4fb7739cf0216d54c799d29e13e2332bde8b9b7c/hyper_parallel/models/jt_deepseek_v3/modeling_jt_deepseek_v3.py#L738-L741)）。不一致的有两处：

| 不一致 | 实验编号 | 判定 |
|---|---|---|
| 1. 位置编号 | 4 | 配置差异，数学等价，没有对错 |
| 2. 旋转的计算精度 | 7 | 框架差异，没有语义上的对错 |

文中链接：

- HP 指向 [hf1s/hyper-parallel@4fb7739c](https://github.com/hf1s/hyper-parallel/tree/4fb7739cf0216d54c799d29e13e2332bde8b9b7c)，与实验所用代码相同。
- MF 指向 [mindspore-ai/mindformers r2.1.0](https://github.com/mindspore-ai/mindformers/tree/c9a1697718614b4a63aec1379681ab5ca413c2d5)（`c9a16977`），引用的逻辑与实验所用版本相同。
- HF 指向 [huggingface/transformers v5.13.1](https://github.com/huggingface/transformers/tree/v5.13.1)，即实验中 HP 使用的版本。

## 1. 位置编号：MF 整个窗口连续编号，HP 每篇文档从 0 开始

**MF**：`use_eod_reset: False` 时丢弃 `position_ids`，整个窗口用 0..S−1：[`gpt_model.py:448-449`](https://github.com/mindspore-ai/mindformers/blob/c9a1697718614b4a63aec1379681ab5ca413c2d5/mindformers/parallel_core/training_graph/base_models/gpt/gpt_model.py#L448-L449)

```python
if not self.config.use_eod_reset:
    position_ids = None
```

**HP**：

- `reset_position_ids: true`（JT 默认配置）时，按 `cu_seq_lens` 在每篇文档开头把位置重置为 0：[`get_batch.py:409-424`](https://github.com/hf1s/hyper-parallel/blob/4fb7739cf0216d54c799d29e13e2332bde8b9b7c/hyper_parallel/data/batching/get_batch.py#L409-L424)
- 没有传入 `position_ids` 时，JT 模型同样按文档编号：[`modeling_jt_deepseek_v3.py:729-737`](https://github.com/hf1s/hyper-parallel/blob/4fb7739cf0216d54c799d29e13e2332bde8b9b7c/hyper_parallel/models/jt_deepseek_v3/modeling_jt_deepseek_v3.py#L729-L737)

**判定**：配置差异，数学等价，没有对错。

- 两边的 attention 都按文档隔离，同一篇文档内的相对位置相同，而 RoPE 的注意力分数只取决于相对位置。
- 差别只在 cos / sin 的取值，以及由此带来的舍入差异。

## 2. 旋转精度：MF 在 fp32 里旋转，HP 在激活 dtype（bf16）里旋转

**MF**：

- `rotary_dtype` 默认为 fp32：[`configuration_deepseek_v3.py:143`](https://github.com/mindspore-ai/mindformers/blob/c9a1697718614b4a63aec1379681ab5ca413c2d5/mindformers/models/deepseek3/configuration_deepseek_v3.py#L143)
- 输入、cos、sin 都转成 fp32，旋转后只转回一次：[`rope_utils.py:138-150`](https://github.com/mindspore-ai/mindformers/blob/c9a1697718614b4a63aec1379681ab5ca413c2d5/mindformers/parallel_core/training_graph/base_models/common/embeddings/rope_utils.py#L138-L150)

**HP**：

- cos / sin 在 fp32 里算出后转成激活 dtype：[`modeling_jt_deepseek_v3.py:746-750`](https://github.com/hf1s/hyper-parallel/blob/4fb7739cf0216d54c799d29e13e2332bde8b9b7c/hyper_parallel/models/jt_deepseek_v3/modeling_jt_deepseek_v3.py#L746-L750)
- 旋转在激活 dtype 里做：[`modeling_jt_deepseek_v3.py:101-104`](https://github.com/hf1s/hyper-parallel/blob/4fb7739cf0216d54c799d29e13e2332bde8b9b7c/hyper_parallel/models/jt_deepseek_v3/modeling_jt_deepseek_v3.py#L101-L104)

  ```python
  ordered = torch.cat((values[..., ::2], values[..., 1::2]), dim=-1)
  first, second = ordered.chunk(2, dim=-1)
  rotated = torch.cat((-second, first), dim=-1)
  return ordered * cos.unsqueeze(1) + rotated * sin.unsqueeze(1)
  ```

**判定**：框架差异，没有语义上的对错。

- HP 的精度约定与 HF 上游相同：cos / sin 以激活 dtype 返回（[`modeling_deepseek_v32.py:130`](https://github.com/huggingface/transformers/blob/v5.13.1/src/transformers/models/deepseek_v32/modeling_deepseek_v32.py#L130)），旋转在激活 dtype 里做（[`modeling_deepseek_v32.py:302-337`](https://github.com/huggingface/transformers/blob/v5.13.1/src/transformers/models/deepseek_v32/modeling_deepseek_v32.py#L302-L337)）。
- MF 在 fp32 里完成乘加，少了 cos / sin 和中间乘积的 bf16 舍入，精度更高。


</details>
