<!-- correctness-review-20261018 -->
## 2026-10-18 正确性复核：packed MTP：独立文档语义优先于归档轨迹

**当前判断**：若 packed 数据表示互相独立的文档，MTP 输入、label 和 loss mask 的移位都必须尊重同一文档边界。让文档 A 的末位置监督文档 B 的目标会改变训练问题；不能为贴近旧 MF 将这种目标恢复为 HP 默认。

**最新源码**：[MF PyNative MTP 移位调用](https://gitcode.com/mindspore/mindformers/blob/409019aac0c6579bb4fb8a9b04a75c259d98a96e/mindformers/pynative/transformers/multi_token_prediction.py#L651) 的 660–661 行仍对 label/mask 调用普通 `roll_tensor`，embedding 移位也没有文档边界参数。[Megatron packed MTP](https://github.com/NVIDIA/Megatron-LM/blob/1effc75a46b26c6b93478355ff9fc3bb8ac1fafc/megatron/core/transformer/multi_token_prediction.py#L219) 在提供 `packed_seq_params` 时于 273–275 行转入文档内移位。必须核对调用者是否传入 packed 元数据。

**触发条件**：仅当样本确为独立 packed 文档，且上层没有事先将相关边界监督全部置零时，这一代码差异才会形成跨文档目标。未运行最新 MF 完整数据流水线，不能声称它的所有配置都出错；普通单文档序列的全局 shift 本身合理。

**修改方向与验收**：HP 保留文档内 shift；未来 MF 修复需让 input/label/mask 三条路径使用同一边界，或由明确数据协议排除边界监督。验收应包括两文档 toy、有效 mask 边界、输入与梯度，不能只看 scalar loss。#245 的跨文档目标仅是旧归档复现，已降为 Draft，不作为产品正确性修复。


本次固定参考：MF GitCode master `409019aac0c6579bb4fb8a9b04a75c259d98a96e`（只核查 `mindformers/pynative/`，未用旧 training_graph 冒充），Megatron main `1effc75a46b26c6b93478355ff9fc3bb8ac1fafc`，当前 JT_dev `a1e6c222eab097c9609d6e744a3c5f98aeed6bdc`。源码路径/行号/SHA-256 见[完整正确性复核](https://github.com/xingleixu123/hyper-parallel/blob/xxl-dev/jt-correctness-review-20261018/AUDIT.md)与[来源清单](https://github.com/xingleixu123/hyper-parallel/blob/xxl-dev/jt-correctness-review-20261018/source-catalog.json)。本轮不新增训练结果，不以 MF loss 距离代替独立正确性判据。


<details><summary>原问题与历史实验记录（适用旧基线；当前判断以上方复核为准）</summary>

# MTP：MindFormers 与 Hyper-Parallel 的不一致

两边 MTP 的结构相同：1 层；enorm、hnorm 拼接后经 `eh_proj` 进入 transformer 层；loss 系数 0.3。不一致的只有一处：

| 不一致 | 实验编号 | 判定 |
|---|---|---|
| 文档边界上的移位 | 3 | MF 写法问题（bug），有对错；HP 正确 |

文中链接：

- HP 指向 [hf1s/hyper-parallel@4fb7739c](https://github.com/hf1s/hyper-parallel/tree/4fb7739cf0216d54c799d29e13e2332bde8b9b7c)，与实验所用代码相同。
- MF 指向 [mindspore-ai/mindformers r2.1.0](https://github.com/mindspore-ai/mindformers/tree/c9a1697718614b4a63aec1379681ab5ca413c2d5)（`c9a16977`），引用的逻辑与实验所用版本相同。
- Megatron-LM 指向 [e95ea35a](https://github.com/NVIDIA/Megatron-LM/blob/e95ea35ae5686e7ece6da212ca544cf353b23cf0/megatron/core/transformer/multi_token_prediction.py)，作为判定依据。

## 1. 文档边界：MF 整行移位，HP 在边界截断

设文档 A 在位置 e 结束，下一篇文档 B 从位置 e 开始，A 的最后一个位置是 e−1。

| | MF | HP |
|---|---|---|
| 实现 | [`multi_token_prediction.py:425-432`](https://github.com/mindspore-ai/mindformers/blob/c9a1697718614b4a63aec1379681ab5ca413c2d5/mindformers/parallel_core/training_graph/transformer/multi_token_prediction.py#L425-L432) `roll_tensor`：对整行 slice 后在末位补 0 | [`modeling_jt_deepseek_v3.py:729-734`](https://github.com/hf1s/hyper-parallel/blob/4fb7739cf0216d54c799d29e13e2332bde8b9b7c/hyper_parallel/models/jt_deepseek_v3/modeling_jt_deepseek_v3.py#L729-L734) 根据 `actual_seq_len` 生成 `sequence_end_mask`；[`components/modules/mtp.py:135-137`](https://github.com/hf1s/hyper-parallel/blob/4fb7739cf0216d54c799d29e13e2332bde8b9b7c/hyper_parallel/components/modules/mtp.py#L135-L137) 和 [`components/losses/mtp.py:49-54`](https://github.com/hf1s/hyper-parallel/blob/4fb7739cf0216d54c799d29e13e2332bde8b9b7c/hyper_parallel/components/losses/mtp.py#L49-L54) 用它截断 |
| 作用对象 | input_ids（[`:477`](https://github.com/mindspore-ai/mindformers/blob/c9a1697718614b4a63aec1379681ab5ca413c2d5/mindformers/parallel_core/training_graph/transformer/multi_token_prediction.py#L477)）、labels（[`:504`](https://github.com/mindspore-ai/mindformers/blob/c9a1697718614b4a63aec1379681ab5ca413c2d5/mindformers/parallel_core/training_graph/transformer/multi_token_prediction.py#L504)）、loss_mask（[`:505`](https://github.com/mindspore-ai/mindformers/blob/c9a1697718614b4a63aec1379681ab5ca413c2d5/mindformers/parallel_core/training_graph/transformer/multi_token_prediction.py#L505)） | MTP 输入 token、MTP target |
| 位置 e−1 的输入 | 文档 B 的首 token x[e] | id 0 |
| 位置 e−1 的 target | 文档 B 的首位 label y[e] | 忽略 |

**图解**：一行里打包了文档 A（位置 0–3）和文档 B（位置 4–7），e = 4。x 是主干的输入 token，y 是主干在每个位置的 LM 目标（已左移一位的 label，不监督的位置为 −100）。MTP 只有 1 层：在位置 t 用主干隐状态 $h_t$ 和 $x_{t+1}$ 的 embedding 预测 $y_{t+1}$。

| 位置 t | 0 | 1 | 2 | **3 = e−1** | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| 文档 | A | A | A | A | B | B | B | B |
| 主干输入 | x₀ | x₁ | x₂ | x₃ | x₄ | x₅ | x₆ | x₇ |
| LM 目标 | y₀ | y₁ | y₂ | y₃ | y₄ | y₅ | y₆ | y₇ |
| MTP 输入（HP） | x₁ | x₂ | x₃ | **0** | x₅ | x₆ | x₇ | 0 |
| MTP 输入（MF） | x₁ | x₂ | x₃ | **x₄** | x₅ | x₆ | x₇ | 0 |
| MTP 目标（HP） | y₁ | y₂ | y₃ | **忽略** | y₅ | y₆ | y₇ | 忽略 |
| MTP 目标（MF） | y₁ | y₂ | y₃ | **y₄** | y₅ | y₆ | y₇ | 忽略 |

两边只在文档 A 的最后一个位置 e−1 不同；整行的最后一个位置两边相同。

**公式**：记 S 为行长，doc(t) 为位置 t 所属的文档。HP 在文档边界截断，MF 整行左移：

```math
\tilde x_t^{\mathrm{HP}} = \begin{cases} x_{t+1}, & t+1 \lt S,\ \mathrm{doc}(t+1)=\mathrm{doc}(t)\\ 0, & \text{otherwise} \end{cases}
\qquad
\tilde y_t^{\mathrm{HP}} = \begin{cases} y_{t+1}, & t+1 \lt S,\ \mathrm{doc}(t+1)=\mathrm{doc}(t)\\ -100, & \text{otherwise} \end{cases}
```

```math
\tilde x_t^{\mathrm{MF}} = \begin{cases} x_{t+1}, & t+1 \lt S\\ 0, & t=S-1 \end{cases}
\qquad
\tilde y_t^{\mathrm{MF}} = \begin{cases} y_{t+1}, & t+1 \lt S\\ -100, & t=S-1 \end{cases}
```

两边的 MTP loss 是同一个公式，只对目标不为 −100 的位置取平均：

```math
\mathcal{L}_{\mathrm{MTP}} = \frac{1}{N}\sum_{t:\ \tilde y_t \neq -100} \mathrm{CE}\big(f(h_t,\ E[\tilde x_t]),\ \tilde y_t\big),
\qquad N = \bigl|\{t:\ \tilde y_t \neq -100\}\bigr|,
\qquad \mathcal{L} = \mathcal{L}_{\mathrm{LM}} + 0.3\,\mathcal{L}_{\mathrm{MTP}} + \mathcal{L}_{\mathrm{aux}}
```

其中 $h_t$ 是主干最后一层的输出（final norm 之前）；共享 embedding 记为 $E$；MTP 层记为 $f$（hnorm 与 enorm 拼接 → eh_proj → transformer 层 → final norm → 共享输出头）。代入 t = e−1：HP 的目标是 −100，不进 loss；MF 的输入是 $x_e$、目标是 $y_e$。两边的 attention 都按文档隔离，e−1 又是文档 A 的最后一个位置，所以其余位置的 MTP 输出不受这个输入影响。

**判定**：MF 写法问题（bug），有对错；HP 正确。

- packed 数据配合按文档隔离的 attention 时，MTP 的左移应在文档边界截断。Megatron-LM 上游就是这样做的：
  - `roll_tensor` 的 docstring（[L238-239](https://github.com/NVIDIA/Megatron-LM/blob/e95ea35ae5686e7ece6da212ca544cf353b23cf0/megatron/core/transformer/multi_token_prediction.py#L238-L239)）："For packed sequences: Respects sequence boundaries when rolling to avoid mixing tokens from different sequences."
  - `_roll_tensor_packed_seq`（[L360-410](https://github.com/NVIDIA/Megatron-LM/blob/e95ea35ae5686e7ece6da212ca544cf353b23cf0/megatron/core/transformer/multi_token_prediction.py#L360-L410)）对每篇文档单独移位；[L407-408](https://github.com/NVIDIA/Megatron-LM/blob/e95ea35ae5686e7ece6da212ca544cf353b23cf0/megatron/core/transformer/multi_token_prediction.py#L407-L408) 把越过文档边界的位置清零。
  - labels、loss_mask（[L1220-1257](https://github.com/NVIDIA/Megatron-LM/blob/e95ea35ae5686e7ece6da212ca544cf353b23cf0/megatron/core/transformer/multi_token_prediction.py#L1220-L1257)）和 input_ids、position_ids（[L1618-1647](https://github.com/NVIDIA/Megatron-LM/blob/e95ea35ae5686e7ece6da212ca544cf353b23cf0/megatron/core/transformer/multi_token_prediction.py#L1618-L1647)）的调用都传入了 `packed_seq_params`。
- MF 自身前后不一致：attention 按文档隔离（`actual_seq_len` 经 [`gpt_model.py:456-459`](https://github.com/mindspore-ai/mindformers/blob/c9a1697718614b4a63aec1379681ab5ca413c2d5/mindformers/parallel_core/training_graph/base_models/gpt/gpt_model.py#L456-L459) 传给每一层；MTP 层见 [`multi_token_prediction.py:215`](https://github.com/mindspore-ai/mindformers/blob/c9a1697718614b4a63aec1379681ab5ca413c2d5/mindformers/parallel_core/training_graph/transformer/multi_token_prediction.py#L215)、[`:250`](https://github.com/mindspore-ai/mindformers/blob/c9a1697718614b4a63aec1379681ab5ca413c2d5/mindformers/parallel_core/training_graph/transformer/multi_token_prediction.py#L250)），只有 MTP 的移位跨越了边界。
- 位置 e−1 的 hidden state 只包含文档 A 的上下文。MF 要求它根据文档 B 的首 token 预测 B 的第二个 token，A 和 B 之间没有因果关系，这是一个噪声 target。
- 影响：本实验数据中每篇文档的首位 label 都是 −100，跨文档的 MTP target 数为 0，loss 不受影响；只有位置 e−1 的 MTP 输入不同（MF 为 x[e]，HP 为 id 0），经 MTP 层的 router 影响 aux loss 和 expert bias。如果数据中下一篇文档的首位 label 有监督，MF 会在每个文档边界多训练一个跨文档 target。



</details>
