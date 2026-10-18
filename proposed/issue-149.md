<!-- correctness-review-20261018 -->
## 2026-10-18 正确性复核：Muon 分组/缩放：区分存储契约、优化配方与写回精度

**当前判断**：本单的旧训练图比较保留为历史证据，不能直接等同最新 MF PyNative 或当前 JT 的状态。三个因素必须分开，不能整体判为“HP 都应向 MF 靠齐”。

1. **存储等价性**：#242 / PR #243 针对 JT 明确由原 Q 与 latent-KV、expert gate/up 融合存储的边界。若只重排存储，应保持原逐矩阵优化；最小验收是独立参数与融合存储的更新相等。进一步拆 latent/RoPE、Q-up/KV-up 是另一个配方。
2. **shape scaling**：旧归档 Q-up 共用末片 shape 的行为不推荐用于新训练。当前 [MF 每个 piece 独立缩放](https://gitcode.com/mindspore/mindformers/blob/409019aac0c6579bb4fb8a9b04a75c259d98a96e/mindformers/pynative/optimizer/muon.py#L1238) 已在循环内按各自最后两个维度计算 scale，不能把旧缺陷写成最新 master 仍未修复。
3. **已有 FP32 update 的写回**：#215 / PR #213 只保护已经产生的 FP32 更新；当前默认 Muon 输入/更新通常已 BF16，不能把覆写 NS 的归档实验收益泛化到默认路径。

**行业依据**：[MF master PyNative 矩阵 schema](https://gitcode.com/mindspore/mindformers/blob/409019aac0c6579bb4fb8a9b04a75c259d98a96e/mindformers/pynative/base_models/gpt/gpt_model.py#L798)、[Megatron 可选 QKV 拆分](https://github.com/NVIDIA/Megatron-LM/blob/1effc75a46b26c6b93478355ff9fc3bb8ac1fafc/megatron/core/optimizer/emerging_optimizers.py#L458)。Megatron 的 `split_qkv` 默认 false，部分 GTP 模式还有整矩阵 fallback，因此它不能证明“任何融合矩阵整体 NS 都是 bug”。PR #243 的依据是本 JT 的原投影存储契约。

**修改方向与验收**：保留契约修复及明确的旧配方退出开关；用独立/融合更新等价、精确触发 dtype、恢复状态与迁移说明验收。旧 MF schema、epsilon 定义、系数或算术顺序不直接改成通用默认。长程最大误差变化只是特定背景影响，不是正确性或可相加贡献率。


本次固定参考：MF GitCode master `409019aac0c6579bb4fb8a9b04a75c259d98a96e`（只核查 `mindformers/pynative/`，未用旧 training_graph 冒充），Megatron main `1effc75a46b26c6b93478355ff9fc3bb8ac1fafc`，当前 JT_dev `a1e6c222eab097c9609d6e744a3c5f98aeed6bdc`。源码路径/行号/SHA-256 见[完整正确性复核](https://github.com/xingleixu123/hyper-parallel/blob/xxl-dev/jt-correctness-review-20261018/AUDIT.md)与[来源清单](https://github.com/xingleixu123/hyper-parallel/blob/xxl-dev/jt-correctness-review-20261018/source-catalog.json)。本轮不新增训练结果，不以 MF loss 距离代替独立正确性判据。


<details><summary>原问题与历史实验记录（适用旧基线；当前判断以上方复核为准）</summary>

两边的 Muon 更新公式相同：Nesterov 动量 β = 0.95；5 步 Newton–Schulz（NS，legacy 系数，bf16）；缩放 √max(rows, cols)·0.2；解耦 weight decay。不一致的有三处：

| 不一致 | 实验编号 | 判定 |
|---|---|---|
| 1. 融合矩阵做 NS 时怎么分块 | 1 | `linear_qkv`、`experts.gate_up_proj`：HP 写法问题；其余分块：框架差异（MF 的设计选择） |
| 2. 缩放系数按哪个维度算 | 1 | MF 写法问题，有对错 |
| 3. 缩放后的更新是否先舍入成 bf16 | 9 | HP 写法问题，只影响精度 |

文中链接：

- HP 指向 [hf1s/hyper-parallel@4fb7739c](https://github.com/hf1s/hyper-parallel/tree/4fb7739cf0216d54c799d29e13e2332bde8b9b7c)，与实验所用代码相同。
- MF 指向 [mindspore-ai/mindformers r2.1.0](https://github.com/mindspore-ai/mindformers/tree/c9a1697718614b4a63aec1379681ab5ca413c2d5)（`c9a16977`），引用的逻辑与实验所用版本相同。
- HF 指向 [huggingface/transformers v5.13.1](https://github.com/huggingface/transformers/tree/v5.13.1)，即实验中 HP 使用的版本。

## 1. 融合矩阵的 NS：MF 分块，HP 整体

**MF**：先按 schema 把融合矩阵拆成块，每块单独做 NS，再合并回原参数。

- schema：[`gpt_model.py:838-880`](https://github.com/mindspore-ai/mindformers/blob/c9a1697718614b4a63aec1379681ab5ca413c2d5/mindformers/parallel_core/training_graph/base_models/gpt/gpt_model.py#L838-L880)（`make_model_muon_fns`）
- 拆分规则：[`muon_utils.py:183-207`](https://github.com/mindspore-ai/mindformers/blob/c9a1697718614b4a63aec1379681ab5ca413c2d5/mindformers/core/optim/muon_utils.py#L183-L207)（`muon_split`）、[`muon_utils.py:240-286`](https://github.com/mindspore-ai/mindformers/blob/c9a1697718614b4a63aec1379681ab5ca413c2d5/mindformers/core/optim/muon_utils.py#L240-L286)（`make_muon_fns`）
- 逐块 NS 后合并：[`muon.py:129-146`](https://github.com/mindspore-ai/mindformers/blob/c9a1697718614b4a63aec1379681ab5ca413c2d5/mindformers/core/optim/muon.py#L129-L146)

**HP**：整个参数只做一次 NS。

- 分块钩子 `reshape_fn` / `ns_transform_fn` 默认是 `None`：[`muon.py:83-84`](https://github.com/hf1s/hyper-parallel/blob/4fb7739cf0216d54c799d29e13e2332bde8b9b7c/hyper_parallel/core/optimizer/muon.py#L83-L84)
- JT 的 `build_optimizer` 不设置这两个钩子：[`jt_optimizer.py:137-159`](https://github.com/hf1s/hyper-parallel/blob/4fb7739cf0216d54c799d29e13e2332bde8b9b7c/hyper_parallel/models/jt_deepseek_v3/adapter/runtime/jt_optimizer.py#L137-L159)
- `reshape_fn is None` 时，整个参数作为一块进入 NS：[`muon.py:746-747`](https://github.com/hf1s/hyper-parallel/blob/4fb7739cf0216d54c799d29e13e2332bde8b9b7c/hyper_parallel/core/optimizer/muon.py#L746-L747)

**分块不同的参数**（128 头，qk_nope 128，qk_rope 64，v 128，q_lora 1536，kv_lora 512，hidden 5120）：

| HP 参数（形状） | MF 对应参数与分块 | HP |
|---|---|---|
| `self_attn.q_b_proj.weight`（24576×1536） | `linear_q_up_proj`，`periodic` (128, 64, 128)：每个头拆成 nope / rope，得到 16384×1536 和 8192×1536 两块 | 24576×1536 整体 |
| `self_attn.linear_qkv.weight`（2112×5120） | 两个参数：`linear_q_down_proj` 1536×5120 不拆；`linear_kv_down_proj` 576×5120 按 `periodic` (512, 64, 1) 拆成 512×5120 和 64×5120 | 2112×5120 整体 |
| `self_attn.kv_b_proj.weight`（32768×512） | `linear_kv_up_proj`，`periodic` (128, 128, 128)：每个头拆成 k_nope / v，得到两块 16384×512 | 32768×512 整体 |
| `mlp.experts.gate_up_proj`（8×3072×5120） | `experts.weight1`，`reshape_concat`：每个专家拆成 gate / up 两块 | 每个专家的 gate+up 整体 |

**两边一致的参数**：

- `o_proj`、`down_proj`、`experts.down_proj`、router gate、`eh_proj` 两边都是整体做 NS。
- dense MLP 和 shared experts 的 fc1：MF 按 `alt_pair_periodic` 拆成 gate / up；HP 本来就是分开的 `gate_proj` / `up_proj`，结果等价。

**判定**：分两类。

- `linear_qkv`、`experts.gate_up_proj`：HP 写法问题。
  - 这两个参数是存储层面的融合。`linear_qkv` 是 JT 把 HF 的 `q_a_proj` 和 `kv_a_proj_with_mqa` 拼成的一个参数（[`modeling_jt_deepseek_v3.py:172-174`](https://github.com/hf1s/hyper-parallel/blob/4fb7739cf0216d54c799d29e13e2332bde8b9b7c/hyper_parallel/models/jt_deepseek_v3/modeling_jt_deepseek_v3.py#L172-L174)）；`gate_up_proj` 是 HF 专家的存储方式（[`modeling_deepseek_v32.py:544`](https://github.com/huggingface/transformers/blob/v5.13.1/src/transformers/models/deepseek_v32/modeling_deepseek_v32.py#L544)）。
  - HP 按参数整体做 NS，结果因此取决于存储方式。同样是 SwiGLU 的 gate / up：dense MLP 和 shared experts 里是两个参数（[`modeling_deepseek_v32.py:489-490`](https://github.com/huggingface/transformers/blob/v5.13.1/src/transformers/models/deepseek_v32/modeling_deepseek_v32.py#L489-L490)），分开做；routed experts 里存成一个参数，合在一起做。
  - 分块用的钩子 HP 已经提供（上面的 `muon.py:83-84`），JT 没有配置。
- `q_b_proj`、`kv_b_proj` 按头拆分，`linear_kv_down_proj` 拆成 512 / 64：框架差异。这是 MF 的设计选择，比 HF 的参数粒度更细，没有公认的标准。

## 2. 缩放系数：MF 用最后一块的维度，HP 用整个参数的维度

**MF**：[`muon.py:132-150`](https://github.com/mindspore-ai/mindformers/blob/c9a1697718614b4a63aec1379681ab5ca413c2d5/mindformers/core/optim/muon.py#L132-L150)

```python
dim_a, dim_b = None, None
for ns_inputs_item in ns_inputs_list:
    dim_a, dim_b = op_shape(ns_inputs_item)[-2:]
    ...
adjusted_ratio = op_sqrt(op_cast(max(dim_a, dim_b), mstype.float32)) * matched_adamw_rms
```

- `dim_a, dim_b` 在循环里被逐块覆盖，循环结束后只剩最后一块的维度。
- 用它算出的 `adjusted_ratio` 乘在合并后的整个参数上。

**HP**：整个参数按自身的 √max(rows, cols)·0.2 缩放。

**影响范围**：只有一个参数内各块的 max(dim) 不同时，两边的缩放才不一样。本模型中只有 `q_b_proj` 属于这种情况：nope 块 16384 行，rope 块 8192 行。

| `q_b_proj` | 缩放系数 | 每步更新 ‖ΔW‖_F ÷ lr | 逐元素 RMS ÷ lr |
|---|---|---|---|
| MF | nope、rope 两块都是 √8192·0.2 = 18.10 | 1003 | nope 0.141，rope 0.200 |
| HP | √24576·0.2 = 31.35 | 1229 | 平均 0.200 |

‖ΔW‖_F 按 NS 输出的奇异值 ≈ 1 估算：MF 为 √(2·1536)·18.10，HP 为 √1536·31.35。

**判定**：MF 写法问题，有对错。

- 缩放规则是按矩阵取 √max(rows, cols)·0.2。分块之后，按每块算或按整个参数算都说得通，但都不会只取最后一块。
- `dim_a, dim_b` 在循环中被覆盖，看起来是循环变量泄漏（推断）。

## 3. 缩放后的更新：HP 先舍入成 bf16，MF 在 fp32 里乘

**MF**：[`muon.py:150-153`](https://github.com/mindspore-ai/mindformers/blob/c9a1697718614b4a63aec1379681ab5ca413c2d5/mindformers/core/optim/muon.py#L150-L153)

```python
adjusted_ratio = op_sqrt(op_cast(max(dim_a, dim_b), mstype.float32)) * matched_adamw_rms
adjusted_lr = lr * adjusted_ratio
update_with_lr = adjusted_lr * x_ret
next_param = param_fp32 - op_reshape(update_with_lr, op_shape(param_fp32))
```

- `x_ret` 是 bf16 的 NS 输出，乘上 fp32 的 `adjusted_lr` 后在 fp32 里从参数中减去。

**HP**：

- NS 的输入和输出是 bf16：[`muon.py:496`](https://github.com/hf1s/hyper-parallel/blob/4fb7739cf0216d54c799d29e13e2332bde8b9b7c/hyper_parallel/core/optimizer/muon.py#L496)、[`muon.py:546-547`](https://github.com/hf1s/hyper-parallel/blob/4fb7739cf0216d54c799d29e13e2332bde8b9b7c/hyper_parallel/core/optimizer/muon.py#L546-L547)
- 缩放系数原地乘进这个 bf16 张量：[`muon.py:969-974`](https://github.com/hf1s/hyper-parallel/blob/4fb7739cf0216d54c799d29e13e2332bde8b9b7c/hyper_parallel/core/optimizer/muon.py#L969-L974)

  ```python
  scale = compute_muon_slice_scale(
      update, group["matched_adamw_rms"], zero_rms_scale_mode=group["zero_rms_scale_mode"]
  )
  ...
  update.mul_(scale)
  ```

- 然后以 `alpha=-lr` 加到 fp32 参数上：[`muon.py:585`](https://github.com/hf1s/hyper-parallel/blob/4fb7739cf0216d54c799d29e13e2332bde8b9b7c/hyper_parallel/core/optimizer/muon.py#L585)

**判定**：HP 写法问题，只影响精度，没有语义上的对错。`scale·NS` 多了一次 bf16 舍入，相对误差最大约 2⁻⁹（0.2%）。参数更新本身在 fp32 里做，这次舍入不是框架必需的。


</details>
