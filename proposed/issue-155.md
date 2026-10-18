<!-- correctness-review-20261018 -->
## 2026-10-18 正确性复核：SwiGLU：融合与分离是实现选择，按导数和容差判定

**当前判断**：`silu(gate) * up` 的融合/分离路径一般会有舍入差异；仅因单卡和 EP 采用不同 kernel，不能直接判为正确性 bug。必须统一 gate/up 顺序、激活定义（包括是否 clamp）、输入 dtype，再判断输出和导数是否符合协议。

**最新源码**：[MF PyNative expert 激活路径](https://gitcode.com/mindspore/mindformers/blob/409019aac0c6579bb4fb8a9b04a75c259d98a96e/mindformers/pynative/transformers/moe/experts.py#L216) 在 `activation_type == fusedswiglu` 时走融合算子，其它 gated 路径分开激活/相乘。最新 MF 自己就支持不同激活路径；[Megatron MLP 配置与激活路径](https://github.com/NVIDIA/Megatron-LM/blob/1effc75a46b26c6b93478355ff9fc3bb8ac1fafc/megatron/core/transformer/mlp.py#L1) 也依据融合配置选择实现。不能把旧 MF 某个默认路径说成唯一正确实现。

**需撤回的泛化**：融合不自动意味着更高精度；只看 Python 调用不能证明 CANN 内部累加 dtype 或任意输入的 FP64 误差更小。历史原文对此使用了推断，本次不把推断升级为已核实事实。

**修改方向与验收**：出现 gate/up 布局错误、clamp/导数错误或超过明确误差契约时再修复。用独立 FP64 forward/backward、极值/饱和/零输入及单卡/EP 同配方对照验收；性能改进单独实测。不为制造逐位 MF 一致而统一所有 kernel。


本次固定参考：MF GitCode master `409019aac0c6579bb4fb8a9b04a75c259d98a96e`（只核查 `mindformers/pynative/`，未用旧 training_graph 冒充），Megatron main `1effc75a46b26c6b93478355ff9fc3bb8ac1fafc`，当前 JT_dev `a1e6c222eab097c9609d6e744a3c5f98aeed6bdc`。源码路径/行号/SHA-256 见[完整正确性复核](https://github.com/xingleixu123/hyper-parallel/blob/xxl-dev/jt-correctness-review-20261018/AUDIT.md)与[来源清单](https://github.com/xingleixu123/hyper-parallel/blob/xxl-dev/jt-correctness-review-20261018/source-catalog.json)。本轮不新增训练结果，不以 MF loss 距离代替独立正确性判据。


<details><summary>原问题与历史实验记录（适用旧基线；当前判断以上方复核为准）</summary>

### Checklist

- [ ] 1. I have searched the existing issues (https://gitcode.com/mindspore/hyper-parallel/issues)
- [ ] 2. I have read the relevant documentation.
- [ ] 3. I have created a minimal reproduction case that clearly demonstrates the issue, including a complete code example and the error message with full traceback and error logs.

### 🐛 Describe the bug

# SwiGLU：MindFormers 与 Hyper-Parallel 的不一致

dense MLP 和 shared experts 两边都是先算 silu、再与 up 相乘，写法一致：

- MF：实验配置 `use_interleaved_weight_layout_mlp: True`，dense MLP 用 `MLPInterleaved`（[`gpt_layer_specs.py:46`](https://github.com/mindspore-ai/mindformers/blob/c9a1697718614b4a63aec1379681ab5ca413c2d5/mindformers/parallel_core/training_graph/base_models/gpt/gpt_layer_specs.py#L46)），shared experts 用 `SharedExpertMLPInterleaved`（[`moe_module_specs.py:40`](https://github.com/mindspore-ai/mindformers/blob/c9a1697718614b4a63aec1379681ab5ca413c2d5/mindformers/parallel_core/training_graph/base_models/gpt/moe_module_specs.py#L40)）；`hidden_act: silu` 时两者都走 [`mlp.py:220-225`](https://github.com/mindspore-ai/mindformers/blob/c9a1697718614b4a63aec1379681ab5ca413c2d5/mindformers/parallel_core/training_graph/transformer/mlp.py#L220-L225)。
- HP：HF 的 `DeepseekV32MLP`，[`modeling_deepseek_v32.py:489-490`](https://github.com/huggingface/transformers/blob/v5.13.1/src/transformers/models/deepseek_v32/modeling_deepseek_v32.py#L489-L490)。

不一致的只有 routed experts：

| 不一致 | 实验编号 | 判定 |
|---|---|---|
| routed experts 的 SwiGLU 是否融合 | 10 | 框架差异，没有对错；另有 HP 内部单卡与 EP 走不同实现 |

文中链接：

- HP 指向 [hf1s/hyper-parallel@4fb7739c](https://github.com/hf1s/hyper-parallel/tree/4fb7739cf0216d54c799d29e13e2332bde8b9b7c)，与实验所用代码相同。
- MF 指向 [mindspore-ai/mindformers r2.1.0](https://github.com/mindspore-ai/mindformers/tree/c9a1697718614b4a63aec1379681ab5ca413c2d5)（`c9a16977`），引用的逻辑与实验所用版本相同。
- HF 指向 [huggingface/transformers v5.13.1](https://github.com/huggingface/transformers/tree/v5.13.1)，即实验中 HP 使用的版本。

## 1. routed experts：MF 用融合的 Swiglu，HP 单卡路径分开算

**MF**：GroupedMatmul → 融合 `Swiglu` → GroupedMatmul：[`ffn.py:130`](https://github.com/mindspore-ai/mindformers/blob/c9a1697718614b4a63aec1379681ab5ca413c2d5/mindformers/parallel_core/training_graph/transformer/moe/ffn.py#L130)、[`ffn.py:178-181`](https://github.com/mindspore-ai/mindformers/blob/c9a1697718614b4a63aec1379681ab5ca413c2d5/mindformers/parallel_core/training_graph/transformer/moe/ffn.py#L178-L181)

**HP**：

- 单卡（不开 EP）走 `local_routed_forward`，逐专家 `F.linear` 后分开算 silu 和乘法：[`modeling_jt_deepseek_v3.py:470-473`](https://github.com/hf1s/hyper-parallel/blob/4fb7739cf0216d54c799d29e13e2332bde8b9b7c/hyper_parallel/models/jt_deepseek_v3/modeling_jt_deepseek_v3.py#L470-L473)

  ```python
  pair = F.linear(flat[tokens], self.experts.gate_up_proj[expert])
  gate, up = pair.chunk(2, dim=-1)
  values = F.silu(gate) * up
  values = F.linear(values, self.experts.down_proj[expert])
  ```

  silu 的结果先舍入成 bf16，再与 up 相乘。
- 开 EP 时走 `forward_expert_major`，用融合的 `torch_npu.npu_swiglu`：[`modeling_jt_deepseek_v3.py:74-84`](https://github.com/hf1s/hyper-parallel/blob/4fb7739cf0216d54c799d29e13e2332bde8b9b7c/hyper_parallel/models/jt_deepseek_v3/modeling_jt_deepseek_v3.py#L74-L84)、[`npu_grouped_swiglu.py:118`](https://github.com/hf1s/hyper-parallel/blob/4fb7739cf0216d54c799d29e13e2332bde8b9b7c/hyper_parallel/components/functional/npu_grouped_swiglu.py#L118)

**判定**：

- MF 与 HP 之间：kernel 选择的差异，没有对错。融合算子内部的计算精度没有核实；推断为在内部高精度计算、只舍入一次。
- HP 内部：同一个模型在单卡和 EP 下走不同的 SwiGLU 实现，数值随并行配置变化。这是 HP 的写法问题，属于内部一致性，不影响正确性。


### Expected behavior

*

### Additional context

_No response_

### Environment info

*

</details>
