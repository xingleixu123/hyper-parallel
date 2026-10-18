<!-- correctness-review-20261018 -->
## 2026-10-18 正确性复核：Router：计算精度契约已修复，存储精度是独立配置

**当前状态**：本单历史 BF16 GEMM 后才转 FP32 的 JT 路径已由 PR #148（`436a1bdd`，已合入）修复。当前 JT gate 明确对输入和权重 `.float()` 后做 linear；不能继续描述为当前 JT 尚未修复。

**最新源码**：[MF PyNative router](https://gitcode.com/mindspore/mindformers/blob/409019aac0c6579bb4fb8a9b04a75c259d98a96e/mindformers/pynative/transformers/moe/router.py#L348) 按 `moe_router_dtype` 转输入、使用对应 gate 权重，并在 score 路径使用 FP32；[Megatron 路由计算 dtype](https://github.com/NVIDIA/Megatron-LM/blob/1effc75a46b26c6b93478355ff9fc3bb8ac1fafc/megatron/core/transformer/moe/router.py#L141) 明确区分 fp32/fp64/输入 dtype。高精度 router 需依据模型的数值稳定性与配置契约，不是仅因 MF 使用它就应照搬。

**仍需区分**：把已被 FSDP 舍入的 BF16 权重再 `.float()`，不能恢复丢失的存储位数。router 计算 dtype 与参数计算副本 dtype 是两个选择；FP32 GEMM 正确实现了计算契约，不等于权重必然一直保持 FP32。

**验收与动作**：保留已有计算修复；用小矩阵线性与 router 导数参考验证 dtype/梯度，再核对当前真实 import 和配置。若需要 FP32 gate 权重策略，应单独给出配置、精度收益及开销证据，不为旧 MF 轨迹无条件修改 FSDP 全局策略。本轮不新增重复修复 PR。


本次固定参考：MF GitCode master `409019aac0c6579bb4fb8a9b04a75c259d98a96e`（只核查 `mindformers/pynative/`，未用旧 training_graph 冒充），Megatron main `1effc75a46b26c6b93478355ff9fc3bb8ac1fafc`，当前 JT_dev `a1e6c222eab097c9609d6e744a3c5f98aeed6bdc`。源码路径/行号/SHA-256 见[完整正确性复核](https://github.com/xingleixu123/hyper-parallel/blob/xxl-dev/jt-correctness-review-20261018/AUDIT.md)与[来源清单](https://github.com/xingleixu123/hyper-parallel/blob/xxl-dev/jt-correctness-review-20261018/source-catalog.json)。本轮不新增训练结果，不以 MF loss 距离代替独立正确性判据。


<details><summary>原问题与历史实验记录（适用旧基线；当前判断以上方复核为准）</summary>

两边的路由规则相同：sigmoid 打分；加 expert bias 选 top-2；从不含 bias 的分数中取权重，归一化后乘 2.827；expert bias 按 sign(mean − count) 每步 ±0.001。不一致的有两处，都是精度（实验编号 2）：

| 不一致 | 判定 |
|---|---|
| 1. router GEMM 的计算精度 | HP 写法问题，有对错；已在分支 `fix/jt-router-fp32`（436a1bdd）修复 |
| 2. router GEMM 所用权重的精度 | 框架差异，没有对错 |

文中链接：

- HP 指向 [hf1s/hyper-parallel@4fb7739c](https://github.com/hf1s/hyper-parallel/tree/4fb7739cf0216d54c799d29e13e2332bde8b9b7c)，与实验所用代码相同。
- MF 指向 [mindspore-ai/mindformers r2.1.0](https://github.com/mindspore-ai/mindformers/tree/c9a1697718614b4a63aec1379681ab5ca413c2d5)（`c9a16977`），引用的逻辑与实验所用版本相同。
- HF 指向 [huggingface/transformers v5.13.1](https://github.com/huggingface/transformers/tree/v5.13.1)，即实验中 HP 使用的版本。

路由规则：HP [`routing.py:93-136`](https://github.com/hf1s/hyper-parallel/blob/4fb7739cf0216d54c799d29e13e2332bde8b9b7c/hyper_parallel/distributed/expert_parallel/routing.py#L93-L136)；MF [`router.py:259-296`](https://github.com/mindspore-ai/mindformers/blob/c9a1697718614b4a63aec1379681ab5ca413c2d5/mindformers/parallel_core/training_graph/transformer/moe/router.py#L259-L296)、[`router.py:369-377`](https://github.com/mindspore-ai/mindformers/blob/c9a1697718614b4a63aec1379681ab5ca413c2d5/mindformers/parallel_core/training_graph/transformer/moe/router.py#L369-L377)。

## 1. GEMM 精度：HP 在 bf16 里算，MF 在 fp32 里算

**MF**：输入和权重都先转成 `moe_router_dtype`（fp32）再做 GEMM：[`router.py:117-118`](https://github.com/mindspore-ai/mindformers/blob/c9a1697718614b4a63aec1379681ab5ca413c2d5/mindformers/parallel_core/training_graph/transformer/moe/router.py#L117-L118)

```python
weight = self.weight.astype(self.moe_router_dtype)
router_logits = self.linear(inputs.astype(self.moe_router_dtype), weight)
```

**HP**：

- JT 的 gate 覆写了 forward，输入和权重都是 bf16：[`modeling_jt_deepseek_v3.py:353-360`](https://github.com/hf1s/hyper-parallel/blob/4fb7739cf0216d54c799d29e13e2332bde8b9b7c/hyper_parallel/models/jt_deepseek_v3/modeling_jt_deepseek_v3.py#L353-L360)

  ```python
  self.router_logits = F.linear(hidden_states.reshape(-1, self.hidden_dim), self.weight)
  ```

- 之后的 `.float()` 发生在 logits 已经是 bf16 之后：[`routing.py:115-119`](https://github.com/hf1s/hyper-parallel/blob/4fb7739cf0216d54c799d29e13e2332bde8b9b7c/hyper_parallel/distributed/expert_parallel/routing.py#L115-L119)
- aux loss 的 scores 取自同一份 logits（[`modeling_jt_deepseek_v3.py:411`](https://github.com/hf1s/hyper-parallel/blob/4fb7739cf0216d54c799d29e13e2332bde8b9b7c/hyper_parallel/models/jt_deepseek_v3/modeling_jt_deepseek_v3.py#L411)），归一化和均值也在 bf16 里算（[`moe_aux.py:100-101`](https://github.com/hf1s/hyper-parallel/blob/4fb7739cf0216d54c799d29e13e2332bde8b9b7c/hyper_parallel/components/losses/moe_aux.py#L100-L101)）。MF 的 aux 用 fp32 的 `router_prob_for_aux`（[`router.py:271-274`](https://github.com/mindspore-ai/mindformers/blob/c9a1697718614b4a63aec1379681ab5ca413c2d5/mindformers/parallel_core/training_graph/transformer/moe/router.py#L271-L274)）。

**判定**：HP 写法问题，有对错。

- `JTDeepseekV3Gate` 继承自 HF 的 `DeepseekV32TopkRouter`。上游的 forward 先把输入和权重转成 fp32：[`modeling_deepseek_v32.py:507-511`](https://github.com/huggingface/transformers/blob/v5.13.1/src/transformers/models/deepseek_v32/modeling_deepseek_v32.py#L507-L511)

  ```python
  router_logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32))
  ```

- JT 覆写 forward 时去掉了这次转换。MF 和 HF 上游都在 fp32 里算 router。
- 现已提交修改PR：[#148 ](https://github.com/hf1s/hyper-parallel/pull/148)

## 2. 权重精度：HP 用 bf16 计算副本，MF 用 fp32 原值

**MF**：参数以 fp32 存储（`params_dtype: float32`；训练日志中 `decoder.layers.1.mlp.router.weight` 为 Float32；定义见 [`router.py:96`](https://github.com/mindspore-ai/mindformers/blob/c9a1697718614b4a63aec1379681ab5ca413c2d5/mindformers/parallel_core/training_graph/transformer/moe/router.py#L96)），GEMM 直接用这份 fp32 权重（上面的 `router.py:117-118`）。

**HP**：FSDP 混合精度按单元把参数转成 `param_dtype`（实验配置为 bfloat16），没有按参数的例外：[`hsdp_param.py:665-673`](https://github.com/hf1s/hyper-parallel/blob/4fb7739cf0216d54c799d29e13e2332bde8b9b7c/hyper_parallel/core/fully_shard/hsdp_param.py#L665-L673)、[`hsdp_param.py:858-859`](https://github.com/hf1s/hyper-parallel/blob/4fb7739cf0216d54c799d29e13e2332bde8b9b7c/hyper_parallel/core/fully_shard/hsdp_param.py#L858-L859)。即使 GEMM 改成 fp32，用的也是舍入过的 bf16 权重。

**HP 能否做到**：能，FSDP 核心不用改。混合精度按 `fully_shard`单元生效，外层单元会跳过内层单元已管理的参数（[`hsdp_state.py:190-191`](https://github.com/hf1s/hyper-parallel/blob/4fb7739cf0216d54c799d29e13e2332bde8b9b7c/hyper_parallel/core/fully_shard/hsdp_state.py#L190-L191)、[`hsdp_state.py:199-209`](https://github.com/hf1s/hyper-parallel/blob/4fb7739cf0216d54c799d29e13e2332bde8b9b7c/hyper_parallel/core/fully_shard/hsdp_state.py#L199-L209)）。把 gate单独作为一个单元并设`param_dtype=float32`，前向用的就是 fp32原值。但 Trainer 没有留配置入口：所有单元共用同一个由`fsdp_config.mix_precision`生成的策略（[`fsdp_adapter.py:198-216`](https://github.com/hf1s/hyper-parallel/blob/4fb7739cf0216d54c799d29e13e2332bde8b9b7c/hyper_parallel/distributed/_builder/fsdp_adapter.py#L198-L216)、[`fsdp_adapter.py:614-628`](https://github.com/hf1s/hyper-parallel/blob/4fb7739cf0216d54c799d29e13e2332bde8b9b7c/hyper_parallel/distributed/_builder/fsdp_adapter.py#L614-L628)）；`ModelAdapterSpec.fsdp_wrap_modules`只能声明单元，不能指定精度（[`adapter_spec.py:78-82`](https://github.com/hf1s/hyper-parallel/blob/4fb7739cf0216d54c799d29e13e2332bde8b9b7c/hyper_parallel/models/adapter_spec.py#L78-L82)）。

**判定**：不是框架能力差异，是 HP Trainer缺少按模块设置精度的配置入口。

- HF 模型用 FSDP 混合精度训练时默认都是这样。MF把参数的存储精度和计算精度分开设置。


</details>
