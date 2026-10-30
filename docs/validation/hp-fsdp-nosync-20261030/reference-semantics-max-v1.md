# 梯度同步：固定版本源码语义与 CPU 公式复核

本报告只记录已核对的源码路径和 CPU 实验的适用范围；源码结论为静态分析，并非三框架的实测或性能比较。

## PyTorch FSDP2

固定版本：v2.12.0，提交 `0d62256a2b23365f8e1604297eb23a6545102aa8`。`set_requires_gradient_sync(False)` 在 HSDP 下同时关闭 reduce-scatter 和 all-reduce；重新启用后，`FSDPParamGroup.post_backward` 可以读取先前保留的 `unsharded_accumulated_grad`。root 的 backward final callback 遍历共享上下文中的 FSDP state，为未运行 post-backward 的 group 补调用该方法。适用条件包括最后一次 backward 确实触发该 callback、设置了最后一次同步，以及各 rank 的 collective 参数和顺序匹配；不据此推广到任意动态路径。

源码：[同步开关](https://github.com/pytorch/pytorch/blob/0d62256a2b23365f8e1604297eb23a6545102aa8/torch/distributed/fsdp/_fully_shard/_fully_shard.py#L326-L371)、[root callback](https://github.com/pytorch/pytorch/blob/0d62256a2b23365f8e1604297eb23a6545102aa8/torch/distributed/fsdp/_fully_shard/_fsdp_state.py#L316-L355)、[参数组 post-backward](https://github.com/pytorch/pytorch/blob/0d62256a2b23365f8e1604297eb23a6545102aa8/torch/distributed/fsdp/_fully_shard/_fsdp_param_group.py#L477-L507)。

## TorchTitan

固定提交：`ad7a9937b5e57ccadebbb98b3eb14ff139ceb2b3`。非流水线且启用 `fsdp_defer_gradient_reduction` 时，训练循环设置 `set_requires_gradient_sync(last)`；启用 HSDP replica、但不推迟完整梯度归约时，`set_requires_all_reduce(last)` 只推迟 replica all-reduce，早期 microbatch 仍进行 reduce-scatter。[训练循环](https://github.com/pytorch/torchtitan/blob/ad7a9937b5e57ccadebbb98b3eb14ff139ceb2b3/torchtitan/training_engine.py#L699-L739)

与上述 PyTorch v2.12 源码组合时，后一模式使用 `_partial_reduce_output`；整个 group 在最后 backward 没有新梯度时，源码显示其 post-backward 可能提前返回而未处理此 buffer。这是源码组合的静态推断，未实测，也未确认 TorchTitan 实际运行时使用该 PyTorch 版本；流水线分支未在此分析。[归约缓存](https://github.com/pytorch/pytorch/blob/0d62256a2b23365f8e1604297eb23a6545102aa8/torch/distributed/fsdp/_fully_shard/_fsdp_collectives.py#L592-L619)、[提前返回](https://github.com/pytorch/pytorch/blob/0d62256a2b23365f8e1604297eb23a6545102aa8/torch/distributed/fsdp/_fully_shard/_fsdp_param_group.py#L492-L523)。

## Megatron-LM

固定提交：`a77288b18f1a3e5a7bfe2e680f6f693a7366c2e5`。在普通非流水线、非 combined-MoE 路径中，若 `config.no_sync_func` 已绑定 DDP 的 `no_sync`，前 N−1 次在该上下文内执行，最后一次在外执行；训练模式且 `finalize_model_grads_func` 非空时才调用最终收尾。DDP hook 将梯度累积到 `main_grad`。

非重叠归约的 `finish_grad_sync` 主动同步每个 bucket；重叠归约后续 batch 依赖最后 microbatch 的 ready counts，缺席可能使 bucket 未派发并触发断言。首 batch 与其 golden counts 建立有特殊处理；不能据此推广到任意动态未使用参数、流水线、MoE 特殊路径或多 DistOpt 配置。

源码：[schedule](https://github.com/NVIDIA/Megatron-LM/blob/a77288b18f1a3e5a7bfe2e680f6f693a7366c2e5/megatron/core/pipeline_parallel/schedules.py#L711-L791)、[DDP hook 和 no_sync](https://github.com/NVIDIA/Megatron-LM/blob/a77288b18f1a3e5a7bfe2e680f6f693a7366c2e5/megatron/core/distributed/distributed_data_parallel.py#L547-L590)、[finalize](https://github.com/NVIDIA/Megatron-LM/blob/a77288b18f1a3e5a7bfe2e680f6f693a7366c2e5/megatron/core/distributed/finalize_model_grads.py#L675-L679)、[bucket 收尾](https://github.com/NVIDIA/Megatron-LM/blob/a77288b18f1a3e5a7bfe2e680f6f693a7366c2e5/megatron/core/distributed/param_and_grad_buffer.py#L778-L819)、[ready counts](https://github.com/NVIDIA/Megatron-LM/blob/a77288b18f1a3e5a7bfe2e680f6f693a7366c2e5/megatron/core/distributed/param_and_grad_buffer.py#L848-L869)。

## CPU Gloo 结果的独立公式复核

`tp-grad-sync-cpu-actual-v3/output` 是两进程 CPU scheduler seam、Gloo SUM 的实验；它并非公开 `fully_shard(cpu)` 路径、Ascend 后端或完整模型的验证。

令 loss 为 `(XW).sum()`，则 W 的每行梯度为对应输入分量之和，四列相同。两 rank、两个 microbatch 的输入分别为：rank 0 `[1,1,1,2]`、`[1,2,2,2]`；rank 1 `[2,1,2,2]`、`[2,2,3,2]`。因此参与两次的 common 梯度四行应为 `[6,6,8,8]`；只参与 micro0 的 early 应为 `[3,2,3,4]`，按第 0 维切成 rank 0 的 `[3,2]` 和 rank 1 的 `[3,4]`，每行四列相同。

记录中 `micro0_only_false` 两 rank 的 early 主梯度均为 `None`，各 rank 的 micro0 本地未分片梯度仍留在 pending buffer；common 及 `both_false`、`both_true`、`micro0_only_true` 对照符合各自公式。此结论只覆盖该实验条件，不证明完整模型会触发同一路径。