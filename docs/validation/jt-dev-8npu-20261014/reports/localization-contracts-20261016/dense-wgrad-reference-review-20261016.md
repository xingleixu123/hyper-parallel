# FP32 权重梯度诊断：MF PyNative / Megatron 源码对照

核对时间：2026-10-16 13:07–13:14（UTC+8）。这是固定源码阅读和当前 FSDP 单轮证据复核；CP2 配对尚未运行，不是训练验收或跨框架性能评测。

## 版本与读取范围

按用户记录的官方入口，MF GitCode `master` 和官方 GitHub `master` 均返回 `5089da308ebcc146d0cbe32180c209c4f1d39d77`。Megatron 官方 `main` 已更新到 `6e7e6ed435e41af04fc41e7007ef862049153fdc`，已获取该 Git 对象并按固定提交读源码。没有把参考 checkout 的旧 HEAD 当作最新代码。

版本查询在 [heads 记录](dense-wgrad-reference-heads-20261016.json)，Megatron 六份相关文件的前后 SHA 在 [逐文件复核](dense-wgrad-megatron-latest-source-check-20261016.json)。相对前次 `3914483`，Linear 梯度实现、model parallel config、recompute 主实现、transformer config 和 MoE layer 字节相同；transformer layer 改动 residual read/write 操作选择及调用方式。不能把“部分文件相同”写成所有重算相关源码完全未变。此前重算能力报告的固定版本结论仍对应其原提交。

## 三种精度阶段必须分开

| 阶段 | 源码证据 | 本轮含义 |
|---|---|---|
| 局部 GEMM 的权重梯度输出/累加 | Megatron 的 fusion 路径按 main_grad dtype 选择 FP32 或低精度累加；FSDP 有专门取缓冲入口 | 输出已舍入为 BF16 后再升 FP32，不能恢复丢失的信息；这正是候选希望避免的阶段 |
| 分布式归约输入与通信 dtype | MF HSDP 显式配置 reduce_dtype=float32；Megatron 新 MFSDPv2 修复使 partial_grad 服从 grad_comm_dtype | 需要检查真正参与归约的缓冲，而不仅看配置字段或最终梯度 dtype |
| 微批/优化器主梯度累积 | MF 多卡使用 main_grad，单卡用 hook 把低精度梯度转 FP32；Megatron 直接主梯度累加另有融合路径 | FP32 归约/主梯度与局部 GEMM 输出精度不是同一承诺 |

MF 的明确依据是 [HSDP mixed precision 设置](https://github.com/mindspore-ai/mindformers/blob/5089da308ebcc146d0cbe32180c209c4f1d39d77/mindformers/pynative/base_models/gpt/parallelize.py#L1357)、[单卡梯度 hook](https://github.com/mindspore-ai/mindformers/blob/5089da308ebcc146d0cbe32180c209c4f1d39d77/mindformers/pynative/trainer/trainer.py#L867) 和 [优化器梯度访问](https://github.com/mindspore-ai/mindformers/blob/5089da308ebcc146d0cbe32180c209c4f1d39d77/mindformers/pynative/trainer/utils.py#L115)。这些代码没有证明所有 MindSpore 局部 Linear dW 都直接输出 FP32，也没有本轮 MindSpore 训练测量。不能把它们引用成“MF 的每个 Linear 都已开启本候选”。

Megatron 的 [Linear 反向](https://github.com/NVIDIA/Megatron-LM/blob/6e7e6ed435e41af04fc41e7007ef862049153fdc/megatron/core/tensor_parallel/layers.py#L814) 则显式处理 FP32 GEMM 主梯度累加。其 [config 默认](https://github.com/NVIDIA/Megatron-LM/blob/6e7e6ed435e41af04fc41e7007ef862049153fdc/megatron/core/model_parallel_config.py#L285) 为 gradient_accumulation_fusion=false，依赖相应 CUDA 扩展；不能说默认对所有模型开启，也不能把 CUDA kernel 或调度假设直接搬到 Ascend。

新主线中的 [MFSDPv2 partial_grad dtype 修复](https://github.com/NVIDIA/Megatron-LM/commit/0adca6528c9fb100cd4ccc425f410795de9beafa) 展示了另一种容易遗漏的阶段：归约输入原先取 grads[0].dtype，修复后取显式通信 dtype。它属于 Megatron 的实现；尚未发现或复现相同缺陷存在于本 JT 固定源，不据此另开 HyperParallel 问题单。

## 可借鉴的合约及当前诊断边界

Megatron 在 forward 捕获共享输出头使用的主梯度引用，并避免把会原地累积的 main_grad 放入 saved_tensors；MTP 多次复用头部时尤其需要准确的缓冲生命周期。FSDP 分支从参数管理对象获取实际缓冲。存在 grad_added_to_main_grad 协议时，它仍提供用于触发 backward hook 的占位梯度。具体参考 [保存与复用](https://github.com/NVIDIA/Megatron-LM/blob/6e7e6ed435e41af04fc41e7007ef862049153fdc/megatron/core/tensor_parallel/layers.py#L701) 与 [占位梯度](https://github.com/NVIDIA/Megatron-LM/blob/6e7e6ed435e41af04fc41e7007ef862049153fdc/megatron/core/tensor_parallel/layers.py#L834)。本诊断使用确定的零占位，不能照搬其中未初始化占位而忽略本地 FSDP 如何消费普通梯度。

当前诊断复用已测试的 NPU grouped-matmul-add，保留 BF16 前向和 dX，直接累积明确 owner 的 FP32 dW。行切片验证 storage、offset、宽度及范围；FSDP 接管后清空 owner.main_grad/flag，每步通过弱引用检查消费状态，不留激活或参数强引用副本。生产 projected-cross-entropy 的非重入重放和共享输出头已在小规模 CPU 用例验证，NPU FSDP8 则实际完成200步：8 rank共38,400次累积、23个Linear全部覆盖、第一步前向各loss对原配方差0。

这只是诊断，尚无通用 TP/EP wrapper、融合开关或产品 PR。TP/sequence parallel 必须保留 gather/reduce-scatter 顺序、DTensor 本地 owner、梯度 dtype 和 stream 完成约束；Megatron 的 CUDA 异步并发条件不构成 Ascend 兼容性证明。若200步配对支持该方案，产品实现还需验证冻结/共享/未用参数、空输入、混合梯度生产者、no-sync/sync、全量/选择重放、保存恢复和各并行布局，不能用全局 monkeypatch 作为正式接口。

## 已测成本，不作收益承诺

本候选 FSDP8 的全rank累计 allocated/reserved 峰值为24.762050629/28.714843750 GiB；21–200步逐步rank MAX后的中位时间为1.185278475秒。相对原配方同口径，allocated增加2.729854584 GiB（12.3903%），中位时间增加2.6213%，监督token吞吐降低2.5443%。数值配方和有界导数元数据采集均发生变化，不能把全部差额归因于单个GEMM；这里只是单次原型成本观测。

额外FP32权重梯度缓冲是合理的显存成本假设，尚无分配栈证明全部2.73GiB来自该缓冲。候选是否值得采用，仍取决于匹配CP2及后续拓扑的精度改善，不能仅因算子oracle更精确就默认启用。见 [FSDP完整运行汇总](f637-dense-wgrad-fsdp8-200-v1-summary.json) 和 [逐步覆盖审计](f637-dense-wgrad-fsdp8-200-v1-audit.json)。
