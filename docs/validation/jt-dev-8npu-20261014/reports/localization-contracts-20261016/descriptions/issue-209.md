总问题单：[#202](https://github.com/hf1s/hyper-parallel/issues/202)。本单只跟踪一个独立问题；对应 PR 和验证边界见下文。

## 检查清单

- [x] 已检索 hf1s/hyper-parallel 的全部现有 issue，并核对相关问题的范围。
- [x] 已阅读本问题涉及的源码及回归接口。
- [x] 已保留可执行复现/回归用例和实际结果。

## 问题描述与影响

优先级：本轮 Top8。JT grouped SwiGLU先产出BF16/FP16专家权重梯度，再交FSDP归约；设置 `reduce_dtype=float32` 只提高归约阶段精度，不能恢复局部梯度已经丢失的信息。token/专家分区改变局部舍入位置，Muon会进一步放大差异。这是当前路径的精度限制，不意味着浮点非结合性本身就是错误，也不能据此默认改配方。

## 预期行为与验收标准

提供默认关闭的 `model.config.expert_fp32_weight_gradients`：保持原前向/dgrad kernel，wgrad直接累加到所属参数实际布局的FP32 main_grad。显式所属参数、避免额外全量转置FP32副本；FSDP在正常消费点处理一次，并保留普通autograd贡献。空专家、延迟同步、多微批、共享权重及重算均须正确，消费后释放缓冲/标志。

## 复现步骤、证据与修复边界

最小算子及真实FSDP回归在PR中：

```bash
pytest -q tests/ut/core/fully_shard/test_fused_grad.py tests/ut/components/functional/test_npu_grouped_swiglu.py
pytest -q tests/torch/fully_shard/test_fused_expert_gradients.py
```

CPU定向套件87 passed、25 subtests；旧实现针对性测试6失败、3通过。真实2卡ST1 passed、23.22秒：两轮×三微批，off/non-reentrant两模式，空专家、全空微批、同一权重两次融合使用＋普通梯度，对CPU FP64参考 `rtol=1e-6, atol=2e-6`；重算开关要求零误差。

commit `630d54358f94e34dc8e7f22c6a57d4043ba211f5`。此前125的联合#201生产配方FSDP0–200完成完整性自检；129当前已完成基线及续训，独立连续参考与跨拓扑200步验收尚待完成。早期不同梯度布局的诊断wrapper长跑仍失败，不能作为本补丁验收。两项选项合用FSDP步时+24.25%，allocated22.282196GiB，属于[组合成本](https://github.com/xingleixu123/hyper-parallel/blob/66464cdd569d864562578eb45e9aa234d81770f4/docs/validation/jt-dev-8npu-20261014/reports/matching-build-topology-costs.md)，不是本补丁单独的收益或成本。

独立 PR：[#200](https://github.com/hf1s/hyper-parallel/pull/200)。每个 PR 一个 commit；本 issue 不使用自动关闭关键字。

[完整记录及按影响排序的报告](https://github.com/xingleixu123/hyper-parallel/blob/66464cdd569d864562578eb45e9aa234d81770f4/docs/validation/jt-dev-8npu-20261014/REPORT_ZH.md)；[配置、曲线与原始标量索引](https://github.com/xingleixu123/hyper-parallel/blob/66464cdd569d864562578eb45e9aa234d81770f4/docs/validation/jt-dev-8npu-20261014/README.md)。

实验已获准在 125/129 空闲时继续，每轮设备释放后至少间隔 20 分钟，不停止其他人的实验。问题单保持开启；草稿 PR 不等于已合入或完整 CI 通过。

## 环境信息

测试基点为 `hf1s/hyper-parallel:JT_dev@ad1c55379767d63ece7e084b0623340b37b83d93`；八卡集成包含 PR #180 head `5de3cceab094bbed18915d1a9c9231ca220629ca`。后续六项修复集成源为 `ec2e16bce26e497519c573f0a8d87a538a329197`，各 PR 的独立 diff 不包含其他修复。

硬件 8×Ascend 910B3 64GiB；CANN 9.2.0-beta.2 / V100R001C12B086；Python 3.12.14，torch 2.12.0+cpu，torch_npu 2.12.0，transformers 5.13.1。训练配置为两层主干＋一层MTP、8个路由专家，seq4096、GBS8、MBS1、seed42、常数LR0.0001。性能结论只适用于该已测配置。


## CPU 默认设备隔离补充（2026-10-15）

最新 JT_dev 的组合回归暴露了 CPU 测试的环境依赖：导入 `torch_npu` 后，checkpoint 默认设备变成 `npu`，即使输入全部在 CPU，回放构造 autocast 上下文时仍会初始化 NPU。最小复现确认 PyTorch 原生和 Hyper 重计算均有相同行为；只显式设为 CPU 后，两条路径的梯度均精确通过。

本次仅补充测试 fixture：在修改前通过 `addCleanup` 登记旧值恢复，再设置 CPU。测试异常也会恢复全局状态；生产设备推断未改，原有 loss、梯度、路由计数、重放断言和容差全部保留。独立工作树的定向 CPU 回归 **13项测试、14项子测试，0.78秒，全部通过**；容器不挂载 NPU，editable 导入路径已核验。规范与验收条件两轴增量复审通过，修改文件 pylint 前后均零告警（本机缺 Torch，排除 import-error/no-member）。

更新后仍只有一个提交：`630d54358f94e34dc8e7f22c6a57d4043ba211f5`。修正前提交已备份；没有把本次 CPU 测试结果写成 NPU 或200步精度验收通过。最新基点集成的82项测试、120项子测试通过是多修复组合证据，不全部归因于本 PR。

[独立 CPU 原始日志](https://github.com/xingleixu123/hyper-parallel/blob/a55e7b2a56742d91b2c07a354908897ceba52a58/docs/validation/jt-dev-8npu-20261014/reports/cpu-fixtures/pr200-cpu-fixture-v2-ut.log)、[无 NPU 容器凭据](https://github.com/xingleixu123/hyper-parallel/blob/a55e7b2a56742d91b2c07a354908897ceba52a58/docs/validation/jt-dev-8npu-20261014/reports/cpu-fixtures/pr200-cpu-fixture-v2.receipt.json)、[修正前后与最小复现汇总](https://github.com/xingleixu123/hyper-parallel/blob/a55e7b2a56742d91b2c07a354908897ceba52a58/docs/validation/jt-dev-8npu-20261014/reports/cpu-fixtures/cpu-fixture-validation.json)、[独立提交重写与备份凭据](https://github.com/xingleixu123/hyper-parallel/blob/a55e7b2a56742d91b2c07a354908897ceba52a58/docs/validation/jt-dev-8npu-20261014/reports/cpu-fixtures/pr200-cpu-fixture-rewrite.json)。

独立测试第一次因入口引用了集成树专有文件而未收集用例，已单独保留失败入口记录；上述通过结果来自修正入口后的第二次运行。


## 18602c4历史配方：跨并行200步补充（2026-10-16，UTC+8）

固定JT_dev f637ef7及集成源18602c4同时使用FP32专家权重梯度、FP32 Muon NS与其他已检视修复。当前FSDP8基线与checkpoint200恢复后的200步对独立连续400参考，各loss及梯度范数差0；off/full/selective也各完成200步对齐。但EP2、CP2、TP2完整200步均仍未达到全部分量0.005门槛：

| 当前集成配方 | LM最大差 | 按系数还原MTP最大差 | 按系数还原aux最大差 |
|---|---:|---:|---:|
| EP2 | 0.007523060 | 0.015432835 | 0.138989417 |
| CP2 | 0.011061668 | 0.022786458 | 0.204604585 |
| TP2 | 0.021448135 | 0.024669965 | 0.155838206 |

新TP2轮完整200步×8rank、退出0且释放，输入/初态/配置完整性与独立复算均通过，精度结论仍为FAIL。上述集成结果不能单独归因于本PR，也不支持默认开启本选项或宣称跨拓扑问题已解决。配方和门槛保持不变，下一步按四臂短步对照定位局部输入梯度传播，未将候选诊断作为生产修复提交。

[完整中文记录及曲线](https://github.com/xingleixu123/hyper-parallel/blob/a043b502dcf5291c2c78052a738a7a9d87517421/docs/validation/jt-dev-8npu-20261014/reports/f637-tp2-r4-complete/README_ZH.md) · [逐步全rank严格比较](https://github.com/xingleixu123/hyper-parallel/blob/a043b502dcf5291c2c78052a738a7a9d87517421/docs/validation/jt-dev-8npu-20261014/reports/f637-tp2-r4-complete/f637-tp2-r4-strict-comparison.json) · [数值独立复算](https://github.com/xingleixu123/hyper-parallel/blob/a043b502dcf5291c2c78052a738a7a9d87517421/docs/validation/jt-dev-8npu-20261014/reports/f637-tp2-r4-complete/f637-tp2-r4-precision-independent-review.json) · [资源与释放独立审计](https://github.com/xingleixu123/hyper-parallel/blob/a043b502dcf5291c2c78052a738a7a9d87517421/docs/validation/jt-dev-8npu-20261014/reports/f637-tp2-r4-complete/f637-tp2-r4-resource-independent-review.json)。只更新验证证据，原PR单提交、草稿状态及小问题与总单#202关联均保持。

## 具体定位、触发条件与两侧正确性

固定本PR提交 `630d54358f94e34dc8e7f22c6a57d4043ba211f5`。相关位置：[hyper_parallel/components/functional/npu_grouped_swiglu.py::_accumulate_fp32_weight_gradient:115](https://github.com/hf1s/hyper-parallel/blob/630d54358f94e34dc8e7f22c6a57d4043ba211f5/hyper_parallel/components/functional/npu_grouped_swiglu.py#L115)；[hyper_parallel/core/fully_shard/hsdp_param.py::_accumulate_fused_grad_if_needed:837](https://github.com/hf1s/hyper-parallel/blob/630d54358f94e34dc8e7f22c6a57d4043ba211f5/hyper_parallel/core/fully_shard/hsdp_param.py#L837)；[hyper_parallel/components/functional/npu_grouped_swiglu.py::backward:59](https://github.com/hf1s/hyper-parallel/blob/630d54358f94e34dc8e7f22c6a57d4043ba211f5/hyper_parallel/components/functional/npu_grouped_swiglu.py#L59)。

触发条件：低精度 grouped SwiGLU dW 在 FSDP 的 FP32 归约之前已经舍入；新增默认关闭的 expert_fp32_weight_gradients 选择，不能把默认 BF16 的舍入直接等同为算法 bug。

正确性参考：CPU FP64 算子参考约束局部梯度；正常 autograd 贡献必须保留，forward/dX 配方保持；验证显式原始 Parameter owner 与生命周期。

修复侧或并行侧核验：CPU 87 passed、25 subtests，旧实现 6 fail/3 pass；真实 2 卡 ST 1 passed，2 轮×3 微批、off/非重入重算、空专家/全空/共享融合+普通梯度，FP64 参考 rtol=1e-6、atol=2e-6，重算差0。

覆盖边界：生产策略合用的历史 FSDP 200/续训/重算可自洽；EP2/CP2/TP2 200 步仍失败。早期不同布局诊断 wrapper 不是生产 PR 验收。

### 最新官方实现对照与适用范围

核对时点为2026-10-16，固定MF官方GitCode master `1c9258ca845e21ddd66fde83e262d5635359bd30`、Megatron官方main `3e2807ce5e787f81b7275be2383092fe0c3bfc71`；MF GitHub镜像当时仍为旧`5089da3`。

需分别核对local GEMM产出、microbatch累积、collective输出和main_grad接管的dtype与完成时点。MF的Linear在compute_dtype执行matmul，后续FSDP的FP32归约设置不能证明局部dW已在FP32产出；Megatron的融合dW则按main_grad.dtype走条件分支。其最新GTP专用FP32累加路径还要求归约输出保持FP32直到main_grad.add_，且仅在flag开启、group size>2、非symmetric pool时启用。依据：[MF Linear](https://gitcode.com/mindspore/mindformers/blob/1c9258ca845e21ddd66fde83e262d5635359bd30/mindformers/pynative/layers/linear.py#L105)、[MF FSDP精度配置](https://gitcode.com/mindspore/mindformers/blob/1c9258ca845e21ddd66fde83e262d5635359bd30/mindformers/pynative/base_models/gpt/parallelize.py#L1405)、[Megatron局部dW](https://github.com/NVIDIA/Megatron-LM/blob/3e2807ce5e787f81b7275be2383092fe0c3bfc71/megatron/core/tensor_parallel/layers.py#L738)、[Megatron GTP归约输出](https://github.com/NVIDIA/Megatron-LM/blob/3e2807ce5e787f81b7275be2383092fe0c3bfc71/megatron/core/tensor_parallel/generalized_tensor_parallelism.py#L2084)。这些提供检查阶段，不能证明JT存在GTP同一缺陷，也不能把默认BF16舍入称作错误。本PR仍为默认关闭的策略；普通Linear FP32 dW的另一消融已在159步共有窗口失败，不归入本PR成功证据。

## 当前整体验收边界

本问题的定向回归与上文按版本标注的历史集成证据分别保留。最新训练源544526f7目前只有201的单次FSDP8完整200步及checkpoint CPU读回，共享机同源参考、跨机精度、最新续训和重算/并行矩阵尚未验收。当前先完成全200步、8rank主LM loss最大绝对差≤0.005；主loss全矩阵通过后先汇报，由用户决定是否继续MTP/aux专项。最新进度、曲线和资源约束见总问题单[#202](https://github.com/hf1s/hyper-parallel/issues/202)。本次仅补充说明，代码提交未改变，现有CI空结果不构成通过。
