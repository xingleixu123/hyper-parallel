总问题单：[#202](https://github.com/hf1s/hyper-parallel/issues/202)。本单只跟踪一个独立问题；对应 PR 和验证边界见下文。

## 检查清单

- [x] 已检索 hf1s/hyper-parallel 的全部现有 issue，并核对相关问题的范围。
- [x] 已阅读本问题涉及的源码及回归接口。
- [x] 已保留可执行复现/回归用例和实际结果。

## 问题描述与影响

优先级：本轮 Top2。不同微批监督token数不相等时，Trainer逐微批用局部分母计算梯度，再作微批平均，结果不等于整个optimizer step的有效token平均。真实Trainer路径的最小构造中，两个微批分别1/3个有效token，旧结果loss=5、gradient=10，整步参考应为loss=7、gradient=14。改变DP/TP/CP会改变累积微批数，即使全局样本一致也会改变训练目标。

这与#180修复的 `(loss*N)/N` 浮点非恒等性不同：本单是不同有效token数下的目标权重错误。

## 预期行为与验收标准

每个optimizer step提前确定完整监督分母，各微批共用整步分母；LM/MTP及CP各自的有效目标计数必须正确。零有效token、显式loss mask、迭代器提前耗尽及兼容callable provider均有确定行为。累积与不切微批的loss/梯度应在定向浮点容差内相同。

## 复现步骤、证据与修复边界

回归入口（PR分支含完整最小夹具）：

```bash
pytest -q tests/ut/trainer/test_text_trainer.py -k accumulation_matches_whole_step_token_mean
pytest -q tests/ut/data/test_get_batch.py
```

独立补丁CPU定向套件9 passed、15 subtests；覆盖真实Trainer、空监督、lazy输入构造与StopIteration。8卡集成TP2×EP2的200步LM最大偏差从0.922379971降至0.012481689，MTP从0.290920973降至0.004909277；剩余LM仍失败，另单跟踪。FSDP基线/续训与连续参考仍零误差。NPU数据属于多个独立修复的集成结果，不能替代本补丁的独立CPU回归。

对应commit `bad2550675ff5e32ca6aba465aacbaa28af0480c`。

独立 PR：[#194](https://github.com/hf1s/hyper-parallel/pull/194)。每个 PR 一个 commit；本 issue 不使用自动关闭关键字。

[完整记录及按影响排序的报告](https://github.com/xingleixu123/hyper-parallel/blob/66464cdd569d864562578eb45e9aa234d81770f4/docs/validation/jt-dev-8npu-20261014/REPORT_ZH.md)；[配置、曲线与原始标量索引](https://github.com/xingleixu123/hyper-parallel/blob/66464cdd569d864562578eb45e9aa234d81770f4/docs/validation/jt-dev-8npu-20261014/README.md)。

实验已获准在 125/129 空闲时继续，每轮设备释放后至少间隔 20 分钟，不停止其他人的实验。问题单保持开启；草稿 PR 不等于已合入或完整 CI 通过。

## 环境信息

测试基点为 `hf1s/hyper-parallel:JT_dev@ad1c55379767d63ece7e084b0623340b37b83d93`；八卡集成包含 PR #180 head `5de3cceab094bbed18915d1a9c9231ca220629ca`。后续六项修复集成源为 `ec2e16bce26e497519c573f0a8d87a538a329197`，各 PR 的独立 diff 不包含其他修复。

硬件 8×Ascend 910B3 64GiB；CANN 9.2.0-beta.2 / V100R001C12B086；Python 3.12.14，torch 2.12.0+cpu，torch_npu 2.12.0，transformers 5.13.1。训练配置为两层主干＋一层MTP、8个路由专家，seq4096、GBS8、MBS1、seed42、常数LR0.0001。性能结论只适用于该已测配置。

## 具体定位、触发条件与两侧正确性

固定本PR提交 `bad2550675ff5e32ca6aba465aacbaa28af0480c`。相关位置：[hyper_parallel/trainer/text_trainer.py::train_step:245](https://github.com/hf1s/hyper-parallel/blob/bad2550675ff5e32ca6aba465aacbaa28af0480c/hyper_parallel/trainer/text_trainer.py#L245)；[hyper_parallel/data/batching/get_batch.py::prepare_step:268](https://github.com/hf1s/hyper-parallel/blob/bad2550675ff5e32ca6aba465aacbaa28af0480c/hyper_parallel/data/batching/get_batch.py#L268)。

触发条件：同一 optimizer step 的微批有效监督数不同（含空目标或 mask）；旧实现逐微批覆盖 step_token_counts。

正确性参考：整批有效 token 的 sum(loss)/sum(valid) 是独立目标；已保存真实 Trainer 反例的正确 loss/grad 为 7/14，旧值为 5/10。

修复侧或并行侧核验：真实 train_step 与 reducer 对整批和不同微批切分比较 loss/参数梯度；包含不均匀、空监督、mask。当前定向 UT 9 passed、15 subtests。

覆盖边界：旧集成跨拓扑误差降低不构成此单 PR 的 200 步因果证明；18602c4 的 FSDP 续训/重算 200 步零差只属于固定多修复配方。

### 最新官方实现对照与适用范围

核对时点为2026-10-16，固定MF官方GitCode master `1c9258ca845e21ddd66fde83e262d5635359bd30`、Megatron官方main `3e2807ce5e787f81b7275be2383092fe0c3bfc71`；MF GitHub镜像当时仍为旧`5089da3`。

最新MF官方master的per-token路径先反传原始分子，累积完成后按全局token计数缩放梯度；其loss sense已经包含1/(DP×TP×CP)，因此最终除数采用每rank平均计数。Megatron相应路径将DDP scaling设为1，再除全局token总数。两侧应重建相同的sum(loss)/sum(valid)目标，并核对SUM/AVG、TP补偿及每个MTP深度的有效分母，不能只照搬某个倍数。依据：[MF反传入口](https://gitcode.com/mindspore/mindformers/blob/1c9258ca845e21ddd66fde83e262d5635359bd30/mindformers/pynative/trainer/trainer.py#L1998)、[MF最终梯度归一化](https://gitcode.com/mindspore/mindformers/blob/1c9258ca845e21ddd66fde83e262d5635359bd30/mindformers/pynative/trainer/trainer.py#L2084)、[Megatron DDP缩放设置](https://github.com/NVIDIA/Megatron-LM/blob/3e2807ce5e787f81b7275be2383092fe0c3bfc71/megatron/core/distributed/distributed_data_parallel.py#L250)、[Megatron最终token缩放](https://github.com/NVIDIA/Megatron-LM/blob/3e2807ce5e787f81b7275be2383092fe0c3bfc71/megatron/core/distributed/finalize_model_grads.py#L741)。本次是源码语义核对，未执行跨框架训练对照；主LM的局部回归不自动覆盖每种MTP/aux定义。

## 当前整体验收边界

本问题的定向回归与上文按版本标注的历史集成证据分别保留。最新训练源544526f7目前只有201的单次FSDP8完整200步及checkpoint CPU读回，共享机同源参考、跨机精度、最新续训和重算/并行矩阵尚未验收。当前先完成全200步、8rank主LM loss最大绝对差≤0.005；主loss全矩阵通过后先汇报，由用户决定是否继续MTP/aux专项。最新进度、曲线和资源约束见总问题单[#202](https://github.com/hf1s/hyper-parallel/issues/202)。本次仅补充说明，代码提交未改变，现有CI空结果不构成通过。
