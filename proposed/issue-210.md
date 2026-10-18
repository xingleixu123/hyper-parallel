<!-- correctness-review-20261018 -->
## 2026-10-18 正确性复核：NS 精度选项：可配置配方，不等于 BF16 默认错误

**当前判断**：本单/PR #201 是显式、可持久化的 NS 精度能力提案；BF16 默认和 FP32 选项属于不同数值策略。不能由另一框架采用 FP32 预归一化就认定默认 BF16 算法错误，也不能把“更像旧 MF”作为改变默认的理由。

**最新参考**：[MF master PyNative NS](https://gitcode.com/mindspore/mindformers/blob/409019aac0c6579bb4fb8a9b04a75c259d98a96e/mindformers/pynative/optimizer/muon.py#L1195) 使用 FP32 norm/clamp 后进入 BF16 迭代；Megatron 当前委托 Emerging Optimizers，开发源声明 v0.3.0，该版本要求 FP32 输入进行归一化，后续 matmul 精度受策略控制。入口/预归一化、NS 迭代、relay、主参数写回必须分开，不将这些概念合并成单一“FP32 Muon”。

**验收重点**：保留本单已有默认兼容、参数组持久化、分布式通信和 checkpoint 规则；以各自同配方的独立实例/参考更新比较。通用选项的价值来自明确的用户选择和精度/成本收益，不来自 MF 距离。

**范围保持**：下方是其它八卡工作流的历史验证；本次仅作关联正确性说明，没有重新执行该工作流、没有修改其代码、PR 状态或实验安排，不把单卡旧归档复现冒记为跨拓扑问题已解决。


本次固定参考：MF GitCode master `409019aac0c6579bb4fb8a9b04a75c259d98a96e`（只核查 `mindformers/pynative/`，未用旧 training_graph 冒充），Megatron main `1effc75a46b26c6b93478355ff9fc3bb8ac1fafc`，当前 JT_dev `a1e6c222eab097c9609d6e744a3c5f98aeed6bdc`。源码路径/行号/SHA-256 见[完整正确性复核](https://github.com/xingleixu123/hyper-parallel/blob/xxl-dev/jt-correctness-review-20261018/AUDIT.md)与[来源清单](https://github.com/xingleixu123/hyper-parallel/blob/xxl-dev/jt-correctness-review-20261018/source-catalog.json)。本轮不新增训练结果，不以 MF loss 距离代替独立正确性判据。


<details><summary>原问题与历史实验记录（适用旧基线；当前判断以上方复核为准）</summary>

总问题单：[#202](https://github.com/hf1s/hyper-parallel/issues/202)。本单只跟踪一个独立问题；对应 PR 和验证边界见下文。

## 🚀 功能描述

优先级：本轮 Top9。Muon现有NS输入与update relay固定为BF16；FP32主参数和FP32梯度归约不能选择完整FP32 NS路径，也没有明确可恢复的optimizer精度配方。请求提供默认BF16、可选FP32的参数组配置；不改变NS系数/轮数、LR、momentum存储或HF32后端设置。

已检索全部现有issue：#149处理MF/HP矩阵拆分与缩放舍入语义，#50为旧的一次FP32实验。本单针对通用可配置API及分布式checkpoint恢复契约，不声称实现MF精确算术或解决全部跨并行漂移。

## 现有替代方案

外部monkeypatch只能做诊断，难以可靠覆盖空分片/缺失梯度rank、通信缓冲dtype和DCP参数组恢复。单独改变NS函数内部dtype也不能覆盖前后的BF16边界。保持BF16默认仍是兼容方案；FP32应显式启用并重建同配方FSDP参考。

## 可靠性与可恢复性要求

验证矩阵包含BF16默认一致性、FP32通信/更新、无梯度rank、空分片、共享参数组和checkpoint兼容。旧checkpoint缺字段恢复历史BF16；weights-only保留新配置；非法Muon精度在任何optimizer叶子写入前报错。batch软上限按实际dtype字节数计算，但不会切分单矩阵，不是严格内存上限。

## 提议的新API

`Muon(..., ns_dtype="bfloat16" | "float32")`，也支持各参数组。内置momentum输出、NS计算、relay缓冲保持所选类型；自定义NS回调仍负责自己的数学精度。自定义momentum返回值隔离存储，避免NS原地修改活跃梯度/state。

## 是否影响现有API

新增可选字段默认保持BF16行为，随optimizer参数组持久化。DCP加载模板支持chained/mixed-precision optimizer及平铺/嵌套参数组。现有API调用不必改动；新接口仍等待维护者评审。

## 补充信息

独立单commit 草稿 PR [#201](https://github.com/hf1s/hyper-parallel/pull/201)，commit `59a03c63490f2e47d0fd89d9e7974982abe1e9e2`。

```bash
pytest -q tests/ut/core/optimizer/test_muon_ns_precision.py tests/ut/trainer/test_muon_precision_checkpoint.py
pytest -q tests/torch/optimizer/test_muon_ns_precision.py
```

CPU44 passed、52 subtests；真实NPU ST2 passed、53.47秒：8卡全分片/2×4混合各10次更新与独立实例的同实现未分片Muon比较，另有2卡缺失梯度/空分片各3更新、本地分片零误差。首次ST遇到DTensor.full_tensor不均匀重建形状异常，保留原始失败并另行建单；本PR没有修复该重建问题。

验收包括上述语义、旧checkpoint兼容及默认配方不变；不将3步集成通过作为200步目标完成。此前125的#200+#201生产配方FSDP0–200完成完整性自检；随后18602c4历史配方基线/续训已与独立连续400参考对齐，TP/CP/EP主LM仍失败，详见下方分段。相同build观测下两选项合用步时+24.25%，并非NS孤立成本；原“6.8倍NS开销”混入runtime观测，已撤回。[同口径成本证据](https://github.com/xingleixu123/hyper-parallel/blob/66464cdd569d864562578eb45e9aa234d81770f4/docs/validation/jt-dev-8npu-20261014/reports/matching-build-topology-costs.md)。

测试基点为 `hf1s/hyper-parallel:JT_dev@ad1c55379767d63ece7e084b0623340b37b83d93`；八卡集成包含 PR #180 head `5de3cceab094bbed18915d1a9c9231ca220629ca`。后续六项修复集成源为 `ec2e16bce26e497519c573f0a8d87a538a329197`，各 PR 的独立 diff 不包含其他修复。

硬件 8×Ascend 910B3 64GiB；CANN 9.2.0-beta.2 / V100R001C12B086；Python 3.12.14，torch 2.12.0+cpu，torch_npu 2.12.0，transformers 5.13.1。训练配置为两层主干＋一层MTP、8个路由专家，seq4096、GBS8、MBS1、seed42、常数LR0.0001。性能结论只适用于该已测配置。

实验已获准在 125/129 空闲时继续，每轮设备释放后至少间隔 20 分钟，不停止其他人的实验。问题单保持开启；草稿 PR 不等于已合入或完整 CI 通过。


## 18602c4历史元数据与续训证据

基点JT_dev `f637ef7`，含#180及已检视修复的固定集成源码为`18602c4524dc664b81b2bad9a7b624455ce1494f`，CANN 9.2.0-beta.2。先完成8卡FSDP 0–200，再从该checkpoint200恢复；本次8rank各完整记录201–400共200步，退出码0，约376.91秒后释放全部设备。恢复前只读CPU检查发现28项Muon NS dtype均持久化为float32、46个优化器标量步数均为200；实际恢复从global_step200继续到400。**本轮运行配置与checkpoint同为float32，只说明该配方完成保存、恢复和训练；配置冲突时的覆盖优先级仍依据已有定向测试。随后已补齐独立连续400参考，完整比较结果见下方18602c4历史配方段。**多项修复共同参与本次集成，不能将整轮结果全部归因于此单一PR；原PR代码提交未改变，继续保持单提交草稿状态。

[完整8rank恢复证据](https://github.com/xingleixu123/hyper-parallel/blob/f39bb5661e3e63b3f1b0df1c53902285db76ce43/docs/validation/jt-dev-8npu-20261014/reports/f637-fsdp8-resume-r2-complete/f637-fsdp8-resume-r2-complete-analysis.json) · [实际恢复日志](https://github.com/xingleixu123/hyper-parallel/blob/f39bb5661e3e63b3f1b0df1c53902285db76ce43/docs/validation/jt-dev-8npu-20261014/reports/f637-fsdp8-resume-r2-complete/rank0-stdout.log) · [checkpoint元数据](https://github.com/xingleixu123/hyper-parallel/blob/f39bb5661e3e63b3f1b0df1c53902285db76ce43/docs/validation/jt-dev-8npu-20261014/reports/f637-checkpoint-metadata/f637-checkpoint-metadata-cpu-summary.json) · [中文总报告及边界说明](https://github.com/xingleixu123/hyper-parallel/blob/f39bb5661e3e63b3f1b0df1c53902285db76ce43/docs/validation/jt-dev-8npu-20261014/REPORT_ZH.md)。总问题单仍为[#202](https://github.com/hf1s/hyper-parallel/issues/202)。

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

固定本PR提交 `59a03c63490f2e47d0fd89d9e7974982abe1e9e2`。相关位置：[hyper_parallel/core/optimizer/muon.py::_update_muon_momentum:550](https://github.com/hf1s/hyper-parallel/blob/59a03c63490f2e47d0fd89d9e7974982abe1e9e2/hyper_parallel/core/optimizer/muon.py#L550)；[hyper_parallel/core/optimizer/muon.py::_fused_broadcast_and_apply:1127](https://github.com/hf1s/hyper-parallel/blob/59a03c63490f2e47d0fd89d9e7974982abe1e9e2/hyper_parallel/core/optimizer/muon.py#L1127)；[hyper_parallel/core/optimizer/muon.py::load_state_dict:307](https://github.com/hf1s/hyper-parallel/blob/59a03c63490f2e47d0fd89d9e7974982abe1e9e2/hyper_parallel/core/optimizer/muon.py#L307)；[hyper_parallel/core/optimizer/muon.py::_gather_and_compute_shard_updates:647](https://github.com/hf1s/hyper-parallel/blob/59a03c63490f2e47d0fd89d9e7974982abe1e9e2/hyper_parallel/core/optimizer/muon.py#L647)。

触发条件：Muon 默认将 NS 输入及 relay 缓冲转 BF16；新增默认 bfloat16、可选 float32 的参数组字段，需覆盖 momentum 输出、NS、空/无梯度 rank、relay 与恢复。

正确性参考：默认与显式 BF16 精确一致；checkpoint 指定值恢复、旧 checkpoint 缺字段回到 BF16、weights-only 保留新配置；局部 momentum 手工公式/回调别名有定向契约测试。

修复侧或并行侧核验：CPU 44 passed、52 subtests 含 12 种实际 DCP 恢复；NPU 2 ST：FSDP8/HSDP2×4 各 10 更新，2 卡无梯度/空分片各3更新。reference 和 candidate 都使用本仓库 Muon 的独立实例，一侧未分片、一侧分片。

覆盖边界：NPU ST 证明同实现分片/relay/状态等价，不是外部独立 NS 数学实现。18602c4 FP32 恢复和连续400一致仅是多修复配方；没有跨 MF/Megatron 实测或独立 NS 成本实验。

### 最新官方实现对照与适用范围

核对时点为2026-10-16，固定MF官方GitCode master `1c9258ca845e21ddd66fde83e262d5635359bd30`、Megatron官方main `3e2807ce5e787f81b7275be2383092fe0c3bfc71`；MF GitHub镜像当时仍为旧`5089da3`。

当前201实验显式选择ns_dtype=float32；MF当前Muon先FP32归一化再转BF16执行NS，使用m_next=βm+g。Megatron实际实现由pyproject固定的Emerging-Optimizers v0.3.0提供，依赖采用EMA momentum，medium模式也是FP32归一化后BF16 NS。JT使用norm+eps，参考路径采用clamp类归一化，因此需要同时固定momentum/Nesterov、epsilon、NS系数/次数/dtype、完整矩阵及QKV/专家拆分、shape scale、decay和写回舍入才能做数学对照。依据：[MF NS输入](https://gitcode.com/mindspore/mindformers/blob/1c9258ca845e21ddd66fde83e262d5635359bd30/mindformers/pynative/optimizer/muon.py#L1195)、[MF实际更新](https://gitcode.com/mindspore/mindformers/blob/1c9258ca845e21ddd66fde83e262d5635359bd30/mindformers/pynative/optimizer/muon.py#L1327)、[Megatron依赖版本](https://github.com/NVIDIA/Megatron-LM/blob/3e2807ce5e787f81b7275be2383092fe0c3bfc71/pyproject.toml#L261)、[实际依赖step](https://github.com/NVIDIA-NeMo/Emerging-Optimizers/blob/b309e2f01cda75dc96a6dc1a2355a7b3b64b5e16/emerging_optimizers/orthogonalized_optimizers/orthogonalized_optimizer.py#L157)、[实际依赖NS](https://github.com/NVIDIA-NeMo/Emerging-Optimizers/blob/b309e2f01cda75dc96a6dc1a2355a7b3b64b5e16/emerging_optimizers/orthogonalized_optimizers/muon_utils.py#L127)。已确认的是这些配方差异，未证明它们导致当前并行漂移。现有NPU ST两侧都用本仓库Muon，证明同实现的分片/relay/状态等价；不能升级为外部数学oracle。

## 当前整体验收边界

本问题的定向回归与上文按版本标注的历史集成证据分别保留。最新训练源544526f7目前只有201的单次FSDP8完整200步及checkpoint CPU读回，共享机同源参考、跨机精度、最新续训和重算/并行矩阵尚未验收。当前先完成全200步、8rank主LM loss最大绝对差≤0.005；主loss全矩阵通过后先汇报，由用户决定是否继续MTP/aux专项。最新进度、曲线和资源约束见总问题单[#202](https://github.com/hf1s/hyper-parallel/issues/202)。本次仅补充说明，代码提交未改变，现有CI空结果不构成通过。


</details>
