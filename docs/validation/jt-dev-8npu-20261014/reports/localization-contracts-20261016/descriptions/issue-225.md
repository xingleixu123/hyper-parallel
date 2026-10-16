## 问题概述

总跟踪与问题排序：#202。本问题单独跟踪 Trainer 的资源统计归约域，不涉及训练 loss 或梯度计算。

`EnvironMeterCallback` 将 `performance/step_time`、`memory/device_max_allocated_gb`、`memory/device_max_reserved_gb` 与样本、token、loss 一样，在当前 DP+CP 进程组内归约。TP 或 PP 大于 1 时，该组不覆盖整个训练任务。最慢或显存峰值最大的 rank 如果位于其他 TP lane / PP stage，rank0 日志就可能漏报；由步耗时计算的 tokens/s 也会偏高。仅在 DP+CP 已覆盖整个 world 时，这个问题不会显现。

### 实际观测

固定集成树 `18602c4524dc664b81b2bad9a7b624455ce1494f` 的八卡 DP4×TP2、CP1、PP1 训练保留了 69 个完整步骤、每步 8 份 rank 日志。26/69 步的 rank0 耗时小于同一步全部 rank 的最大值。第 2 步：rank0 为 `2.86460018157959 s`，全 rank 最大值位于 rank1，为 `2.8686962127685547 s`，少报 `0.004096031188964844 s`。

该训练因另一任务开始使用设备而主动释放，未完成 200 步。以上是资源统计口径的实际反例，不是 200 步精度或最终性能收益结论。下方显存数值属于可控单测数据，不是实测显存收益。

## 最小复现与实际结果

新增回归 `tests/ut/trainer/test_environ_meter_callback.py::TestEnvironMeterCallback::test_tensor_parallel_resource_extrema_cover_all_ranks` 构造 DP2×CP2×TP2 的 8-rank 资源输入；rank0 所在 DP+CP 组为 `(0,2,4,6)`，将资源峰值放在另一 TP lane。设备和 collective 边界用 mock 替代，执行真实回调逻辑，无需 NPU。

| 指标 | 八个 rank 的受控输入 | 修复前 rank0 输出 | 期望输出 |
| --- | --- | ---: | ---: |
| 步耗时（秒） | 2、8、3、4、5、6、7、9 | 7 | 9 |
| 峰值 allocated（GiB） | 1、13、2、3、4、5、6、7 | 6 | 13 |
| 峰值 reserved（GiB） | 2、17、3、4、5、6、7、8 | 7 | 17 |
| tokens/s | 整步有效 token 数为 24 | 24/7 | 24/9 |

同时要求 token 数仍为 24、样本数在去除 CP 副本后仍为 4、LM loss 仍为 1；这些值不能因扩大资源归约域而被 TP 副本重复统计。

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 HYPER_PARALLEL_TEST_DEVICE_TYPE=cpu \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -p no:cacheprovider -q \
  tests/ut/trainer/test_environ_meter_callback.py
```

将同一份新增测试放在未修复的固定基点上，结果为 `4 failed, 5 passed, 3 subtests passed`。四个失败是上述耗时、吞吐、allocated、reserved 子测试；这不是进程崩溃或异常栈，而是错误数值被断言捕获。应用修复后结果为 `5 passed, 7 subtests passed`，退出 0。

## 根因与修复边界

- `_metric_group()` 返回 `mesh.dp_cp_mesh.get_group()`；原先 `_reduce()` 的全部调用都使用此组。
- 三项资源极值应在整个 world 上执行 MAX，覆盖所有 TP/PP rank；样本、token、loss 和梯度范数仍使用原 DP+CP 归约。
- 单进程时直接返回本地值，不要求初始化分布式进程组。
- world collective 要求全部 rank 按相同顺序调用回调；已检视三个 Trainer 入口的回调调用路径，没有 rank0 专属调用。本修复不新增 collective 次数，只改变这三次 MAX 的参与范围。
- 耗时仍是现有 callback 计时范围，显存仍是设备累计峰值；本项不会把它们改造成端到端墙钟耗时或单步增量显存，也不会修复 TP2 的 loss 偏差。

## 环境与版本

- 独立复现、修复基点：`hf1s/hyper-parallel:JT_dev@f637ef7ee4d4b416800716f1ebaaae7709b8a06a`，包含 #180；提交前已再次核实远端分支。
- 实测日志来自上述基点上的任务集成树 `18602c4`；资源回调在两树中相同。
- 129：Ascend 910B3，镜像 `jt-clean:hp-910b-20261009`，CANN `9.2.0-beta.2`，Python `3.12.14`、Torch/TorchNPU `2.12.0`。
- 红绿回归在同镜像的无 NPU 挂载容器执行，pytest `9.1.1`，4 CPU / 8 GiB 限额；独立 editable 安装已核对导入路径。未执行修复后的真实 NPU/PP 分布式回归。

## 验收标准与当前状态

- [x] 其他 TP lane 上的耗时、allocated、reserved 峰值可被检测到。
- [x] tokens/s 使用全 rank 最大耗时，数据计数与 loss 不被 TP 副本放大。
- [x] 单进程保留本地统计，不执行 collective。
- [x] 同一回归在未修复基点失败、补丁后通过；规范与需求两轴独立检视通过。
- [ ] 维护者确认并合入；后续真实多卡回归与当前 head CI 以实际结果补充。

本问题保持开启；完整 200 步精度、并行收益和重计算验收继续由 #202 跟踪。


## 独立提交与公开证据

对应小问题[#225](https://github.com/hf1s/hyper-parallel/issues/225)，独立单提交草稿PR[#226](https://github.com/hf1s/hyper-parallel/pull/226)，提交`b58fed7fcdce38a8a62c5fb5bae845247fcbbeb7`，两文件。总跟踪[#202](https://github.com/hf1s/hyper-parallel/issues/202)统一记录问题排序及完整训练验收。

[修复前原始日志](https://github.com/xingleixu123/hyper-parallel/blob/0163eb6b4c826f25568a72594693124427d4ecf8/docs/validation/jt-dev-8npu-20261014/reports/global-resource-metrics/global-resource-metrics-red.log) · [修复后原始日志](https://github.com/xingleixu123/hyper-parallel/blob/0163eb6b4c826f25568a72594693124427d4ecf8/docs/validation/jt-dev-8npu-20261014/reports/global-resource-metrics/global-resource-metrics-green.log) · [两轴检视与最终源码指纹](https://github.com/xingleixu123/hyper-parallel/blob/0163eb6b4c826f25568a72594693124427d4ecf8/docs/validation/jt-dev-8npu-20261014/reports/global-resource-metrics/global-resource-metrics-two-axis-review.json) · [八份CPU原始文件的大小/SHA](https://github.com/xingleixu123/hyper-parallel/blob/0163eb6b4c826f25568a72594693124427d4ecf8/docs/validation/jt-dev-8npu-20261014/reports/global-resource-metrics/global-resource-metrics-test-evidence.json) · [69步资源观测](https://github.com/xingleixu123/hyper-parallel/blob/0163eb6b4c826f25568a72594693124427d4ecf8/docs/validation/jt-dev-8npu-20261014/reports/global-resource-metrics/global-resource-metrics-real-observation.json)。

补充逐步复算：本轮3/69步rank0 reserved低于全rank最大值，最大少报0.0390625 GiB；allocated虽有lane间差异，但rank0始终位于较大一侧，不能说该次allocated也少报。当前提交没有check run，combined status为pending，尚不能宣称CI通过。CPU回归后只完善docstring、注释及断言失败消息，未再改功能逻辑；最终文件指纹与原始测试记录分别保存。


## 新增真实八进程CPU/Gloo红绿回归（2026-10-16，UTC+8）

在精确基点`f637ef7`及本PR单提交`b58fed7`的独立editable源码树上分别运行8个CPU进程。使用真实Gloo、真实进程组和实际运行时all_reduce，仅时钟与显存读数是人工哨兵。覆盖DP2×CP2×TP2、DP2×CP2×PP2对应的组边界，以及DP8 world控制；每组两步，两侧合计96条观测、1152项指标比较。

| 最终v2判定 | 修复前 | 修复后 |
|---|---:|---:|
| 相对正确全局极值的资源差异 | 64项（两种子组边界各32） | 0 |
| 相对各版本独立oracle的非预期差异 | 0：逐项符合旧代码组内MAX | 0：逐项符合world MAX |
| 数据、loss、累计计数、world控制偏差 | 0 | 0 |

独立两轴复算通过，58份原始证据大小/SHA、两个源码归档各1946个Git blob以及最终探针/runner指纹均核对。v1初次观察保留；独立检视要求加强base oracle及失败退出码落盘，v2采纳后重新运行，最终验收使用v2。两次CPU容器均exit0、无NPU映射、各worker未初始化NPU，结束后容器移除。临时展开源、独立venv和重复归档已清理，全部原始证据保留。

这是CPU/Gloo归约边界验证，不是TP/PP模型训练、NPU/HCCL回归或200步训练；人工秒数/GiB不作为真实性能或显存收益。HCCL成本、真实PP调度及CI仍需实际验证。只补充证据，原PR保持同一个提交与草稿状态。

[详细中文说明及完整复现入口](https://github.com/xingleixu123/hyper-parallel/blob/57a3af3fa15413d6c321a866d4fe9ed5354e9d18/docs/validation/jt-dev-8npu-20261014/reports/pr226-gloo-8cpu/README_ZH.md) · [需求轴独立复算](https://github.com/xingleixu123/hyper-parallel/blob/57a3af3fa15413d6c321a866d4fe9ed5354e9d18/docs/validation/jt-dev-8npu-20261014/reports/pr226-gloo-8cpu/pr226-gloo-v2-spec-independent-review.json) · [规范与证据边界审计](https://github.com/xingleixu123/hyper-parallel/blob/57a3af3fa15413d6c321a866d4fe9ed5354e9d18/docs/validation/jt-dev-8npu-20261014/reports/pr226-gloo-8cpu/pr226-gloo-v2-standards-independent-review.json) · [全部原始文件清单](https://github.com/xingleixu123/hyper-parallel/blob/57a3af3fa15413d6c321a866d4fe9ed5354e9d18/docs/validation/jt-dev-8npu-20261014/reports/pr226-gloo-8cpu/pr226-gloo-evidence-manifest.json)。


## 补充完整200步的修复前NPU观测（2026-10-16，UTC+8）

当前固定集成源18602c4中的本回调仍为修复前实现。TP2第四轮独立完成200步×8rank，退出0且正常释放，没有资源冲突；rank0在51/200步（去前20步后的窗口内46步）低于另一TP lane的步时，最大少报1.213431358毫秒于step5。该轮allocated/reserved没有rank0峰值漏报，不能把旧69步轮次的reserved偏差搬到本轮。

逐步离线全八rankMAX的中位步时为1.857553244秒、21–200步总耗时333.968660831秒；按监督token3325552计一次得到9957.676842 token/s。该完整NPU观测进一步确认修复前统计域问题，但PR未加入本轮训练源，仍不能称为修复后HCCL成本或PP训练验收。TP2数值精度FAIL（LM/还原MTP/aux最大差0.0214481354/0.0246699651/0.1558382064），资源证据审计PASS与训练通过分别陈述。

[完整中文记录及曲线](https://github.com/xingleixu123/hyper-parallel/blob/a043b502dcf5291c2c78052a738a7a9d87517421/docs/validation/jt-dev-8npu-20261014/reports/f637-tp2-r4-complete/README_ZH.md) · [逐步全rank严格比较](https://github.com/xingleixu123/hyper-parallel/blob/a043b502dcf5291c2c78052a738a7a9d87517421/docs/validation/jt-dev-8npu-20261014/reports/f637-tp2-r4-complete/f637-tp2-r4-strict-comparison.json) · [数值独立复算](https://github.com/xingleixu123/hyper-parallel/blob/a043b502dcf5291c2c78052a738a7a9d87517421/docs/validation/jt-dev-8npu-20261014/reports/f637-tp2-r4-complete/f637-tp2-r4-precision-independent-review.json) · [资源与释放独立审计](https://github.com/xingleixu123/hyper-parallel/blob/a043b502dcf5291c2c78052a738a7a9d87517421/docs/validation/jt-dev-8npu-20261014/reports/f637-tp2-r4-complete/f637-tp2-r4-resource-independent-review.json)。只更新验证证据，原PR单提交、草稿状态及小问题与总单#202关联均保持。

## 具体定位、触发条件与两侧正确性

固定本PR提交 `b58fed7fcdce38a8a62c5fb5bae845247fcbbeb7`。相关位置：[hyper_parallel/trainer/callbacks/environ_meter_callback.py::_reduce:158](https://github.com/hf1s/hyper-parallel/blob/b58fed7fcdce38a8a62c5fb5bae845247fcbbeb7/hyper_parallel/trainer/callbacks/environ_meter_callback.py#L158)；[hyper_parallel/trainer/callbacks/environ_meter_callback.py::on_step_end:296](https://github.com/hf1s/hyper-parallel/blob/b58fed7fcdce38a8a62c5fb5bae845247fcbbeb7/hyper_parallel/trainer/callbacks/environ_meter_callback.py#L296)；[hyper_parallel/trainer/callbacks/environ_meter_callback.py::_memory_metrics:220](https://github.com/hf1s/hyper-parallel/blob/b58fed7fcdce38a8a62c5fb5bae845247fcbbeb7/hyper_parallel/trainer/callbacks/environ_meter_callback.py#L220)。

触发条件：资源仅在 DP+CP 子组 MAX，最慢或最大显存 rank 位于另 TP lane/PP stage 时漏报；数值 loss/token 不能改成 world sum。

正确性参考：人工时钟/显存哨兵的独立 world MAX 与原子组 MAX oracle；tokens/loss 按原 DP+CP 归约，不重复计副本。

修复侧或并行侧核验：UT 红4 fail、绿5 passed/7 subtests；真实8进程 CPU/Gloo v2 两侧96条观测/1152比较，资源64错→0、非资源0；真实旧TP2 69步日志证明部分漏报。

覆盖边界：真实 collective 但秒数/GiB 是哨兵；不是TP/PP模型训练、NPU/HCCL成本或200步收益。

### 参考适用范围

本项的正确性由上述所有权、设备/API、布局、累计状态或统计契约及其独立回归证明；没有强行套用MF/Megatron的不同实现，也未用该修复解释尚未定位的TP/CP/EP长步漂移。数值、重算和Muon的最新官方源码对照统一关联总问题单[#202](https://github.com/hf1s/hyper-parallel/issues/202)。

## 当前整体验收边界

本问题的定向回归与上文按版本标注的历史集成证据分别保留。最新训练源544526f7目前只有201的单次FSDP8完整200步及checkpoint CPU读回，共享机同源参考、跨机精度、最新续训和重算/并行矩阵尚未验收。当前先完成全200步、8rank主LM loss最大绝对差≤0.005；主loss全矩阵通过后先汇报，由用户决定是否继续MTP/aux专项。最新进度、曲线和资源约束见总问题单[#202](https://github.com/hf1s/hyper-parallel/issues/202)。本次仅补充说明，代码提交未改变，现有CI空结果不构成通过。
