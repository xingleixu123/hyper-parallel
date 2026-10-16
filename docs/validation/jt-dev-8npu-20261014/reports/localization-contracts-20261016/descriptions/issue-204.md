总问题单：[#202](https://github.com/hf1s/hyper-parallel/issues/202)。本单只跟踪一个独立问题；对应 PR 和验证边界见下文。

## 检查清单

- [x] 已检索 hf1s/hyper-parallel 的全部现有 issue，并核对相关问题的范围。
- [x] 已阅读本问题涉及的源码及回归接口。
- [x] 已保留可执行复现/回归用例和实际结果。

## 问题描述与影响

优先级：本轮 Top3，实测显存收益最明确。FSDP已经clone分片并替换模块别名，但scheduler/state的 `source_shard_infos` 字典键及显式replicate参数集合仍持有完整源Parameter。原参数已经在设备上，导致不再参与训练的完整初始化存储长期驻留。

8卡200步前后对照：EP2 peak allocated从30.605194降到21.183023GiB（节省9.422171GiB）；CP2从31.183506降到21.058210GiB（节省10.125296GiB）。

## 预期行为与验收标准

HSDPParam消费初始化元数据后释放内部source容器及独立replicate集合引用；保留caller-owned容器、ignored参数、tied别名和活跃元数据，不原地resize调用者原storage。旧源Parameter弱引用应失效；同拓扑loss与梯度轨迹不得变化。

## 复现步骤、证据与修复边界

最小回归：

```bash
pytest -q tests/ut/core/fully_shard/test_hsdp_scheduler.py -k source_parameters_are_released_after_sharding
```

PR中的测试经过实际CPU HSDPParam clone/替换路径，六种weakref场景在旧实现全部失败；修复后的五文件套件112 passed、12 subtests。EP2/CP2前后各200步×8rank的LM/MTP/aux/total loss和grad norm逐值完全相同。显存差与被保留参数的理论storage大小一致；不到0.1%的步时差不宣称吞吐收益。跨拓扑对FSDP的剩余精度问题未解决。

commit `b948684ca3f5b3dd52ccbb385bf4123b8fa2c0b6`。

![显存实测](https://raw.githubusercontent.com/xingleixu123/hyper-parallel/ee310ea9e02d4d97e30037450247d8092f540805/docs/validation/jt-dev-8npu-20261014/figures/source-parameter-lifetime-zh.png)

独立 PR：[#198](https://github.com/hf1s/hyper-parallel/pull/198)。每个 PR 一个 commit；本 issue 不使用自动关闭关键字。

[完整记录及按影响排序的报告](https://github.com/xingleixu123/hyper-parallel/blob/66464cdd569d864562578eb45e9aa234d81770f4/docs/validation/jt-dev-8npu-20261014/REPORT_ZH.md)；[配置、曲线与原始标量索引](https://github.com/xingleixu123/hyper-parallel/blob/66464cdd569d864562578eb45e9aa234d81770f4/docs/validation/jt-dev-8npu-20261014/README.md)。

实验已获准在 125/129 空闲时继续，每轮设备释放后至少间隔 20 分钟，不停止其他人的实验。问题单保持开启；草稿 PR 不等于已合入或完整 CI 通过。

## 环境信息

测试基点为 `hf1s/hyper-parallel:JT_dev@ad1c55379767d63ece7e084b0623340b37b83d93`；八卡集成包含 PR #180 head `5de3cceab094bbed18915d1a9c9231ca220629ca`。后续六项修复集成源为 `ec2e16bce26e497519c573f0a8d87a538a329197`，各 PR 的独立 diff 不包含其他修复。

硬件 8×Ascend 910B3 64GiB；CANN 9.2.0-beta.2 / V100R001C12B086；Python 3.12.14，torch 2.12.0+cpu，torch_npu 2.12.0，transformers 5.13.1。训练配置为两层主干＋一层MTP、8个路由专家，seq4096、GBS8、MBS1、seed42、常数LR0.0001。性能结论只适用于该已测配置。

## 具体定位、触发条件与两侧正确性

固定本PR提交 `b948684ca3f5b3dd52ccbb385bf4123b8fa2c0b6`。相关位置：[hyper_parallel/core/fully_shard/hsdp_scheduler.py::__init__:99](https://github.com/hf1s/hyper-parallel/blob/b948684ca3f5b3dd52ccbb385bf4123b8fa2c0b6/hyper_parallel/core/fully_shard/hsdp_scheduler.py#L99)；[hyper_parallel/core/fully_shard/hsdp_state.py::__init__:61](https://github.com/hf1s/hyper-parallel/blob/b948684ca3f5b3dd52ccbb385bf4123b8fa2c0b6/hyper_parallel/core/fully_shard/hsdp_state.py#L61)。

触发条件：分片初始化已克隆并替换模块参数后，scheduler/state 的 source_shard_infos 和显式 replicated 集合仍强引用完整源参数。

正确性参考：真实 HSDPParam + weakref 检查源参数可回收；调用者仍持有、ignored/tied/活跃参数引用不能被错误清除。

修复侧或并行侧核验：红测 6 失败；修复后相关 5 文件 112 passed、12 subtests；不在共享 storage 上 resize。

覆盖边界：历史同拓扑 200 步 LM/MTP/aux/total/grad_norm 零差；EP/CP allocated 分别减少 9.422171/10.125296 GiB，属于相应固定集成版本，不能移到最新 544526f7 或宣称单 PR 完整跨拓扑精度。

### 参考适用范围

本项的正确性由上述所有权、设备/API、布局、累计状态或统计契约及其独立回归证明；没有强行套用MF/Megatron的不同实现，也未用该修复解释尚未定位的TP/CP/EP长步漂移。数值、重算和Muon的最新官方源码对照统一关联总问题单[#202](https://github.com/hf1s/hyper-parallel/issues/202)。

## 当前整体验收边界

本问题的定向回归与上文按版本标注的历史集成证据分别保留。最新训练源544526f7目前只有201的单次FSDP8完整200步及checkpoint CPU读回，共享机同源参考、跨机精度、最新续训和重算/并行矩阵尚未验收。当前先完成全200步、8rank主LM loss最大绝对差≤0.005；主loss全矩阵通过后先汇报，由用户决定是否继续MTP/aux专项。最新进度、曲线和资源约束见总问题单[#202](https://github.com/hf1s/hyper-parallel/issues/202)。本次仅补充说明，代码提交未改变，现有CI空结果不构成通过。
