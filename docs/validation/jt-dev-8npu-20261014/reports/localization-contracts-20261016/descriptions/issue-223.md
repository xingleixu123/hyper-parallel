## 问题概述

总跟踪：#202。本问题仅跟踪测试调度器继承 NPU 可见设备的缺陷，已提交独立、单提交 PR #224 修复。

上层将 `ASCEND_RT_VISIBLE_DEVICES` 限定为 `4,5,6,7` 后，`tests.common.parallel_case.parallel_run` 仍从零生成设备编号。两个各需两卡的用例实际收到 `0,1`、`2,3`；`run_case` 又将这些编号写回子进程环境，从而丢失上层分配。预期分别使用 `4,5`、`6,7`。

这是可以在无设备进程的条件下复现的逻辑错误。它可能造成跨出预分配范围、用例与其他任务争卡，或在容器未映射目标设备时初始化失败；这些后果是代码路径推论，本问题不声称已经对他人实验造成干扰。

## 版本与环境

- 发现及修复基点：`hf1s/hyper-parallel:JT_dev@f637ef7ee4d4b416800716f1ebaaae7709b8a06a`，已包含 #180；本轮提交前再次读取远端分支确认。
- 初次定位使用的训练集成树为 `18602c4524dc664b81b2bad9a7b624455ce1494f`；该文件与上述基点相同，随后在基点独立检出再次复现。
- 回归环境：129 上的 `jt-clean:hp-910b-20261009`，Python 3.12.14、pytest 9.1.1。容器没有挂载 NPU，限制 2 CPU / 1 GiB，实际设备/进程边界由 mock 替代。
- 镜像包含 CANN 9.2.0-beta.2，但本问题的单测不初始化 CANN、Torch、TorchNPU 或 HyperParallel。

## 最小复现

在 Linux 仓库根目录执行以下脚本，不会启动 torchrun 或占用 NPU：

```python
import os
from unittest.mock import patch
from tests.common.parallel_case import TorchCase, parallel_run

with patch.dict(os.environ, {
    "HYPER_PARALLEL_TEST_DEVICE_TYPE": "npu",
    "ASCEND_RT_VISIBLE_DEVICES": "4,5,6,7",
}), patch("tests.common.parallel_case._parallel_run_context") as context:
    process = context.return_value.Process.return_value
    process.is_alive.return_value = False
    process.exitcode = 0
    parallel_run([
        TorchCase("worker.py", "first", 23456, 2),
        TorchCase("worker.py", "second", 23457, 2),
    ])
    print([call.kwargs["args"][0]
           for call in context.return_value.Process.call_args_list])
```

修复前输出 `[[0, 1], [2, 3]]`，期望 `[[4, 5], [6, 7]]`。本例没有异常栈，因为启动器错误地把错误分配当成有效结果。独立的 worker 边界测试还确认 `run_case` 会使用所传设备列表改写 `ASCEND_RT_VISIBLE_DEVICES`。

## 根因与影响范围

1. `parallel_run` 用累计进程数生成 `range(...)`，未读取父进程可见设备列表。
2. 仅以 `global_num_proc` 断言总卡数，未将可见设备数量作为实际上限；默认预算 8 不应使仅授权两卡的任务拥有额外设备。
3. 启动前缺乏有效设备列表和正整数进程数校验，不能保证切分范围有效且不重叠。

影响通过 `parallel_run` 启动的 NPU ST。直接调用 `torchrun_case` 的路径不在本问题修复范围；模型、通信算子、优化器、loss 和 checkpoint 格式均不涉及。

## 期望行为与验收标准

- 父进程设置合法可见设备列表时，按其原有顺序分配互不重叠的子集，不将设备 ID 重新编号为零开始。
- 实际上限同时受 `global_num_proc` 与可见设备数量约束；超过上限，在分配端口或启动子进程之前明确报错。
- 非正整数进程数，空列表、重复 ID、非数字、空格、负值或非升序的设备列表明确报 `ValueError`，不回退到全部设备。
- 不设置可见列表时保持现有零起始分配；CPU 模式忽略 NPU 列表并继续清除 worker 的 NPU 可见性设置。
- 在全新解释器中导入启动器，不导入 `torch`、`torch_npu`、`hyper_parallel`。

环境变量中的 ID 是 Runtime 所见的逻辑设备 ID，与 worker 内重映射后的索引有区别。按照[昇腾官方环境变量文档](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/910/maintenref/envvar/envref_07_0028.html)，采用逗号分隔、升序且不含其他字符的列表契约；测试启动器主动拒绝畸形输入，不尝试模拟 Runtime 的截断行为，也不加载 Runtime 探测设备是否存在。

## 已完成的回归与边界

同一份新增回归在修复前输出 `24 failed, 4 passed`（包含子测试失败计数）；修复后输出 `10 passed, 18 subtests passed`。覆盖连续和稀疏设备列表、两组切分、预算约束、非法输入、CPU 模式、worker 环境透传，以及全新解释器的轻量导入。

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/ut/common/test_parallel_case.py
```

本项属于调度逻辑回归，不等同于实际多卡训练验证，不提供吞吐、显存收益或 200 步 loss 对齐结论。总任务的其他并行与重计算验收仍由 #202 分项记录。问题保持开启，待独立 PR 检视和后续维护者回归。


## 独立提交与公开证据

独立单提交 PR：[#224](https://github.com/hf1s/hyper-parallel/pull/224)，提交 `a37d6bcb699d0509477882a71348c2d1d3d83177`，两文件；规范与需求两轴检视均通过，当前仍为草稿。

[修复前原始日志](https://github.com/xingleixu123/hyper-parallel/blob/4432472f2667832980835fb2bff9a815e38570a4/docs/validation/jt-dev-8npu-20261014/reports/parallel-case-visible-devices/parallel-case-visible-devices-before.log) · [修复后原始日志](https://github.com/xingleixu123/hyper-parallel/blob/4432472f2667832980835fb2bff9a815e38570a4/docs/validation/jt-dev-8npu-20261014/reports/parallel-case-visible-devices/parallel-case-visible-devices-after.log) · [两轴检视与指纹](https://github.com/xingleixu123/hyper-parallel/blob/4432472f2667832980835fb2bff9a815e38570a4/docs/validation/jt-dev-8npu-20261014/reports/parallel-case-visible-devices/parallel-case-visible-devices-review.json) · [静态检查基点对照](https://github.com/xingleixu123/hyper-parallel/blob/4432472f2667832980835fb2bff9a815e38570a4/docs/validation/jt-dev-8npu-20261014/reports/parallel-case-visible-devices/parallel-case-visible-devices-lint.json)

## 具体定位、触发条件与两侧正确性

固定本PR提交 `a37d6bcb699d0509477882a71348c2d1d3d83177`。相关位置：[tests/common/parallel_case.py::_visible_device_pool:82](https://github.com/hf1s/hyper-parallel/blob/a37d6bcb699d0509477882a71348c2d1d3d83177/tests/common/parallel_case.py#L82)；[tests/common/parallel_case.py::_device_assignments:97](https://github.com/hf1s/hyper-parallel/blob/a37d6bcb699d0509477882a71348c2d1d3d83177/tests/common/parallel_case.py#L97)；[tests/common/parallel_case.py::parallel_run:119](https://github.com/hf1s/hyper-parallel/blob/a37d6bcb699d0509477882a71348c2d1d3d83177/tests/common/parallel_case.py#L119)。

触发条件：上层 ASCEND_RT_VISIBLE_DEVICES=4,5,6,7，旧 parallel_run 仍分配 0,1 和2,3；会越过资源授权范围。

正确性参考：Runtime 逻辑设备 ID 池及 global_num_proc 双重预算；按父列表顺序分组且互不重叠；非法输入先于端口/进程副作用失败。

修复侧或并行侧核验：同新增回归红测24 fail/4 pass；绿测10 passed、18 subtests；连续/稀疏选卡、非法列表、预算、CPU 清变量、导入不加载框架。

覆盖边界：仅无设备调度逻辑；不验证实际 ID 存在、多卡通信或200步。已有 Ascend 官方环境变量契约链接。

### 参考适用范围

本项的正确性由上述所有权、设备/API、布局、累计状态或统计契约及其独立回归证明；没有强行套用MF/Megatron的不同实现，也未用该修复解释尚未定位的TP/CP/EP长步漂移。数值、重算和Muon的最新官方源码对照统一关联总问题单[#202](https://github.com/hf1s/hyper-parallel/issues/202)。

## 当前整体验收边界

本问题的定向回归与上文按版本标注的历史集成证据分别保留。最新训练源544526f7目前只有201的单次FSDP8完整200步及checkpoint CPU读回，共享机同源参考、跨机精度、最新续训和重算/并行矩阵尚未验收。当前先完成全200步、8rank主LM loss最大绝对差≤0.005；主loss全矩阵通过后先汇报，由用户决定是否继续MTP/aux专项。最新进度、曲线和资源约束见总问题单[#202](https://github.com/hf1s/hyper-parallel/issues/202)。本次仅补充说明，代码提交未改变，现有CI空结果不构成通过。
