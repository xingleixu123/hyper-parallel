关联总问题单：#202。本单仅跟踪 CPU 设备索引契约错误；125 新容器无法发现 NPU 的原因另行记录，不把两者合并为一个修复结论。

## 缺陷与影响

`hyper_parallel.models.build_options.get_device_id()` 注解为整数，却直接返回当前后端的 `current_device()`。CPU 后端返回字符串 `"cpu"`；JT 构建入口随后调用 `torch.device(mesh.device_mesh.device_type, get_device_id())`，因此在模型转移到目标设备前报错：

```text
TypeError: device(): argument 'index' (position 2) must be int, not str
```

影响无加速器环境下 JT 的随机初始化、参考权重加载和 CPU 组件验证。原有 JT CPU 测试把 `get_device_id` 固定 mock 为0，遮住了真实接口行为。普通 HyperAutoModel 的 `_current_device()` 已显式处理 CPU 分支，本单不声称该入口也发生同样异常。

## 最小复现

在不挂 NPU 的隔离容器中，以 editable 方式安装指定源码后执行：

```python
from unittest.mock import patch
import torch
from hyper_parallel.models import build_options

with patch.object(build_options, "IS_CUDA_AVAILABLE", False), \
        patch.object(build_options, "IS_NPU_AVAILABLE", False):
    kind = build_options.get_device_type()
    index = build_options.get_device_id()
    print(kind, repr(index), type(index).__name__)
    print(torch.device(kind, index))
```

修复前输出 `cpu 'cpu' str`，随后抛出上述 TypeError；只隔离后端可用性，不替换 `get_device_id()` 返回值。完整调用栈见[原始异常](https://github.com/xingleixu123/hyper-parallel/blob/94242c81939643ae54e12cdcb0674a5a157f0be9/docs/validation/jt-dev-8npu-20261014/reports/cpu-device-index/startup-rank3-stderr.log)，最小复现见[CPU 日志](https://github.com/xingleixu123/hyper-parallel/blob/94242c81939643ae54e12cdcb0674a5a157f0be9/docs/validation/jt-dev-8npu-20261014/reports/cpu-device-index/r125-cpu-device-index-probe.log)。

## 预期行为与修复范围

CPU 返回无设备编号的 `None`，使调用方构造规范的 `torch.device("cpu")`；CUDA/NPU 保留其当前整数编号。更新返回注解为 `Optional[int]` 及函数说明，不修改设备可用性判断、后端优先级、分布式通信、模型损失或优化器。

JT 原有 CPU 测试移除 `get_device_id=0` 的返回值 mock，改为隔离 CUDA/NPU 可用性标志，让初始化、加载和训练测试调用真实 API。另加 CPU 设备构造和 CUDA/NPU 当前索引透传检查；后两者只模拟后端，无实际设备分配。

## 验证结果与验收条件

- 同一测试入口 `python -m pytest -q tests/ut/auto_models/test_build_options.py tests/ut/models/jt_deepseek_v3/test_jt_builder.py`：修复前5失败、17通过、5子测试通过，19.00秒；修复后20通过、7子测试通过，19.06秒。保留原有初始化、参考加载、前后向和参数更新断言。
- 最小复现修复后返回 `None`，成功构造无索引 CPU 设备。
- 规范和验收条件两轴检视通过；静态检查与基点相比无新增告警，两者均6条既有告警。本机静态检查排除缺少 Torch 导致的 import-error/no-member。
- 第一次测试容器缺少 pytest，未收集测试；该失败单独保留，不算红回归或通过。随后使用隔离测试 venv，editable 导入路径已核对。
- 验收要求：上述真实 CPU 构建路径完整通过，CUDA/NPU 索引透传不变。CPU 结果不代替 NPU/HCCL 回归，也不证明125设备发现已恢复。

[修复前测试日志](https://github.com/xingleixu123/hyper-parallel/blob/94242c81939643ae54e12cdcb0674a5a157f0be9/docs/validation/jt-dev-8npu-20261014/reports/cpu-device-index/r125-cpu-device-index-red-r2.log) · [修复后测试日志](https://github.com/xingleixu123/hyper-parallel/blob/94242c81939643ae54e12cdcb0674a5a157f0be9/docs/validation/jt-dev-8npu-20261014/reports/cpu-device-index/r125-cpu-device-index-green.log) · [检视及补丁指纹](https://github.com/xingleixu123/hyper-parallel/blob/94242c81939643ae54e12cdcb0674a5a157f0be9/docs/validation/jt-dev-8npu-20261014/reports/cpu-device-index/cpu-device-index-review.json) · [完整独立补丁](https://github.com/xingleixu123/hyper-parallel/blob/94242c81939643ae54e12cdcb0674a5a157f0be9/docs/validation/jt-dev-8npu-20261014/reports/cpu-device-index/full-fix.patch)。

## 环境与排重

源码基点：`hf1s/hyper-parallel:JT_dev@f637ef7ee4d4b416800716f1ebaaae7709b8a06a`，已包含#180。CPU 复现环境为 Python3.12.14、Torch2.12.0+cpu、TorchNPU2.12.0、pytest9.1.1；镜像 CANN9.2.0-beta.2，CPU 容器没有挂载 NPU。

本次由125八卡启动失败触发排查：上层先报告无法获得ascend_hal设备数，回落CPU后暴露本接口错误。只读设备查询显示125新容器返回`device is used / -8020`，而129相同挂载查询成功；具体环境原因仍待确认，未停止他人容器或修改全机驱动设置。

已检索157条现有问题；#151中的CPU/设备字段涉及另一项MF→HP对齐工作，未描述本单的索引类型错误。对应独立修复：[PR #219](https://github.com/hf1s/hyper-parallel/pull/219)，单提交 `86930ddc7263f6ff53a79b504b9e8db33dc83e3b`，3个文件。本单保持开启，等待合入及维护者回归确认；当前提交尚无完整CI通过证据。

## 具体定位、触发条件与两侧正确性

固定本PR提交 `86930ddc7263f6ff53a79b504b9e8db33dc83e3b`。相关位置：[hyper_parallel/models/build_options.py::get_device_id:78](https://github.com/hf1s/hyper-parallel/blob/86930ddc7263f6ff53a79b504b9e8db33dc83e3b/hyper_parallel/models/build_options.py#L78)。

触发条件：CPU current_device() 返回字符串 cpu，get_device_id() 却把它当整数传给 torch.device(type,index)，JT 在设备构造时报 TypeError。

正确性参考：CPU 的无索引设备应可构造；CUDA/NPU 保留实际当前整数编号。原测试 get_device_id=0 的返回 mock 遮住缺陷。

修复侧或并行侧核验：移除返回值 mock 后红测5失败；绿测20 passed、7 subtests，真实 JT 初始化/参考加载/前后向和参数更新保留；后端可用性隔离。

覆盖边界：实际 CPU 接口缺陷修复，不证明 125 的 NPU 发现恢复，不替代 NPU/HCCL 或200步。

### 参考适用范围

本项的正确性由上述所有权、设备/API、布局、累计状态或统计契约及其独立回归证明；没有强行套用MF/Megatron的不同实现，也未用该修复解释尚未定位的TP/CP/EP长步漂移。数值、重算和Muon的最新官方源码对照统一关联总问题单[#202](https://github.com/hf1s/hyper-parallel/issues/202)。

## 当前整体验收边界

本问题的定向回归与上文按版本标注的历史集成证据分别保留。最新训练源544526f7目前只有201的单次FSDP8完整200步及checkpoint CPU读回，共享机同源参考、跨机精度、最新续训和重算/并行矩阵尚未验收。当前先完成全200步、8rank主LM loss最大绝对差≤0.005；主loss全矩阵通过后先汇报，由用户决定是否继续MTP/aux专项。最新进度、曲线和资源约束见总问题单[#202](https://github.com/hf1s/hyper-parallel/issues/202)。本次仅补充说明，代码提交未改变，现有CI空结果不构成通过。
