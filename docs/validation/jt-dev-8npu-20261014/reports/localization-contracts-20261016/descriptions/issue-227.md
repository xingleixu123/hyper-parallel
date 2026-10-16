## 问题描述

关联总问题单：#202。对应独立单提交草稿 PR：[#228](https://github.com/hf1s/hyper-parallel/pull/228)。本问题只跟踪 Trainer 的 NPU 高精度初始化开关，不与其他数值算法或资源统计修改合并。

`BaseTrainer` 初始化会调用 `enable_high_precision_for_bf16()`。当前 NPU 分支写入 `torch.npu.matmul.allow_tf32` 和 `allow_bf16_reduced_precision_reduction`，但 TorchNPU 2.12.0 的 matmul 配置对象使用 `allow_hf32`；对前两个名称的写入不执行任何设置，也不抛异常。若应用此前启用了 HF32，调用 Trainer 的高精度函数后 HF32 仍然开启，初始化意图没有落实。

已在与八卡实验相同的 `jt-clean:hp-910b-20261009` 镜像中，用安装版本的真实配置对象复现。容器未映射设备，全程 `torch.npu.is_initialized() == False`。只在测试中强制选择 NPU 条件分支，没有把真实 backend 配置对象替换成 Mock。

## 复现环境与源码

- 上游：`hf1s/hyper-parallel:JT_dev`，`f637ef7ee4d4b416800716f1ebaaae7709b8a06a`，包含 #180 和 #186。
- 生产文件：`hyper_parallel/trainer/runtime/random.py`；调用点：`hyper_parallel/trainer/base.py:290`。
- 镜像 ID：`sha256:5f9e677f439d11ab4e7c888f54b142335a27eee7411cbd2ce894a4798a70146d`。
- Python 3.12.14，Torch 2.12.0+cpu，TorchNPU 2.12.0；镜像中的 CANN 为 9.2.0-beta.2（V100R001C12B086）。本复现没有执行 NPU 算子，因此 CANN 版本不是这次 API 读写验证的硬件验收结论。
- 无设备容器：4 个 CPU 配额、8 GiB 内存上限、禁用网络，完成后移除。

## 最小复现

在安装了 TorchNPU 和该源码的环境执行：

```python
from unittest.mock import patch
import torch
import torch_npu
from hyper_parallel.trainer.runtime import random as controls

initial = torch.npu.matmul.allow_hf32
try:
    torch.npu.matmul.allow_hf32 = True
    with patch.object(controls, "IS_NPU_AVAILABLE", True), patch.object(controls, "IS_CUDA_AVAILABLE", False):
        controls.enable_high_precision_for_bf16()
    print(torch.npu.matmul.allow_hf32)
finally:
    torch.npu.matmul.allow_hf32 = initial
```

设备可用性条件的 patch 仅使无设备容器走到待测分支；选项的 setter/getter 仍来自真实 TorchNPU。旧代码输出 `True`，预期为 `False`。实际 probe 另外核对了设备运行时未初始化，并恢复原有选项值。

## 实测结果与预期

| 检查 | 旧代码 | 独立修复后 |
|---|---|---|
| probe 进程的默认 HF32 值 | `False` | `False` |
| 显式开启 HF32 后调用高精度函数 | 仍为 `True` | 变为 `False` |
| 直接调用受支持的 `allow_hf32=False` | 能关闭 | 能关闭 |
| 无效的 CUDA 风格属性读回 | 均为 `None` | 不再写入，读回仍为 `None` |
| NPU 运行时初始化状态 | `False` | `False` |
| 同一组 CPU UT | NPU 两项子测试失败 | 3 项测试、2 项子测试通过 |

修复应使用 `torch.npu.matmul.allow_hf32 = False`，删除两次无效属性写入。CUDA 的 TF32/BF16 reduction 行为及函数签名保持兼容；CPU 分支不访问加速器接口。文档须明确：NPU HF32 开关控制 FP32 matmul 的降精度路径，本修复不声称它能控制 BF16 累加精度。

## 验收边界

1. 覆盖 HF32 原来开启和原来关闭两种情况，调用后都关闭，且不引入 CUDA 专属属性。
2. CUDA 两个原有策略保持，CPU-only 环境无加速器属性访问。
3. 同一 CPU 回归在未修复基点失败、修复后通过；真实已安装 TorchNPU 的开关读写也通过。
4. 单独 PR、单提交，正文回链本问题和 #202；保留草稿等待维护者检视。
5. **该问题尚未证明是 EP2/CP2 200 步精度漂移的根因。** 实际 probe 的默认 HF32 值为关闭，当前固定训练配方不会为了此修复中途切换源码。实际硬件 matmul、200 步数值与吞吐影响未在本独立修复上验证，不能据此宣称八卡验收通过。

已搜索仓库 HF32 相关问题，#210/#201 涉及 Muon NS 精度配置，#151 仅记录对齐实验中的 HF32 设置，与本问题的错误属性写入不同。已检查实际安装源码及 [TorchNPU 官方配置实现](https://github.com/Ascend/pytorch/blob/master/torch_npu/npu/npu_config.py)。

完整证据已发布：[中文证据说明](https://github.com/xingleixu123/hyper-parallel/blob/dabd2b719dbced0c69fac6bc0fe05c937c8b52a6/docs/validation/jt-dev-8npu-20261014/reports/npu-precision-switch/README_ZH.md)、[所有文件指纹](https://github.com/xingleixu123/hyper-parallel/blob/dabd2b719dbced0c69fac6bc0fe05c937c8b52a6/docs/validation/jt-dev-8npu-20261014/reports/npu-precision-switch/manifest.json)、[规范轴检视](https://github.com/xingleixu123/hyper-parallel/blob/dabd2b719dbced0c69fac6bc0fe05c937c8b52a6/docs/validation/jt-dev-8npu-20261014/reports/npu-precision-switch/npu-precision-standards-review.json)、[需求轴检视](https://github.com/xingleixu123/hyper-parallel/blob/dabd2b719dbced0c69fac6bc0fe05c937c8b52a6/docs/validation/jt-dev-8npu-20261014/reports/npu-precision-switch/npu-precision-spec-review.json)。两轴均通过。初版批次的 UT 已完成，但尾部 JSON 读取受到驱动 stdout 文本干扰而返回 1；后续改为独立 JSON 输出的 probe 退出 0。这两种状态分别记录，不将整个初版批次误标为通过。

## 具体定位、触发条件与两侧正确性

固定本PR提交 `453fe5723429b0e45ee6f0236d6e262cbe58b04f`。相关位置：[hyper_parallel/trainer/runtime/random.py::enable_high_precision_for_bf16:34](https://github.com/hf1s/hyper-parallel/blob/453fe5723429b0e45ee6f0236d6e262cbe58b04f/hyper_parallel/trainer/runtime/random.py#L34)。

触发条件：创建 Trainer 前 HF32=true；NPU 分支写 CUDA 的 allow_tf32/BF16 属性被 TorchNPU 静默忽略，未关闭真实 allow_hf32。

正确性参考：TorchNPU 2.12.0 实际 setter/getter、CPU 不触碰加速器、CUDA 原有两选项不变；HF32影响FP32 matmul降精度，不等于BF16累加控制。

修复侧或并行侧核验：替身 UT 红2子测试失败→3 passed/2 subtests；真实安装 TorchNPU 配置对象 false/true 读写探针，NPU未初始化，最后恢复原值。

覆盖边界：默认 HF32=false；未证明是长步漂移原因。无独立修复设备 matmul/200步/收益验证。

### 参考适用范围

本项的正确性由上述所有权、设备/API、布局、累计状态或统计契约及其独立回归证明；没有强行套用MF/Megatron的不同实现，也未用该修复解释尚未定位的TP/CP/EP长步漂移。数值、重算和Muon的最新官方源码对照统一关联总问题单[#202](https://github.com/hf1s/hyper-parallel/issues/202)。

## 当前整体验收边界

本问题的定向回归与上文按版本标注的历史集成证据分别保留。最新训练源544526f7目前只有201的单次FSDP8完整200步及checkpoint CPU读回，共享机同源参考、跨机精度、最新续训和重算/并行矩阵尚未验收。当前先完成全200步、8rank主LM loss最大绝对差≤0.005；主loss全矩阵通过后先汇报，由用户决定是否继续MTP/aux专项。最新进度、曲线和资源约束见总问题单[#202](https://github.com/hf1s/hyper-parallel/issues/202)。本次仅补充说明，代码提交未改变，现有CI空结果不构成通过。
