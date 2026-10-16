# 共享机实际空卡设备发现记录（2026-10-17）

129 在实际空卡下完成最小设备发现，但容器内 NPU 可用性为 false、设备数为 0，设备索引返回字符串 `cpu`，随后 `torch.device('npu', index)` 抛出 TypeError。**这次是设备发现失败，未执行训练；不能据此判断任何 loss 精度。** 201 仍缺同源共享机完整 200 步参考，尚未准入后续矩阵。

## 用户要求与实际调度

按最新明确要求，设备命名空间只作诊断。宿主 `npu-smi` 的计算进程、core 利用率、显存和健康状态确认空闲，并在启动前再次检查；映射存在不再单独阻断启动。125/129 每次确认释放后全局至少间隔 1200 秒。发现真实他人进程进入所选卡时，只停止自有实验。

| 实际运行 | 结果 | 证据边界 |
| --- | --- | --- |
| 125 单卡发现 v1 | 12.791 秒后发现其他容器的实际计算 PID；自有容器 exit 143、已释放 | 两次进程身份与归属核实；没有完整探针 JSON，不判发现 PASS 或 FAIL |
| 129 单卡发现 v1 | 18.560 秒，exit 1，已释放并确认自有容器数为 0 | 启动前两次实际空卡检查；运行中没有其他 PID 打断；完整原始探针结果 FAIL |
| 201 同源单卡发现 v1 | 已有 PASS：NPU 可用、设备数 1、整数索引 0 | 相同探针源码；不同宿主硬件/驱动，不能单凭差异认定根因 |

125 的其他任务进程和容器身份仅保留在私有原文，未公开。[125 脱敏汇总](r125-summary.json)、[129 汇总](r129-summary.json)、[129 原始日志](r129-probe.log)、[201 原始日志](r201-probe.log)。

## 固定源码、导入路径与故障位置

两机探针固定集成源码 `544526f7db8342c607244fd41009b2a74adb84c8`、镜像 `sha256:5f9e677f439d11ab4e7c888f54b142335a27eee7411cbd2ce894a4798a70146d`，CANN 9.2.0-beta.2、Torch/TorchNPU 2.12.0。129 工作树干净，实际导入 `/workspace/hyper-parallel/hyper_parallel/models/build_options.py`，不是旧 editable 路径。[导入核对](r129-import-path.json)与[原始最小探针](probe_a1e6_device_discovery.py)。

129 本次明确捕获：`cached_npu_available=false`、`npu_available_now=false`、`npu_device_count=0`、`device_type='cpu'`、`device_index_type='str'`。这补齐了旧训练失败日志未保存的可用性与索引证据。`build_options.py:49` 缓存 NPU 可用性，`:55–80` 选择设备类型及当前索引，JT 构建器 `jt_builder.py:150` 将该索引交给 mesh 指定的设备类型；真实训练在此之前尚无 step 指标。

当前尚未保存 `drvGetDevNum` 原始返回码和 C 后备设备数。日志 `DrvMngGetConsoleLogLevel failed. (ret=4)` 来自控制台日志接口，**不能当作 drvGetDevNum 的返回码**。下一轮补齐这两个观察量，区分容器隔离、宿主驱动配套及框架可见性/缓存，不能通过改成 CPU 索引或强制缓存为 true 掩盖故障。

## 容器共享与因果边界

只读查询显示三机 card 0 的 `Device-share Status` 均为 `False`；未修改宿主共享设置。[查询原文](device-share-query.json)。官方文档说明设备共享模式控制多个容器能否映射同一设备，但当前事实只支持继续做对照定位，不证明本次完整因果链。[官方共享模式说明](https://www.hiascend.com/document/detail/en/mindcluster/2610/clustersched/schedulingug/docs/en/scheduling/04_usage/02_virtual_instance/01_virtual_instance_with_vcann_rt/01_soft_allocation_scheduling_inference.md)。不同版本的支持范围和硬件约束需分别核实。

下一步仍是同机成功发现后取得完整 200 步 × 8 rank 参考，再与 201 的同源、同数据、同初态主 LM loss 逐点比较，全部差值 ≤0.005 才迁移。既有 18602c4/f637 基线、续训和重算结论均保留为历史结果，不重标为 544526f7 通过。

本轮没有改变训练源码或已发表 PR 的提交；不新增未确定根因的产品修复。两次探针共 18 个收集文件逐 SHA 核对，复制完成后的两端重复归档已清理，原始证据及 checkpoint 保留。
