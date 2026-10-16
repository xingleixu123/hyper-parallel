# 共享机实际空卡设备发现记录（2026-10-17）

129 在实际空卡下完成最小设备发现，但容器内 NPU 可用性为 false、设备数为 0，设备索引返回字符串 `cpu`，随后 `torch.device('npu', index)` 抛出 TypeError。**这次是设备发现失败，未执行训练；不能据此判断任何 loss 精度。** 详细探针 v2 进一步记录事后直接 HAL 查询返回 `87`、出参仍为 `-1`，TorchNPU C 接口返回 `0`；已装驱动将 `87` 定义为 `DRV_ERROR_RESOURCE_OCCUPIED`，具体触发链仍未确定。v3 保留此前两份结果，并在其后对唯一选中节点实际打开，新增 `errno=16 EBUSY` 证据；不能据此反推此前 HAL 的同一次调用栈。201 仍缺同源共享机完整 200 步参考，尚未准入后续矩阵。

## 用户要求与实际调度

按最新明确要求，设备命名空间只作诊断。宿主 `npu-smi` 的计算进程、core 利用率、显存和健康状态确认空闲，并在启动前再次检查；映射存在不再单独阻断启动。125/129 每次确认释放后全局至少间隔 1200 秒。发现真实他人进程进入所选卡时，只停止自有实验。

| 实际运行 | 结果 | 证据边界 |
| --- | --- | --- |
| 125 单卡发现 v1 | 12.791 秒后发现其他容器的实际计算 PID；自有容器 exit 143、已释放 | 两次进程身份与归属核实；没有完整探针 JSON，不判发现 PASS 或 FAIL |
| 129 单卡发现 v1 | 18.560 秒，exit 1，已释放并确认自有容器数为 0 | 启动前两次实际空卡检查；运行中没有其他 PID 打断；完整原始探针结果 FAIL |
| 129 单卡详细发现 v2 | 18.530 秒，exit 1，自有实验已释放 | 生产 FAIL；事后 HAL rc=87/out=-1、C 接口设备数 0；有界采样内未发现他人计算进程或中止原因 |
| 129 单卡补充打开 v3 | 18.558 秒，exit 1，自有实验已释放且正确读回自有容器数为 0 | 原生产和 HAL/C 结果与 v2 字节相同；补充 open 返回 errno 16 EBUSY，未返回 fd |
| 201 同源单卡发现 v1 | 已有 PASS：NPU 可用、设备数 1、整数索引 0 | 相同探针源码；不同宿主硬件/驱动，不能单凭差异认定根因 |

125 的其他任务进程和容器身份仅保留在私有原文，未公开。[125 脱敏汇总](r125-summary.json)、[129 汇总](r129-summary.json)、[129 原始日志](r129-probe.log)、[201 原始日志](r201-probe.log)。

## 固定源码、导入路径与故障位置

两机探针固定集成源码 `544526f7db8342c607244fd41009b2a74adb84c8`、镜像 `sha256:5f9e677f439d11ab4e7c888f54b142335a27eee7411cbd2ce894a4798a70146d`，CANN 9.2.0-beta.2、Torch/TorchNPU 2.12.0。129 工作树干净，实际导入 `/workspace/hyper-parallel/hyper_parallel/models/build_options.py`，不是旧 editable 路径。[导入核对](r129-import-path.json)与[原始最小探针](probe_a1e6_device_discovery.py)。

129 本次明确捕获：`cached_npu_available=false`、`npu_available_now=false`、`npu_device_count=0`、`device_type='cpu'`、`device_index_type='str'`。这补齐了旧训练失败日志未保存的可用性与索引证据。`build_options.py:49` 缓存 NPU 可用性，`:55–80` 选择设备类型及当前索引，JT 构建器 `jt_builder.py:150` 将该索引交给 mesh 指定的设备类型；真实训练在此之前尚无 step 指标。

## 详细探针 v2：已观察到 HAL 87，尚未确定触发链

v1 当时未保存 HAL 原始返回码。v2 保留原生产导入和 `get_device_id → torch.device` 路径，先输出[生产 JSON](detail-v2/production.json)，再执行独立的事后诊断；没有创建张量、设置设备或训练。[实际执行的探针源码](detail-v2/probe_a1e6_device_discovery_detail_v2.py)和[导入/日志结构核对](detail-v2/import-path.json)可复核顺序。后者的 `PASS` 仅表示来源、两条 JSON 的完整性和顺序正确，**设备发现结果仍为 `FAIL`**。

[诊断公开副本](detail-v2/post-production-diagnostics.public.json)记录 `ctypes.CDLL('libascend_hal.so').drvGetDevNum` 返回码为 `87`，初始化为 `c_int(-1)` 的出参调用后仍为 `-1`；随后 `torch_npu._C._npu_getDeviceCount()` 返回整数 `0`。诊断前后 `_cached_device_count=null`、`_initialized=false`，实际映射的 HAL 路径仍为 `/usr/local/Ascend/driver/lib64/driver/libascend_hal.so`。这些是生产失败之后的查询，不能反推此前生产调用必然经过同一个 C fallback 分支。日志 `DrvMngGetConsoleLogLevel failed. (ret=4)` 来自另一接口，仍不能当作 `drvGetDevNum` 返回码。

129 已安装驱动版本为 `26.0.rc1`、内部版本 `V100R001C10SPC001B231`；该机 `/usr/local/Ascend/driver/include/ascend_hal_error.h:112` 明确写出 `DRV_ERROR_RESOURCE_OCCUPIED = 87`。[版本、头文件行及 SHA 摘录](detail-v2/installed-driver-header.public.json)。**枚举名称不等同于他人正在计算，也没有证明具体容器/驱动触发链。** 前后八卡快照均健康、计算 PID 为空、core 为 0，三次 HBM 采样最高 3,422 MiB；这些有限采样不能证明全程没有采样盲区。可见的四个字符设备节点、UID/EUID、能力位和自身 cgroup 访问规则仅作为诊断事实，尚不足以排除所有权限、驱动、命名空间或运行时因素。

[独立实际证据复核摘录](detail-v2/independent-review.public.json)确认本轮 11 个文件共 30,374 字节逐 SHA 一致，两条原始 JSON 与单独保存的结果相等，退出码为 1；没有训练或精度通过结论。其证据复核 `PASS` 也不改变生产失败结果。公开诊断仅删除整个 `self_namespace_identity` 字段（自身 PID 和 namespace 标识），其余诊断值保留；不公开包含这些身份的完整原始日志或调度 receipt。[新增清单](detail-v2/manifest.json)列出原始 SHA、公开副本 SHA、摘录范围及删减字段，原文件继续私有保留。

## 补充探针 v3：所选节点真实 open 返回 EBUSY

v3 在原生产与事后 HAL/C 两条 JSON 完成输出后，只对唯一映射的 `/dev/davinci0` 执行一次 `O_RDONLY|O_NOFOLLOW` 打开，实测 flags 为 32768、字符设备 major/minor 为 236/0，结果为 `errno=16 EBUSY`。未返回 fd，因此 `close.attempted=false` 符合控制流；资源释放另由容器退出、移除及正确的自有容器读回确认。[第三条记录公开副本](device-open-v3/selected-device-open.public.json)仅删除 `self_pid`、`self_mntns`，其他键值完全保留。`/sys/fs/cgroup/memory` 的 statvfs 显示只读；这只是同时观察到的事实，不是独立因果证明。

[实际探针](device-open-v3/probe_a1e6_device_open_v3.py)、[实际容器内 runner](device-open-v3/run_fair_a1e6_device_open_v3.sh)及[来源/三条记录顺序检查](device-open-v3/import-path.json)均为实际文件的逐字节副本，只供审阅，不表示再次运行。前两份原始 JSON 与 v2 字节一致，直接引用已有[生产结果](detail-v2/production.json)和[诊断公开副本](detail-v2/post-production-diagnostics.public.json)，避免重复保存；诊断公开副本此前删除整个 `self_namespace_identity` 的规则仍不变。来源/解析 `PASS` 和[独立证据复核通过](device-open-v3/independent-review.public.json)均不改变生产 **FAIL / exit 1**。

本次 12 个采集文件共 40,065 字节全部逐 SHA 核验（运行目录 11 个，另有私有调度 receipt 1 个），18.558 秒后释放，正确读回自有容器与剩余自有 PID 均为 0。记录当时的下一共享机最早启动时间为 **2026-10-16 17:41:08.344891 UTC**，不表示该时刻已经启动其他实验。前后快照及有界采样未记录真实计算冲突，但第二次完整空卡快照未另存，启动复查由固定 guard 控制流和 attempted 佐证，不能宣称采样间隔内绝无短暂活动。

这次只证明补充 open 在其发生时刻返回 EBUSY；不能反推此前 HAL 同一次调用栈、指认具体 namespace 持有者、认定某个权限或只读挂载就是根因，也没有训练、精度或迁移通过结论。原始日志、namespace 标识和完整 receipt 继续私有保留；原始与公开 SHA、删减规则及复用关系见[v3 清单](device-open-v3/manifest.json)。

## 容器共享与因果边界

只读查询显示三机 card 0 的 `Device-share Status` 均为 `False`；未修改宿主共享设置。[查询原文](device-share-query.json)。官方文档说明设备共享模式控制多个容器能否映射同一设备，但当前事实只支持继续做对照定位，不证明本次完整因果链。[官方共享模式说明](https://www.hiascend.com/document/detail/en/mindcluster/2610/clustersched/schedulingug/docs/en/scheduling/04_usage/02_virtual_instance/01_virtual_instance_with_vcann_rt/01_soft_allocation_scheduling_inference.md)。不同版本的支持范围和硬件约束需分别核实。

下一步仍是同机成功发现后取得完整 200 步 × 8 rank 参考，再与 201 的同源、同数据、同初态主 LM loss 逐点比较，全部差值 ≤0.005 才迁移。既有 18602c4/f637 基线、续训和重算结论均保留为历史结果，不重标为 544526f7 通过。

本轮没有改变训练源码或已发表 PR 的提交；不新增未确定根因的产品修复。v1 两次探针共 18 个收集文件逐 SHA 核对，复制完成后的两端重复归档已清理，原始证据及 checkpoint 保留；v2 的 11 个文件单独核对，不覆盖 v1。原 [v1 清单](manifest.json)保持字节不变，其中 README 的哈希对应[原发布提交的版本](https://github.com/xingleixu123/hyper-parallel/blob/1e5b525018833cf4a1ef0eb5ea286d35c6cf7e26/docs/validation/jt-dev-8npu-20261014/reports/actual-idle-discovery-20261017/README_ZH.md)，detail-v2 清单中的 README 哈希对应[已发布的 61ee782 版本](https://github.com/xingleixu123/hyper-parallel/blob/61ee782d1337b2a9d8d6e57da68b981f423ee58e/docs/validation/jt-dev-8npu-20261014/reports/actual-idle-discovery-20261017/README_ZH.md)，该清单和全部 v2 文件保持不变；当前 README 哈希由新增 [v3 清单](device-open-v3/manifest.json)绑定。
