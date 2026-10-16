# 当前 PTA 精度开关：源码核对与后续候选边界

2026-10-16（UTC+8）在四臂 dgrad 诊断的冷却窗口进行只读核查，没有改变任何训练请求、环境或计算设置。本文件是后续候选记录，不是新增缺陷结论或硬件验证。

从固定镜像中创建但从未启动的无设备容器，读取 `torch_npu/version.py` 和 `npu/npu_config.py` 后立即移除容器。实际 PTA 为 `2.12.0`，Git 版本 `fa0f83fe49d309dcbc31e264e9e6ed6e5dc49d2d`；读取与删除凭据见 [镜像源文件核验](fixed-image-npu-precision-source-inspection.json)。随后读取官方仓库这一精确提交的源码，未用当前 master 代替安装版本。

1. 已安装的 `torch.npu.matmul` 支持 `cube_math_type`，枚举包含 `USE_FP32_ADD`。官方 CANN 9.1 的 A2/A3 Matmul 文档将该选项限定为 FLOAT32 且 K 大于 2048 的分组累加；不能把它描述成 BF16 全网高精度开关，也不能直接外推为当前 CANN 9.2 的实测效果。当前四臂没有设置此选项。[CANN Matmul 约束](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/aolapi/context/ops-nn/aclnnMatmul.md)
2. 官方 GMM 文档的非量化 BF16 输入对应 BF16 输出。仅设置 `output_dtype=float32` 不能据此认为已实现 FP32 权重梯度；现有专家主梯度融合路径与这个参数的含义不同。[TorchNPU GMM 类型约束](https://www.hiascend.com/document/detail/en/Pytorch/2610/apiref/customapi/docs/en/custom_APIs/torch_npu/torch_npu-npu_grouped_matmul.md)
3. 精确 PTA 提交提供 `set_deterministic_level`。其 level 2 会请求 runtime 的 strong-consistency 设置；Python 接口告警不可与后续 `torch.use_deterministic_algorithms` 混用，相关硬件测试类在该提交被整体跳过。当前 Trainer 已启用普通 deterministic algorithms 和 `CLOSE_MATMUL_K_SHIFT=1`，仍不能由此宣称启用了 level 2 或保证跨形状一致。[Python 合同](https://github.com/Ascend/pytorch/blob/fa0f83fe49d309dcbc31e264e9e6ed6e5dc49d2d/torch_npu/npu/__init__.py#L603)、[runtime 设置](https://github.com/Ascend/pytorch/blob/fa0f83fe49d309dcbc31e264e9e6ed6e5dc49d2d/torch_npu/csrc/core/npu/sys_ctrl/npu_sys_ctrl.cpp#L52)、[跳过的测试](https://github.com/Ascend/pytorch/blob/fa0f83fe49d309dcbc31e264e9e6ed6e5dc49d2d/test/npu/test_deterministic_level.py#L10)
4. 该版本私有 `_get_deterministic_level()` 会在状态不一致时写回设置，不能把它放进宣称纯只读的观测 hook。若以后探索 level 2，需在独立进程检查初始化次序、适用硬件、真实生效状态与性能成本；保持原生/候选两侧分别匹配，不混入本轮 A/B/C/D。
5. Trainer 的 NPU 分支写有 NCCL 名称的变量，但仅凭名称不足以宣布 HCCL 没有确定性：同一 PTA 的通信实现还读取框架状态，并对较新 CANN 使用版本分支。当前没有新建这一推测的 issue/PR。[精确通信源码](https://github.com/Ascend/pytorch/blob/fa0f83fe49d309dcbc31e264e9e6ed6e5dc49d2d/torch_npu/csrc/distributed/ProcessGroupHCCL.cpp#L286)

优先完成已经准备且通过前置检查的四臂实验，再依据实际首步差异是否缩小决定后续探针。`USE_FP32_ADD` 与 strong consistency 都尚未进行本任务的硬件对照，不应宣传为已验证修复。

源码指纹：[PTA runtime 实现](fixed-pta-determinism-implementation.json)、[Python 入口与测试](fixed-pta-determinism-python-contract.json)、[HCCL 源码](fixed-pta-hccl-determinism-source.json)。安装源文件保留在任务 `references/`，均为只读检查证据。

## 后续只读补充：位置一致性与跨 batch 一致性不能混称

CANN 9.1 的官方 ATC 文档将 level 2 描述为强一致性：同一输入位于不同数据位置时，计算结果不再因位置变化而不同；文档同时说明这是调试功能，且有性能成本。该页是 ATC/图侧接口文档，不能代替当前 eager BF16 Linear、GMM 和 HCCL 的逐算子实测。[官方级别语义](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/910/devaids/atctool/atlasatcparam_16_0100.html)

已安装 PTA 对应的 `SetDeterministicFromLevel` 只允许 0/1/2，level 2 同时开启 `ACL_OPT_DETERMINISTIC` 和 `ACL_OPT_STRONG_CONSISTENCY`。不能把更新版本的 level 3／batch 一致性接口套用到该精确提交。官方测试以 `a[:1] @ b` 对比 `(a @ b)[:1]` 检查跨行数结果，但整个类被 skip，且样例为 FLOAT32；它不证明当前 BF16、CP 分区或专家分组大小变化已经受支持。

若四臂结果仍需要后续定位，优先准备独立进程、相同固定输入的 level 1/2 算子对照，分别检查行置换和行数变化。先在获准空闲的一张卡验证实际收益及限制，再决定是否值得重建训练参考；同样遵守全局20分钟间隔。当前仅记录候选，没有运行该探针，没有修改 A/B/C/D，没有新增缺陷 PR。
