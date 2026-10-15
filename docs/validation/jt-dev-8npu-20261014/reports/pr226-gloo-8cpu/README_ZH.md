# #225 / #226：八进程 CPU/Gloo 资源归约回归

本次补充验证使用真实八进程 Gloo 通信。固定基点 `f637ef7ee4d4b416800716f1ebaaae7709b8a06a` 和 PR #226 单提交 `b58fed7fcdce38a8a62c5fb5bae845247fcbbeb7` 分别从精确 Git 归档安装，逐进程检查 editable 导入路径及回调文件指纹。未修改 PR 代码或增加 PR 提交。

最终验收采用 **v2**：每个版本覆盖三种进程组边界、两步、八个 rank，共 48 条观测，每条检查 12 项指标。每次 `all_reduce`、world size 和进程组均来自真实运行时，没有 mock 通信结果。为了构造确定反例，仅注入时钟及显存计数器哨兵。

| 归约组边界 | 基点相对全局正确值的资源差异项数 | 修复版差异项数 |
| --- | ---: | ---: |
| DP2×CP2×TP2 对应分组：`(0,2,4,6)` / `(1,3,5,7)` | 32 | 0 |
| DP2×CP2×PP2 对应分组：`(0,1,2,3)` / `(4,5,6,7)` | 32 | 0 |
| DP8 world 控制组 | 0 | 0 |
| 合计 | 64 | 0 |

基点的 64 项差异逐项符合旧代码的**组内 MAX**，不是只凭“出现错误”判定红测成功。修复版全部符合 world MAX。两侧的数据计数、累积计数、loss、梯度范数、学习率以及 world 控制项均没有非预期差异。每个版本 48 条观测合计 576 次指标比较；两版本合计 96 条、1152 次。

第一步的预设时间为 `[2,8,3,4,5,6,7,9]` 秒，allocated 为 `[1,13,2,3,4,5,6,7]` GiB，reserved 为 `[2,17,3,4,5,6,7,8]` GiB。第二步资源输入乘二，并验证累积 token/sample 计数。loss 使用各 rank 不同的哨兵，便于发现错误扩大归约域造成的串组平均。

## 运行范围与未覆盖项

- 使用 129 的固定 `jt-clean:hp-910b-20261009` 镜像，Python 3.12.14、Torch 2.12.0+cpu。容器限制为 4 CPU、8 GiB，无 NPU 设备映射；八个进程在 CPU 上调度。
- 两个版本均退出 0，逐 rank 记录 `npu_initialized=false`。无设备容器结束后已移除；临时源树、venv 和传输归档的后续清理另有凭据。
- **这是进程组归约边界测试，不是 TP/PP 模型训练，也不是 NPU/HCCL 回归。** 表中的秒数和 GiB 是人工哨兵，没有性能、显存收益含义。
- 尚未验证修复后 HCCL 的通信成本、真实 PP 调度下的 callback 行为或完整 200 步训练。维护者评审和 CI 状态不能由本次 CPU 结果替代。
- v1 初次观察也完整保留。独立检视要求进一步核对基点组内 oracle，并要求超时/worker 失败先保存退出码；v2 采纳后重新运行，最终验收不使用 v1 的宽松门禁。

## 可复核证据

- [最终修复前汇总](raw/evidence/pr226-gloo-8cpu-v2/base-summary.json) · [最终修复后汇总](raw/evidence/pr226-gloo-8cpu-v2/head-summary.json)
- [最终探针](raw/evidence/pr226-gloo-8cpu-v2/check_resource_metrics_gloo.py) · [执行入口](raw/evidence/pr226-gloo-8cpu-v2/run_resource_metrics_gloo.sh)
- [修复前 rank0 原始记录](raw/evidence/pr226-gloo-8cpu-v2/base-rank0.json) · [修复后 rank0 原始记录](raw/evidence/pr226-gloo-8cpu-v2/head-rank0.json)；同目录保留其余七个 rank。
- [源版本、归档及最终脚本指纹](raw/evidence/pr226-gloo-v2-deployment.json) · [无设备容器退出/移除凭据](raw/evidence/pr226-gloo-8cpu-v2.receipt.json)
- [58 份原始文件清单](pr226-gloo-evidence-manifest.json) · [公开逐字节副本清单](public-manifest.json)
- [需求轴独立复算](pr226-gloo-v2-spec-independent-review.json) · [规范及证据边界独立审计](pr226-gloo-v2-standards-independent-review.json)

注意容器 receipt 中 `source_commit=18602c4` 描述外层默认挂载的任务集成树；本次被测代码实际通过独立 editable 指向 `/work/tmp/pr226-gloo-source/base` 或 `head`。各 worker 的 `source_root`、回调 SHA 和精确归档清单共同证明被测版本，不能将 receipt 的默认挂载误写为被测 PR 源码。

关联：[小问题 #225](https://github.com/hf1s/hyper-parallel/issues/225) · [独立单提交 PR #226](https://github.com/hf1s/hyper-parallel/pull/226) · [中文总跟踪 #202](https://github.com/hf1s/hyper-parallel/issues/202)。
