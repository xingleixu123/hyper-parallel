# 固定 e0 + 单文件 scheduler 补丁：双 rank NPU 合成回归

本目录是**未发布的公开证据候选**。实际 Ascend910B3、两 rank、1D 分片、HCCL、FP32 参数与 main gradient、SUM：两种 fusion、zero-copy 关闭，两个 micro 的四种参与/同步组合，合计 **16/16 rank-case 通过**。两个 rank terminal 都记录无失败，执行进程 exit 0。这不是全模型恢复或精度验收，也不是性能实验。

## 来源、执行与闭合

- [官方固定源码 e0cdf62c9d66f1f9ec48c6f15df87dcf4ce80a5f](https://github.com/mindspore-ai/hyper-parallel/commit/e0cdf62c9d66f1f9ec48c6f15df87dcf4ce80a5f)，2676 文件的原清单 SHA256 `2ee3981b6643d6ffa27c3e670bf1c0fca7500379d436874af0a761e86d3b242a`。原 `hsdp_scheduler.py` SHA `2d311ed2ee490370118100d457dfbfe4f773c7d2ba5997f180238a10a1ad4cdd`；唯一只读覆盖为 SHA `51abc1d50805960b50c334c8f5a5b5c4a5b2a5e0d7984686bae3d2e553499fe2`。固定 worker SHA `5bea6387d6ce46bad195fd6d13d1338d6877fe489c1ac54770865250d04fb00e`。
- [原 scheduler](source/official-e0-scheduler.py)、[补丁](source/scheduler.patch)、[实际覆盖文件](inputs-reviewed/inputs/patched_scheduler.py)、[源码清单](inputs-reviewed/inputs/source-manifest.json)与[实际 worker](inputs-reviewed/inputs/worker.py)均为可复核输入。离线校验器对两文件重新计算 diff 并与补丁逐字节比较。预检的全量源清单和唯一只读 overlay 配置见 [preflight](run/preflight.json)；所纳入的固定 [adapter](inputs-reviewed/adapter.py)显示闭合动作会再次调用全量源 guard，但闭合清单不单独保存全部 after 快照。两个 rank terminal 另保存 10 项关键源的 after pin。
- [闭合清单原件](run/closed-readback.json) SHA `0b50fe294108a46c188a4f78d2d6bafa2abcb3ac28a4d3a821e68ad312164408`，记载的 91 项在私有闭合归档中已全部逐大小和 SHA 核验。为减少冗余，公开候选保留其中 **56 项原字节**及清单本身：16 份 rank-case、2 份 rank terminal、必要的 preflight/门控/入口/终态材料，以及 8 次 SMI 捕获各自的 metadata/stdout/stderr；其余 35 项没有打包。离线脚本逐项检查本包 56 项与原清单一致，无法凭本包复验未纳入的 35 项。
- 父进程[终态原件](run/execute-terminal.json) SHA `1dbd02153a81bb5e32847aff715049be1bd2df86cd5e9679b9cba908895086a5`；[preimport release](run/output/preimport-release.json) 和 [physical mapping release](run/output/physical-mapping-release.json)分别由两个 rank terminal 绑定。preimport gate 在框架导入前；两 rank 完成逻辑设备初始化并核对实际物理映射后才进入 HCCL。全部门控文件及 SMI 是实测原字节，不是 mock。
- 观察时 physical 1 → rank 0，physical 2 → rank 1；实际 SMI 及捕获状态在 `run/smi-observe-*` 和 `run/smi-release-*`。`gate-observation-summary.json` 从未公开的完整 mapping observation 派生，记录真实 rank/PID/starttick 与安全上下文；源 observation SHA 与原始清单及 mapping release 相同，但此包不含其冗余完整进程映射，不能仅凭派生摘要独立重算源 observation 的 SHA。实际 rank 的 UID 0、Seccomp 0、CapEff `000001ffffffffff`；完整 14 项 cgroup 均记录，只有 devices 和 memory 与目标容器一致，不把它说成 Docker 默认安全上下文。
- Python 3.12.14、torch 2.12.0+cpu、torch_npu 2.12.0.post3、CANN 9.2.0。精确 worker argv 和虚拟环境源/版本引用在 [entry evidence](run/private/entry-evidence.json)；运行要求对应的只读源与 overlay、环境以及独立 gate，不能以离线重算替代设备运行。stderr 中 watchdog/执行超时设置 warning 等原文保留。

## 数学验证

`math/verify_toy.py` 用有理数直接推导并逐项检查 16 份原始 case 的权重、两次输入、loss、main_grad、pending、分片、fusion 与运行源码针，**不使用 worker 输出的 oracle 字段**；还检查两个 terminal、门控哈希及保留的原始 SMI。它会将重算结果与封存的 [独立统计原件](math/independent-original.json)逐项比较：

```bash
python3 -B math/verify_toy.py
```

模型是 `sum(x @ W_common + active × x @ W_early)`，`d(loss)/dW[i,j]=x[i]`。micro0 的两 rank SUM 行为 `[3,2,3,4]`，micro1 为 `[3,4,5,4]`，两者累计为 `[6,6,8,8]`；每行 4 列相同，rank 0/1 分别持有前/后两行。

| early 参与条件 | rank 0 两次 loss | rank 1 两次 loss | 最终 common 行（rank 0 / rank 1） | 最终 early 行（rank 0 / rank 1） |
|---|---|---|---|---|
| 两个 micro | 26.75 / 36.25 | 34.25 / 43.75 | `[6,6]` / `[8,8]` | `[6,6]` / `[8,8]` |
| 仅第一个 micro | 26.75 / 16.375 | 34.25 / 19.625 | `[6,6]` / `[8,8]` | `[3,2]` / `[3,4]` |

首 micro 关闭同步时，每 rank 的本地完整 FP32 梯度留在 pending；末 micro 恢复同步后，两个参数的 `main_grad` 与直接 SUM 精确一致，所记录 pending 为空。此结论覆盖 fusion false/true、首 micro 同步 true/false；记录的空 bucket 是采样时点的状态，不能证明内部 fusion 布局或每次 collective 轨迹。HCCL backend/world/rank 由固定 worker 在 case 之前断言，不是逐 collective trace。

## 边界

这是固定 **e0 + 唯一 scheduler 51** 的模块绿臂；本目录没有相同 e0 条件的原 scheduler 红臂。[先前独立证据](https://github.com/xingleixu123/hyper-parallel/tree/ad968fd9c6bd60c74ebf501bbecd65ce42031488/docs/validation/hp-fsdp-nosync-20261030/npu-hccl-20261030/PUBLICATION.md)固定官方 5c，不能把它重标为 e0 的红臂或据此作性能因果比较。此前 TP 完整模型单步和后续性能测试固定的是 **1a + scheduler 51**，与本模块 e0 源版本不同，不应合并成同源整网结论。HSDP 2×2 partial-only、跨 rank 不同分支和默认 zero-copy 也不在本测试覆盖范围。

公开候选未包含完整官方归档、真实 YAML、模型 checkpoint、训练样本或 phase 的完整 base64 传输；[privacy-validation.json](privacy-validation.json)记录机械扫描的范围和局限。文件逐项 SHA256 见 [SHA256SUMS](SHA256SUMS)，清单不含自身。Fast selector 不可用。