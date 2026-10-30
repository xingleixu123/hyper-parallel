# 两卡 NPU：no-sync 末微批缺席的合成诊断

本目录是待审核的公开补充材料，本地候选，**尚未发布**。在两张 Ascend 910B3 上，以公开 `fully_shard` 接口运行两 rank、1D mesh、HCCL、FP32 参数和 main gradient、SUM 的小型合成模型；两个 fusion 设置都显式 `comm_fusion_zero_copy=False`。这不是完整模型数值验收、HSDP 2×2 partial-only 验证或性能测试。

| 两 rank × 两 fusion，逐 case 汇总 | 原 5c | 仅 scheduler 补丁 |
|---|---:|---:|
| both_true | 4/4 通过 | 4/4 通过 |
| both_false | 4/4 通过 | 4/4 通过 |
| micro0_only_true | 4/4 通过 | 4/4 通过 |
| micro0_only_false | 0/4 通过 | 4/4 通过 |
| 合计 | **12/16** | **16/16** |

`micro0_only_false` 表示 early 子模块仅参与第一个微批，第一微批不做梯度同步、最终微批恢复同步；红臂四个 rank×fusion 结果均记录 early `main_grad=None` 与未归约的 `pending_unsharded`，绿臂则得到独立 SUM 预期且 pending 清空。原头红臂两个 rank terminal 均已写出并记录失败，torchrun 总退出1；之后 elastic 对 rank1 记 SIGTERM/−15，不能说两个红 rank 都正常退出。绿臂两个 terminal 均记录无失败，入口退出0。

权重和输入完全为合成值。对 `sum(xW)` 独立用有理数计算，首微批全 rank 输入和是 `[3,2,3,4]`，两微批和是 `[6,6,8,8]`；按参数行维度由 rank0/1 分别持有前/后两行。`math/verify_toy.py` 不读取 worker 的 oracle 作为预期，而从模型定义独立计算初权、输入、loss、梯度与 pending，并检查全部32份原始 rank-case；纯 CPU 执行：

```bash
python3 -B math/verify_toy.py
```

两个 `math/*-independent-sanitized.json` 是原有独立数学复核的去标识派生摘要，保留逐条数值和源文件 SHA，排除实际 PID、namespace、映射原始记录、完整环境及私有路径。`math/offline-recheck.json` 是以上脚本对本目录公开数据的重新计算。[红臂](results/red/)和[绿臂](results/green/)各含16份原始 rank-case 与两份 terminal，按原字节保留，SHA 与原闭合清单一致；闭合清单本身含内部路径，未公开，见 [closure-summary.json](closure-summary.json)。

源码固定官方 `5cde7c54ce3da8be8689d0783cf9b70d5306117a`，2659 文件的 [manifest](source/source-manifest.json) SHA256 `a06da217316eff5f307898e8822c6f98b21ac7e1117a5b326e38b60876460df6`；红臂 scheduler SHA `2d311ed2ee490370118100d457dfbfe4f773c7d2ba5997f180238a10a1ad4cdd`，绿臂仅叠加 scheduler SHA `51abc1d50805960b50c334c8f5a5b5c4a5b2a5e0d7984686bae3d2e553499fe2`，增量见 [scheduler.patch](source/scheduler.patch)。完整源树前后审计和单文件只读 overlay 依据原运行收据；公开原始 result 中每项还记录关键源 pin。

红臂使用 [worker v1](workers/worker-red-v1.py)，绿臂使用 [worker v2](workers/worker-green-v2.py)：唯一字节差异是 preimport/mapping release 等待从180秒延长到600秒，见[diff](workers/worker-v1-to-v2.diff)与[AST校验](workers/worker-ast-verification.json)。两个 worker 使用相同数学模型、oracle、fusion 和公共调用；在原始测试期间，已有三个其他旧任务闲置容器被停止，等待时长也改变，**不能据此作速度或其他性能因果 A/B**。

最初 green-v1 在180秒 preimport release 等待期间退出，未进入 rank 数值 case；该次不是绿色结果，其闭合 SHA 和状态保存在 [lineage-summary.json](lineage-summary.json)。之后 green-v2 以新运行身份完成上述结果，不将前次超时说成数值通过或补丁失败。

运行收据记载 CANN 9.2.0、Python 3.12.14、PyTorch 2.12.0+cpu、PTA `torch_npu` 2.12.0.post3、物理卡0/1和 Ascend910B3。映射是原独立 root 实测的公开摘要；进程身份、环境和设备映射原始材料保留在私有本机归档。红臂首次 preimport 空闲检查失败但未保留原始 SMI，不推断当时有占用 PID；其后 root 依据新观察放行。固定 worker 在写 case 前要求 HCCL 初始化成功并检查 rank/world/backend；case JSON 自身不记录独立 PG 成员表、融合包布局或 collective trace，不能据此宣称验证了这些内部细节。

要复现设备诊断，使用独立干净的上述源码及唯一补丁、匹配运行时、两张已授权设备，按 worker 的固定路径与两次 root 审核 gate 提供源码清单及放行文件。离线脚本只重算公开 toy 数据，不能代替设备运行或证明原始收据；设备测试需要独立物理 PID 映射和安全上下文审阅。

文件逐项 SHA256 见 [SHA256SUMS](SHA256SUMS)，其清单不包含自身。
