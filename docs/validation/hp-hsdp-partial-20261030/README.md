# 四 rank HSDP 延迟同步：CPU 合成证据

本包记录四次独立 CPU/Gloo 运行，每次为 4 rank × 4 case，共 64 份 rank/case 合成记录。它们不是 64 次训练，也不证明真实模型、NPU/HCCL 或性能结论。

两个 4×4 FP32 参数组成嵌套单元：common 在两个 micro 都参与，early 按 case 在两个 micro 或仅第一个 micro 参与。对第一个 micro，suffix false 表示完整 no-sync；suffix true 的记录显示 reduce_grads 启用、replica all-reduce 禁用，且 fusion=false 的分片 partial 数值与 shard reduce-scatter 的预期一致。最后一个 micro 恢复同步开关。主梯度是在裁剪及任何 optimizer 更新之前检查。二种通信融合开关分别独立运行；固定 SUM 是诊断用约定，不能当作生产默认归约模式。

| 源码臂 | fusion | 通过 / 结果 | 非零退出 |
|---|---|---:|---:|
| red：固定原始 scheduler | false | 8 / 16 | 1 |
| red：固定原始 scheduler | true | 8 / 16 | 1 |
| green：单文件候选 patch | false | 12 / 16 | 1 |
| green：单文件候选 patch | true | 12 / 16 | 1 |

四组均完整写出 16 份结果。red 中 early 缺席末 micro 时，两种同步方式在四 rank 均有 main_grad None 和未清空 pending；green 的完整 no-sync 情况恢复正确，但 replica-partial 情况在四 rank 仍未交付梯度。两个 micro 都参与的控制在两臂及两种 fusion 均通过。退出码 1 是诊断捕获失败后的预期终态，不是成功退出。终态、逐 case 状态和原始输出见 [runs](runs/)；来源及复制核验见 [copy-audit.json](copy-audit.json)。

[独立数学复核](independent-math.json)不调用 worker 自带 oracle：对 loss=sum(xW)，每个参与分支的 dL/dW[i,j]=x[i]。micro0 四 rank 行和为 [10,4,10,8]，两 micro 的行和为 [20,12,24,16]。行主序 2×2 mesh 的 shard 组 [0,1] 与 [2,3] 在 micro0 分别得到 [3,2,3,4] 与 [7,2,7,4]；按第 0 维切行后，replica 组 [0,2]、[1,3] 应分别相加得到 [10,4]、[10,8]，每行四列。fusion=false 的逐参数 partial 原值与此一致；fusion=true 只记录 bucket 的 FQN、dtype、形状及哈希，无法仅由哈希确定其内部布局。两种 fusion 都没有 collective 调用 trace；因此证据支持“归约结果未交付”，不能单凭此确定 all-reduce 的具体调用路径。

## 固定来源

原始源版本为 HyperParallel commit 5cde7c54ce3da8be8689d0783cf9b70d5306117a；四次运行对 2659 文件的完整基线在运行前后核验一致。完整源清单不含在本包，因此外部读者不能仅凭本包重算该 2659 文件清单。每臂关键 15 项文件 hash 原样保留在 [red pins](diagnostic/red/source-and-config-pins.json) 和 [green pins](diagnostic/green/source-and-config-pins.json)，唯一差异为 scheduler。原 worker 字节 SHA-256 为 14e249a3565d17232d68384200abc00121f69f53f7802f818cf031466520e8e2；[补丁](hsdp-scheduler.patch)在本地以内存重建过原 scheduler 和 green 版本的字节匹配。

原始镜像 digest、Python 3.12.14 的固定路径、worker 实际报告的 torch 2.12.0+cpu、Gloo、环境和精确四 rank worker 命令见 [复现说明](REPRODUCE.md)。torch_npu 2.12.0.post3 是同主机、同 VENV 的另次独立进程记录；该诊断 worker 没有在其进程内直接报告该版本。环境目录及镜像本身未随包分发；需要先取得相同依赖。原始 stdout 有驱动管理日志，但本次容器没有设备映射，不能据此推断 NPU 状态。

本包仅含合成结果、必要日志、轻量补丁、源文件 hash 和 worker；不包含真实 YAML、数据、checkpoint、完整源码或带 Base64 文件内容的 receipt。[SHA256SUMS](SHA256SUMS)覆盖除自身外的包内文件，复制审计逐项核对 raw JSON 与原始回执。

