# FSDP8 选择性重计算：完整 200 步验收

`r129-f637-fp32-fsdp8-selective-200-r3` 完成独立的 1–200 步、全部八个 rank，退出 0，运行 394.861022 秒，设备已释放且没有本任务残留 PID。该轮从初态独立运行，不拼接前两轮 109/48 步前缀。

与关闭重算的 `r129-f637-fp32-fsdp8-0-200-r4` 比较，全部 rank、全部步骤的 LM、日志加权 MTP/aux、总 loss 和梯度范数最大绝对差均为 **0**；将 MTP/aux 按实际 0.3/0.0001 系数还原后，最大差仍为 **0**。输入身份不一致为 0，初态与归一化数值配方检查无差异，完整性门禁通过。

MTP/aux 还原值来自日志除以已记录系数，并非额外采集的加权前张量；梯度一致性结论限于范数，不宣称每个梯度元素或最终模型权重完全相同。

![完整200步对齐图](../../figures/f637-fsdp8-selective-r3-200-raw-zh.png)

## 关闭、完全与选择重算的资源比较

| 模式 | 中位步时（秒） | allocated 峰值（GiB） | reserved 峰值（GiB） | 有效监督 token/s |
| --- | ---: | ---: | ---: | ---: |
| 关闭 | 1.155003 | 22.032196 | 27.250000 | 16008.407 |
| 完全 | 1.235180 | 19.392933 | 24.320313 | 14969.751 |
| 选择 | 1.232111 | 19.875641 | 24.789063 | 15011.921 |

选择重算相对关闭：allocated 减少 **2.156555 GiB（9.7882%）**，reserved 减少 2.460938 GiB，中位步时增加 **6.6760%**，有效监督吞吐下降 6.2248%。完全重算比选择重算再少用 0.482708 GiB allocated；本次中位步时仅相差约 0.2491%。

这符合重放计算换取显存的总体方向；在当前两主干层加一 MTP 层的小模型配置下，选择重算尚未显示明显优于完全重算的收益。这里只是每种模式各一次匹配观测，0.2491% 的时间差没有重复实验置信区间，不能据此宣称稳定速度优势，也不能外推完整大模型。当前配方的模块消融仍待补齐，旧配方的模块结果另列。

三轮均在 129、相同源码/镜像/数据/数值配方运行，并保留同样的输入身份和逐步指标采集。每步取全部八 rank 已发布耗时的最大值；时间排除前 20 步，吞吐为剩余有效监督 token 总数除以相同步耗时总和。显存取全部 200 步、全部 rank 的累计峰值，包括预热。该口径不是无插桩 Trainer 性能或包含启动、保存等阶段的端到端墙钟时间。

## 源码、配置与资源边界

- `hf1s:JT_dev@f637ef7`，包含 #180；固定修复集成树 `18602c4524dc664b81b2bad9a7b624455ce1494f`，工作区补丁为空。
- CANN 9.2.0-beta.2，固定镜像 `sha256:5f9e677f439d11ab4e7c888f54b142335a27eee7411cbd2ce894a4798a70146d`，Torch/TorchNPU 2.12.0。
- 两主干解码层、一 MTP 层、八专家/top-k 2，hidden 5120、序列长 4096、GBS 8/MBS 1、seed 42、常量 LR 0.0001；专家 wgrad 与 Muon NS 使用 FP32，并非全模型 FP32。
- 最终配置明确启用 `activation_checkpoint.mode=selective`，构建清单包含两个动态派生的 HSDPCheckpointWrapper 和一个 CheckpointWrapper。旧报告漏计动态派生 wrapper 属于独立问题 #229，不能用旧计数推断本轮只覆盖一个区域。
- 启动前先进行了空闲采样，再由原资源守卫两次检查当前八卡空闲。手动采样接续处最大间隔约 94.565 秒，未将其冒充自动观察器满足 90 秒间隔约束的 ready 结果，也不宣称采样间持续空闲。
- 全局 20 分钟间隔满足；本轮未触发外部资源冲突，退出后卡上无本任务进程。下一轮最早为 **2026-10-16 03:57:52.019057（UTC+8）**，还须重新检查空闲。
- 76 份原始文件已逐一核对大小和 SHA，重复传输压缩包已在本地、129 清理。当前 step200 续训 checkpoint 保留。本轮不保存额外 checkpoint。

## 证据与关联

- [严格比较](f637-fsdp8-selective-r3-strict-comparison.json) · [数值和资源摘要](f637-fsdp8-selective-r3-complete-analysis.json)
- [逐步全 rank 资源](f637-fsdp8-selective-r3-all-rank-resources.json) · [关闭参考](f637-fsdp8-r4-all-rank-resources.json) · [完全重算参考](f637-fsdp8-full-200-all-rank-resources.json)
- [数值独立审计](f637-fsdp8-selective-r3-precision-independent-review.json) · [资源与释放独立审计](f637-fsdp8-selective-r3-resource-independent-review.json)
- [原始文件清单](f637-fsdp8-selective-r3-complete-evidence-manifest.json) · [实际配置](raw/runs/r129-f637-fp32-fsdp8-selective-200-r3/config.yaml) · [退出及释放记录](raw/scheduling/r129-f637-fp32-fsdp8-selective-200-r3.receipt.json)
- [启动前采样说明](selective-r3-prelaunch-idle-observations.json) · [释放后全局协调器状态](f637-fsdp8-selective-r3-coordinator-state.json)

总跟踪 [#202](https://github.com/hf1s/hyper-parallel/issues/202)。相关独立问题 [#206](https://github.com/hf1s/hyper-parallel/issues/206) / [PR #195](https://github.com/hf1s/hyper-parallel/pull/195)、[#207](https://github.com/hf1s/hyper-parallel/issues/207) / [PR #196](https://github.com/hf1s/hyper-parallel/pull/196)。本结果属于固定修复集成树，不将全部精度或资源收益归因于其中单个 PR；EP/CP 的当前 200 步精度失败仍然保留。
