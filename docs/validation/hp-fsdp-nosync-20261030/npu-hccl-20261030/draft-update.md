# 拟追加文字（仅供 root 审核，尚未发布）

## Issue #1368

补充两卡 NPU 功能验证：官方5c源码的两 rank HCCL、1D分片合成模型采用 FP32、SUM，分别验证 fusion 关闭/开启（zero-copy 均关闭）。末微批不再经过 early 子模块且首微批 no-sync 时，原 scheduler 两 rank×两 fusion 均出现 early `main_grad` 缺失并保留待归约梯度；应用唯一 scheduler 补丁后清空 pending，逐 rank 梯度与独立 SUM 推导精确一致。原头12/16、补丁16/16；完整32份 case 和四份 rank terminal 已归档（固定公开证据链接待发布）。红臂入口退出1；虽两份 terminal 均已写，elastic 后续报告 rank1 SIGTERM/−15。

前一次 green-v1 在180秒 preimport 等待时超时，未产生数值 case；green-v2 将等待改为600秒，worker 其余字节不变。两次有效结果之间还停止了三个旧闲置任务容器，不能据此比较性能或把可用性归因于某一变化。此结果仅覆盖两 rank 合成模型，不覆盖真实整网、HSDP 2×2 partial-only 或 zero-copy 默认路径；case JSON 自身也不提供 collective trace 或内部融合桶布局。

## Draft PR #1369

补充实际 Ascend910B3 两 rank HCCL 合成诊断：5c 原 scheduler 在 no-sync 后末微批缺席分支出现4/16失败；仅叠加本 PR scheduler 补丁后16/16通过，两种 fusion（zero-copy 关闭）和全局 FP32 SUM 的独立逐值推导均覆盖。原头有完整16个 case/两个 rank terminal 后入口退出1，补丁臂两个 terminal 无失败且入口退出0；worker 字节仅 gate 等待180→600秒，期间三个旧闲置容器停止，故不作为性能对照。green-v1 先前在 preimport 门等待超时，无数值结果（固定公开证据链接待发布）。HSDP partial-only、整网和默认 zero-copy 仍未由此测试验证。
