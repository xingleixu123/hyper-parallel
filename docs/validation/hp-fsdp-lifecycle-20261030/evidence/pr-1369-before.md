**What type of PR is this?**
/kind bug

**What does this PR do / why do we need it**:

嵌套 FSDP 单元仅在较早微批产生梯度时，上轮 no-sync 的 BACKWARD 状态会让末微批根回调跳过该单元，导致 main_grad 为空且梯度遗留。重新开启同步并存在真实未分片累积梯度时，重置该单元的旧状态，使既有根回调完成归约；不清 buffer、不重发已排队通信、不改变既有数学或精度策略。

基于官方 master 5cde7c54，单提交、三个文件：scheduler 最小修复，以及一个真实两 rank CPU/Gloo 回归 worker 与不导入框架的轻量 launcher。关联 https://github.com/mindspore-ai/hyper-parallel/issues/1368 和 [状态机 RFC #229](https://github.com/mindspore-ai/hyper-parallel/issues/229)。

**Test Plan and Test result：What scenarios were tested, and what were the verification results（Function, performance, reliability, etc.）**：
- 同一诊断脚本，两种 comm_fusion 原实现各6/8、补丁各8/8；32份真实 rank-case JSON，原始权重/输入/loss/oracle 配对一致。
- 精确上库 worker 红绿通过；green 完整2 fusion×4 controls。复用原 hook-state normal/non-reentrant/reentrant 三场景通过，launcher 实际 helper 与轻量 import smoke 通过。
- 官方2659源清单前后核验，green 仅 scheduler 只读覆盖；Python3.12.14/Torch2.12.0+cpu，CPU无NPU设备授权。独立导入探针核对 actual editable/source，原ST未打印origin。
- https://github.com/xingleixu123/hyper-parallel/tree/b773c2cd221c172759b96537c33e829c486627d4/docs/validation/hp-fsdp-nosync-20261030

- 最终 launcher `391848…` 仅补充 pytest MonkeyPatch 类型导入及参数/返回注解，已再次通过轻量 import smoke、外层 pytest 和两 rank helper；[追加原始证据](https://github.com/xingleixu123/hyper-parallel/tree/b773c2cd221c172759b96537c33e829c486627d4/docs/validation/hp-fsdp-nosync-20261030/typing-update-v1)。原 32 份诊断与旧 launcher 测试均保留原执行版本范围。

两rank NPU/HCCL合成验证已完成（见下方补充）；真实整网性能/显存尚未由本PR验证。四rank HSDP2×2对照已验证完整no-sync修复；partial-only仍失败，已[关联现有RFC1105](https://github.com/mindspore-ai/hyper-parallel/issues/1105#issuecomment-6516038348)，不在本PR已修复范围。本 PR 保持 Draft；该修复不构成已记录 TP 耗时的归因或提速结论。

**NPU补充（固定5c，2026-10-30）**：

实际 Ascend910B3、两 rank HCCL、1D分片合成模型，采用 FP32 SUM，两种 fusion（zero-copy 均关闭）：原 scheduler12/16、唯一 scheduler补丁16/16。原头 early仅micro0参与且首micro no-sync 时4份rank-case缺 main_grad并遗留pending；修复后逐值符合独立有理数SUM推导并清空pending。32份原始case、4份rank terminal及两版worker/数学复核已[公开归档](https://github.com/xingleixu123/hyper-parallel/tree/ad968fd9c6bd60c74ebf501bbecd65ce42031488/docs/validation/hp-fsdp-nosync-20261030/npu-hccl-20261030/PUBLICATION.md)，55文件均按Git blob逐字节回读。红臂入口退出1，elastic随后报告rank1 SIGTERM；绿臂两terminal无失败、入口退出0。

先前green-v1在preimport等待180秒超时，无数值case；green-v2仅将worker gate TTL180→600，其他字节与计算AST不变。两次有效运行间另停止三个旧闲置任务容器，因此不能作为性能因果A/B或设备恢复的单因素证明。本证据固定5c+sole scheduler51abc；新master8b187eca的参数view修复正独立审查，不能把本结果改称8b测试。整网、HSDP partial-only、跨rank不同分支和默认zero-copy仍不由此证明。

**Which issue(s) this PR fixes**:

Fixes #1368


**Self-checklist**：

- [ ] **设计**：Maintainer 尚未审阅，按要求保持 Draft。
- [x] **测试**：新增真实 Gloo ST 随本 PR；原始及修复版本、既有重计算轨迹已实跑。
- [x] **验证**：CPU 模块目标与限制在上文逐项说明；两rank NPU合成已验；整网待验证。
- [x] **接口**：保持对外签名，修复既有同步开关的梯度语义。
- [x] **文档**：本次无需官网文档变更；复现命令与原始记录单独归档。

<!-- HP-NOSYNC-SUPPLEMENT-20261030-START -->
### 补充验证（草稿，2026-10-30）

- **固定上游模块补证**：官方 `e0cdf62c` + 唯一 scheduler `51abc1d5…`，真实双 rank Ascend910B3 / CANN9.2.0 / PTA2.12.0.post3，FP32 SUM、两种 fusion、zero-copy 关闭；两微批均参与和仅第一微批参与的同步/延迟同步组合，独立有理数推导核权重、输入、loss、main_grad、pending，16/16 rank-case、2/2 terminal通过。[原件与复算](<https://github.com/xingleixu123/hyper-parallel/blob/073880e01a62e7eab5ebf08a5b3385225129a78d/docs/validation/hp-fsdp-nosync-20261030/upstream-e0-npu/PUBLICATION.md>)。这是 e0 绿臂；旧5c红绿来源不改标，未覆盖 HSDP partial-only 或全部 zero-copy 情形。
- **整网最小实验**：固定 `1a39572f` + 同一唯一 scheduler overlay，无 Muon overlay；同一 native1500 训练状态恢复、TP2/DP2、GBS4/LBS1、S4096、原2500步LR，两臂实际样本及初态一致，各执行1501。新完整CPU核对804项跨臂、201项逻辑覆盖，原生144张量/680byte条目/258角色、1152项初态到更新后比较，无非有限或缺失。[聚合、来源与复现边界](<https://github.com/xingleixu123/hyper-parallel/blob/073880e01a62e7eab5ebf08a5b3385225129a78d/docs/validation/hp-fsdp-nosync-20261030/tp-single-update/PUBLICATION.md>)。
- 第一次CPU尾部断言把所有 live optimizer slot 强制副本同SHA，真实差异集中 Muon momentum。源码 owner-only 更新、后续零更新NPU实际cache及固定生命周期条件审阅支持有界owner关联；非owner须保持初值，其他副本规则仍严格，未修改训练算术。该观察不是旧进程缓存录像。近零 `branch_alpha` 参数位不变有SHA约束的普通AdamW舍入解释；诊断轨迹不作为固定算术或最终方案。
- `compared` 表示观测完整，**不等于最终验收**。同期 sync40 已正常完成（1511–1540逐步max-rank本地耗时均值2.017419s、名义8121.268 tokens/s、peak allocated约19.8795GiB）；延迟同步臂已按新258角色审计和完整初态/样本核验，于1501开始连续到2500，首40同口径、后960继续。当前尚无其完整千步或同期收益结论；保留原 `abs(mean(待测−基线))≤0.005` 完整窗口口径，MTP按YAML关闭/N/A。TP+full AC+deferred对原1a FSDP/selective的千步比较属于组合语义，不作no-sync单因素归因。

补充仅更新证据说明；PR代码head保持 `b5839efa`，继续Draft。实际权重、slot样本、完整RNG、训练样本和YAML没有上传公开包。已公开89文件均逐字节回读，既有证据与包外文件保留。
<!-- HP-NOSYNC-SUPPLEMENT-20261030-END -->

