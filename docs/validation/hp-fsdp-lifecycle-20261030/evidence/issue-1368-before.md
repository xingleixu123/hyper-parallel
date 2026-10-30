## 草稿状态与问题

嵌套 FSDP 子模块只参与较早的 microbatch，且使用 `set_requires_gradient_sync(False)` 累积梯度时，重新开启同步后，末次 root backward callback 会因该子模块仍为 `BACKWARD` 而跳过。结果是该参数的 FP32 `main_grad=None`，未分片累积梯度留在 buffer 中，不能参与本次优化更新。

基于官方 master `5cde7c54ce3da8be8689d0783cf9b70d5306117a` 的完整 2659 文件源树，真实两 rank CPU/Gloo 已复现；此前冻结 1a 也有同一反例。与 [FSDP 重构 RFC #229](https://github.com/mindspore-ai/hyper-parallel/issues/229) 的状态机/根回调收尾讨论相关，本单给出一个独立、可执行的条件参与回归。

## 最小触发与数学依据

两 rank、两个 microbatch、共同模块 common 和可选模块 early；两边实际初始权重逐 shard 相等。micro0 两模块均参与；micro1 只调用 common。输入保持确定且逐项记录，使用 FP32 main-grad 和显式 Gloo SUM。

对 `loss=(X@W).sum()`，有 `d(loss)/dW[i,j]=sum(X[i])`。两 rank 仅 micro0 的 early 正确完整梯度各行应为 `[3,2,3,4]`（每行四列相等），对应 rank0 的两行为 3、2，rank1 为 3、4。原实现两 rank 的 early 主梯度均为空，本地梯度仍在 pending；common 和其他三个控制组符合独立普通 PyTorch 图。独立直接导数复算也得到同一结果。

## 根因与最小修复

上轮 no-sync 的 post-backward 已执行，状态设为 BACKWARD，但因同步关闭，只保存了未分片累积梯度。该单元缺席末微批时，不会经过新的 forward/backward hook 更新状态；根回调中的去重早退因而把“上轮 post 已运行”误当成本轮无需处理。

只在重新开启同步的 False→True 转换、状态仍为 BACKWARD、并确有 `unsharded_accumulated_grad` 时，重置该单元的旧状态，使原有 root fallback 可消费梯度。保留 buffer、既有 RS/AR、精度、梯度归一化和算子路径。没有在已排队的本轮归约期间仅凭 pending 重新发通信。

## 实际验证和复现

https://github.com/xingleixu123/hyper-parallel/tree/b773c2cd221c172759b96537c33e829c486627d4/docs/validation/hp-fsdp-nosync-20261030

- Python 3.12.14、Torch 2.12.0+cpu，PTA metadata 2.12.0.post3；CPU 容器没有 NPU 设备授权。
- 两种 `comm_fusion=False/True` 各自同脚本红绿对照：原实现每种 **6/8 rank-case 通过**，失败均为两 rank 的 micro0-only/no-sync；补丁每种 **8/8 通过**。共32份原始 rank-case JSON，输入、loss、初态和 oracle 配对一致。源清单运行前后全量核验，green 仅只读覆盖 scheduler。
- 上库精确新 worker：原实现按预期在首个反例失败；补丁下两 rank 均通过完整2 fusion×4 controls。该 pytest worker 的 red 首错退出，不把诊断脚本的 red fusion=True 冒充为其执行结果。
- 复用现有 hook-state normal、non-reentrant、reentrant 三场景均通过；轻量 launcher 导入未加载 Torch/HP，现有 torchrun_case 的实际两 rank 调用通过。
- 独立同环境导入探针核实 HP、fixture、scheduler 及 editable/.pth 来源；该探针是独立进程，原 ST 本身没有打印来源字段。

- 最终 launcher `391848…` 仅补充 pytest MonkeyPatch 类型导入及参数/返回注解，已再次通过轻量 import smoke、外层 pytest 和两 rank helper；[追加原始证据](https://github.com/xingleixu123/hyper-parallel/tree/b773c2cd221c172759b96537c33e829c486627d4/docs/validation/hp-fsdp-nosync-20261030/typing-update-v1)。原 32 份诊断与旧 launcher 测试均保留原执行版本范围。

补丁仓内复现：
```bash
python -m pytest -q tests/torch/fully_shard/test_fully_shard_gradient_accumulation.py
```

## 范围与后续

覆盖标准顺序微批、同 rank 参与顺序、完整 no-sync 的 FSDP 合成模型；扩展四rank HSDP2×2对照，两种fusion各原8/16、补丁12/16，完整no-sync修复通过，partial-only仍四rank失败。该已RS分片梯度收尾缺口已[关联现有RFC1105及64份原始记录](https://github.com/mindspore-ai/hyper-parallel/issues/1105#issuecomment-6516038348)，尚未修复；不另建重复issue。跨rank不同条件路径和backward中切换策略也不据此宣称支持。两rank NPU/HCCL合成验证已补充；完整模型性能、显存仍未由本问题修复验证。

此问题在评估 TP2/DP2 两微批同步成本时发现；尚无证据表明固定 Xing40 的真实批次触发该条件分支，不能用该反例解释既有 TP 耗时，也未报告提速。训练 YAML 和真实数据保持私有。关联[性能总跟踪 #304](https://github.com/hf1s/hyper-parallel/issues/304)；问题及 PR 保持草稿。


修复草稿：[fix(fully_shard): 归约未参与末微批模块的累积梯度](https://github.com/mindspore-ai/hyper-parallel/pull/1369)。

## NPU补充（固定5c，2026-10-30）

实际 Ascend910B3、两 rank HCCL、1D分片合成模型，采用 FP32 SUM，两种 fusion（zero-copy 均关闭）：原 scheduler12/16、唯一 scheduler补丁16/16。原头 early仅micro0参与且首micro no-sync 时4份rank-case缺 main_grad并遗留pending；修复后逐值符合独立有理数SUM推导并清空pending。32份原始case、4份rank terminal及两版worker/数学复核已[公开归档](https://github.com/xingleixu123/hyper-parallel/tree/ad968fd9c6bd60c74ebf501bbecd65ce42031488/docs/validation/hp-fsdp-nosync-20261030/npu-hccl-20261030/PUBLICATION.md)，55文件均按Git blob逐字节回读。红臂入口退出1，elastic随后报告rank1 SIGTERM；绿臂两terminal无失败、入口退出0。

先前green-v1在preimport等待180秒超时，无数值case；green-v2仅将worker gate TTL180→600，其他字节与计算AST不变。两次有效运行间另停止三个旧闲置任务容器，因此不能作为性能因果A/B或设备恢复的单因素证明。本证据固定5c+sole scheduler51abc；新master8b187eca的参数view修复正独立审查，不能把本结果改称8b测试。整网、HSDP partial-only、跨rank不同分支和默认zero-copy仍不由此证明。

<!-- HP-NOSYNC-SUPPLEMENT-20261030-START -->
### 补充验证（草稿，2026-10-30）

- **固定上游模块补证**：官方 `e0cdf62c` + 唯一 scheduler `51abc1d5…`，真实双 rank Ascend910B3 / CANN9.2.0 / PTA2.12.0.post3，FP32 SUM、两种 fusion、zero-copy 关闭；两微批均参与和仅第一微批参与的同步/延迟同步组合，独立有理数推导核权重、输入、loss、main_grad、pending，16/16 rank-case、2/2 terminal通过。[原件与复算](<https://github.com/xingleixu123/hyper-parallel/blob/073880e01a62e7eab5ebf08a5b3385225129a78d/docs/validation/hp-fsdp-nosync-20261030/upstream-e0-npu/PUBLICATION.md>)。这是 e0 绿臂；旧5c红绿来源不改标，未覆盖 HSDP partial-only 或全部 zero-copy 情形。
- **整网最小实验**：固定 `1a39572f` + 同一唯一 scheduler overlay，无 Muon overlay；同一 native1500 训练状态恢复、TP2/DP2、GBS4/LBS1、S4096、原2500步LR，两臂实际样本及初态一致，各执行1501。新完整CPU核对804项跨臂、201项逻辑覆盖，原生144张量/680byte条目/258角色、1152项初态到更新后比较，无非有限或缺失。[聚合、来源与复现边界](<https://github.com/xingleixu123/hyper-parallel/blob/073880e01a62e7eab5ebf08a5b3385225129a78d/docs/validation/hp-fsdp-nosync-20261030/tp-single-update/PUBLICATION.md>)。
- 第一次CPU尾部断言把所有 live optimizer slot 强制副本同SHA，真实差异集中 Muon momentum。源码 owner-only 更新、后续零更新NPU实际cache及固定生命周期条件审阅支持有界owner关联；非owner须保持初值，其他副本规则仍严格，未修改训练算术。该观察不是旧进程缓存录像。近零 `branch_alpha` 参数位不变有SHA约束的普通AdamW舍入解释；诊断轨迹不作为固定算术或最终方案。
- `compared` 表示观测完整，**不等于最终验收**。同期 sync40 已正常完成（1511–1540逐步max-rank本地耗时均值2.017419s、名义8121.268 tokens/s、peak allocated约19.8795GiB）；延迟同步臂已按新258角色审计和完整初态/样本核验，于1501开始连续到2500，首40同口径、后960继续。当前尚无其完整千步或同期收益结论；保留原 `abs(mean(待测−基线))≤0.005` 完整窗口口径，MTP按YAML关闭/N/A。TP+full AC+deferred对原1a FSDP/selective的千步比较属于组合语义，不作no-sync单因素归因。

补充仅更新证据说明；PR代码head保持 `b5839efa`，继续Draft。实际权重、slot样本、完整RNG、训练样本和YAML没有上传公开包。已公开89文件均逐字节回读，既有证据与包外文件保留。
<!-- HP-NOSYNC-SUPPLEMENT-20261030-END -->

