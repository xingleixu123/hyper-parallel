# JT DeepSeek-V3：当前结果与问题排序

任务仍在进行。FSDP 的 0→200、checkpoint200→400 已通过；跨拓扑精度仍有失败，不能宣布整体完成。所有后续正式运行均纳入用户指定的 [PR #180](https://github.com/hf1s/hyper-parallel/pull/180)。

## 按影响排序的问题

| Top | 问题与影响 | 当前证据与处理 |
|---|---|---|
| 1 | 跨 FSDP/TP/CP/EP 的 200 步精度仍超出 0.005 | 专家 BF16 权重梯度分区舍入和 Muon 更新敏感性已有独立算子及首步证据；TP 还有上游差异。FP32 专家梯度加完整 FP32 NS 的 EP2、CP2、TP2×EP2 长跑仍失败。生产补丁尚未通过完整矩阵，继续定位。 |
| 2 | 梯度累积时使用错误的有效 token 分母 | [PR #194](https://github.com/hf1s/hyper-parallel/pull/194)。改为整步有效 token 数。TP2+EP2 的 LM 最大偏差由 0.92238 降至 0.01248，仍需处理 Top1。CPU 重现与真实 200 步对照均已保留。 |
| 3 | FSDP 初始化后保留完整源参数，浪费大量显存 | [PR #198](https://github.com/hf1s/hyper-parallel/pull/198)。释放已经消费的初始化容器引用。EP2 节省 9.422 GiB，CP2 节省 10.125 GiB；各 200 步、8 rank 同拓扑 loss/梯度误差均为 0。约 0.1% 的步时变化视为噪声。 |
| 4 | CP 拒绝分块输出 loss，阻断低显存训练 | [PR #199](https://github.com/hf1s/hyper-parallel/pull/199)。实现全 CP 有效 token 分母，重放不重复 collective。4 卡 ST 通过；demo 的 CP2 已能完成 200 步，跨拓扑数值问题仍单独跟踪。 |
| 5 | full 重算未覆盖完整 MTP decoder | [PR #195](https://github.com/hf1s/hyper-parallel/pull/195)。补齐 trunk/MTP block 识别、幂等与 JT 重入路径约束。旧配置 loss 相同并不能证明重算覆盖正确。 |
| 6 | 重算重放再次提交 MoE 路由计数 | [PR #196](https://github.com/hf1s/hyper-parallel/pull/196)。保留张量计算，只禁止重放的持久统计更新。覆盖 full/selective、不同 loss 路径及多微批；修复后 FSDP full/selective 200 步误差均为 0。 |
| 7 | 续训后累计 token/sample 监控从零开始 | [PR #197](https://github.com/hf1s/hyper-parallel/pull/197)。保存并恢复 meter 累计量，兼容旧 checkpoint。真实续训 200 步、8 rank 的 loss/梯度/累计计数与连续参考均一致。 |

六个 PR 均为独立单提交，并经过 Standards / Spec 两轴检视及相应测试。当前仍为 draft，GitHub 没有已验证的完整 CI 成功结果；详细状态见 `reports/pr-verification.json`。目前修复中，#194 的正确性优先级最高，#198 的实测显存收益最明确。

![源参数生命周期修复的显存收益](figures/source-parameter-lifetime.png)

## FSDP 顺序与验收口径

先从 step0 训练至 step200，落 checkpoint，再加载它完成 step201–400，并与独立连续400步参考逐步比较。含 #180 的原版本和六项修复集成都已完成这一顺序。

六项修复源 `ec2e16bce26e497519c573f0a8d87a538a329197` 的基线和续训，全部8rank的 LM、MTP、aux、total loss 及梯度范数最大绝对差均为0。续训累计 token/sample 也逐步相同。曲线误差取所有rank最大值，不使用平均误差代替门槛。

![FSDP 0到200](figures/reviewed-fsdp8-baseline.png)

![FSDP 从200续训到400](figures/reviewed-fsdp8-resume.png)

模型日志中的 MTP 带0.3权重，aux带0.0001权重。证据同时保留日志值和去权重后的分量；新补充检查也对未加权分量使用0.005。已有全部误差为0的通过项不受口径影响。原始完整性检查仍验证8rank、全部步骤、初始张量、逐步输入身份、LR、退出状态和完成标记。

## 当前数值诊断

以下为明确标记的诊断插桩，比较双方使用相同数值配方。它们不能代替生产实现验收。

| FP32专家梯度 + 完整FP32 NS | LM最大差 | 日志MTP最大差 | 日志aux最大差 | 未加权MTP最大差 | 未加权aux最大差 | 中位秒/步 | 峰值allocated GiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| FSDP8参考，200步 | — | — | — | — | — | 6.39175 | 22.28220 |
| EP2，200步 | 0.0194788 | 0.00905466 | 0.0000128801 | 0.0301822 | 0.128801 | 6.59964 | 21.18306 |
| CP2，200步 | 0.0152731 | 0.0106752 | 0.0000170822 | 0.0355840 | 0.170822 | 7.06310 | 20.70665 |
| TP2+EP2，200步 | 0.0149679 | 0.00571430 | 0.0000107335 | 0.0190477 | 0.107335 | 7.46236 | 15.93807 |

完整FP32 NS使 FSDP 步耗时约为原配方的6.8倍，且EP/CP/TP组合仍不达标，因此尚不能推荐。专家梯度补丁、Muon精度选项分别通过独立检视和CPU回归；两者合并后的定向CPU回归为133 passed、83 subtests。修正测试清理接口和空分片观察方式后，专家梯度真实2卡ST为1 passed，Muon真实8卡/2卡ST为2 passed，数值容差未放宽；首次失败证据保留。生产配置FSDP8、EP2、CP2、TP2×EP2各3步集成已完成，全部8rank退出成功且短跑误差小于0.005；不作为200步通过结论。

TP 首步的完整分块哈希已将首次偏离定位到第一层输出投影：Q/KV投影输出与 `o_proj` 完整输入共4,608块完全一致，而attention输出采样最大差0.015625。只升高归约精度没有改善；局部投影也使用FP32时，匹配FP32参考的最大差降为0.0009765625，非零采样点比例从38.4%降到0.039%，但后续MLP仍放大残差。FP32投影会改变基线配方，不能与旧BF16基线混用。

FSDP/TP的实际NS矩阵批次一致；EP/CP有4个参数从singleton变为batch2，正用逐矩阵NS消融验证。CPU FP64范数对照已使EP首步28个受观测参数在裁剪前后和更新后的摘要一致，后续仍有差异；该覆盖未包含全部专家参数，且CPU拷贝仅用于定位。纯TP首次测试被融合gate/up连续切分保护拒绝，尚未构成合法纯TP训练验收。详见[定位记录](reports/precision-localization-plan-v5.md)。

![TP首步偏离位置](figures/tp-first-divergence.png)

![EP2诊断：未加权分量](figures/fpgmm-nsall32-ep2-200-raw.png)

![CP2诊断：未加权分量](figures/fpgmm-nsall32-cp2-200-raw.png)

![TP2与EP2组合诊断：未加权分量](figures/fpgmm-nsall32-tp2ep2-200-raw.png)

完整阶段证据：[并行数值诊断](reports/parallel-numerical-diagnosis.md)。

## 重计算性价比与灵活度

在本任务小模型、GBS8、seq4096下，关闭重算吞吐最高。40步独立A/B中，full节省3.095GiB、增加9.13%步时；现有selective节省2.407GiB、增加8.50%；单独MLP节省0.853GiB、增加1.91%。单独MTP block重算没有降低整步峰值，却增加约2.99%步时。norm的小幅时间差接近噪声。

MLP+norm组合另有200步验证，全部8rank的loss/梯度差为0，节省1.165GiB，步时增加约2%。这些模块选择为实验wrapper。JT当前YAML尚不具备MF的任意模块路径/层范围、排除及通信重算配置能力，不能把现有`selective`名称当作同等接口。

![重算策略成本](figures/recompute-module-cost.png)

![模块调用的时间与分配变化](figures/recompute-module-events.png)

模块events是inclusive时间；进入/退出allocated差含输出和临时分配，不是保存激活大小。不能相加嵌套时间，也不能把单次运行的步间分布称为重复实验置信区间。完整口径及最新MF/Megatron固定源码引用见[成本分析](reports/recompute-cost-analysis.md)、[接口与源码对照](reports/recompute-reference-comparison.md)。

## 环境与数据保留

125的隔离目录为 `<isolated experiment directory>`，8×Ascend910B3 64GiB。CANN为9.2.0-beta.2 / V100R001C12B086，torch/torch_npu 2.12.0，Python3.12.14。最新核对JT_dev为`ad1c553`，#180 head为`5de3cce`。本任务模型为示例的两层trunk、一层MTP、8个路由专家，不能外推为完整DeepSeek模型的性能结论。

所有比较保留配置、源码提交/patch、editable导入路径、全rank标量、初始化及输入哈希、清单和可编辑SVG。完成续训对照后，按清单清理四份大型checkpoint，累计约104GiB；仅处理本任务文件。NPU队列在其他任务占卡时等待，不终止其他任务。
