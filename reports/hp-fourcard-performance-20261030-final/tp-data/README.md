# TP deferred：千步精度与首 40 步性能补证候选

**本包待 ROOT 审核，尚未发布。** 在固定三层 Xing40（两层 Dense、一层 MoE）、GBS4/LBS1、S4096、原 2500 步学习率日程下，从同一 native step 1500 恢复。实测 TP2/DP2、full recompute、deferred 的 1501–2500 loss，与历史 original 1a、FSDP4/selective 参考比较，完整 1000 步 `abs(signed mean)` 为 **0.000142573595047**，落在既定 **0.005** 均值界限内。该界限只针对完整千步有符号均值，不是每步误差、MAE 或完整模型参数通过标准。此比较同时改变 TP、重计算与梯度同步策略，不能归因于 no-sync 单项。

MTP 在两臂配置中关闭（层数及 loss factor 均为 0），因此 **MTP = N/A**；没有执行 MTP 模块或 MTP loss 验收。

## 原始千步序列

| candidate − 历史 reference | 数值 |
| --- | ---: |
| 有符号均值 | −0.000142573595047 |
| 有符号均值绝对值 | 0.000142573595047 |
| 平均绝对误差 MAE | 0.000375196695328 |
| 正 / 负 / 零 | 437 / 563 / 0 |
| 最大绝对差 / step | 0.076320648193359 / 1804 |

[loss-pairs-1000.csv](loss-pairs-1000.csv) 给出每个 step 原始 reference loss、TP loss、差值及累计有符号均值，保留全部 1000 行，不平滑、不删除异常点。`delta = TP − reference`；截至 step `s` 的累计均值为 `fsum(delta[1501..s]) / (s−1500)`。累计均值是前缀统计，原始 loss 和逐步差值仍完整保留。

源码在 trainer 内按 token 数在 DP+CP 组内聚合，并累加 micro 的返回 loss；两组记录的 sequence parallel 均为 false。本表直接使用 rank 0 的 trainer 返回值，不再次对四 rank 求和或均值。四 rank 的结果并非逐位完全相同：TP 其他 rank 相对 rank 0 的最大绝对差约为 `1.19e-6` 至 `1.67e-6`。代码和文件 SHA、原始行号及每个 step 到源 JSONL 的一对一行号公式，见 [source-lineage.json](source-lineage.json)。

## step 1804

该点两条主轨迹 loss 分别为 **5.731315136** 和 **5.654994488**，对应 grad norm **60.253479004** 和 **57.202770233**。独立的既有 SAC-fixed selective 对照在同一步 loss **5.654379845**、grad norm **57.127082825**；其相对 original selective 的全 1000 步最大绝对差为 **0.076935291290283 @ 1804**。两者在 1803 和 1805 的原始数值也保存在 [metrics.json](metrics.json)。

这三个历史运行在相同步骤使用一致的样本索引计划，呈现共同的高梯度敏感点。它们的源码、策略和时段不同；已完成的记录不能确定差异原因，也不能证明该步骤所有 prepared 字节已经逐字段直接比对。不要把均值达标写成这个单点也受 0.005 限制。

## 首 40 步同期策略对照

两臂为同一 source 0f（固定 1a + 唯一 scheduler overlay）、同配置，只在同步策略布尔值上不同；sync 执行至 1540，deferred 在同一进程组继续至 2500。物理 4–7 四卡，1501–1510 预热，1511–1540 计时。每步取四个 rank 各自同步 train_step 时长的最大值，然后对 30 步统计；这不是跨 rank 最早开始到最晚结束的全局跨度。名义吞吐为 `16384 / 平均秒数`，未扣 padding 或 mask。

| 计时指标 | sync | deferred |
| --- | ---: | ---: |
| 平均 / 总体标准差 (s) | 2.017418988 / 0.004819293 | 1.824370796 / 0.040858218 |
| CV | 0.238884% | 2.239579% |
| p50 / p90 (s) | 2.016456725 / 2.022904738 | 1.816370585 / 1.820086712 |
| min / max (s) | 2.010611360 / 2.036823060 | 1.813393110 / 2.044071760 |
| 名义 tokens/s | 8,121.268 | 8,980.630 |

本组的耗时均值比为 **0.904309**，名义吞吐比为 **1.105816**；同编号 30 个计时步中 deferred 29 步较短、1 步较长，最慢点为 step 1531。均值、总体标准差、CV、分位数、最值和逐步数据均保留在 [metrics.json](metrics.json) 与 [first40.csv](first40.csv)。

| rank / 物理卡 | allocated peak GiB：sync → deferred | reserved peak GiB：sync → deferred | warm 唯一 storage GiB：sync → deferred |
| --- | ---: | ---: | ---: |
| 0 / 4 | 19.879516 → 20.197734 | 20.546875 → 23.669922 | 6.223887 → 6.223887 |
| 1 / 5 | 19.879516 → 20.197734 | 20.546875 → 23.669922 | 6.223887 → 6.223887 |
| 2 / 6 | 19.879515 → 20.197733 | 20.546875 → 23.669922 | 6.223887 → 6.223887 |
| 3 / 7 | 19.879515 → 20.197733 | 20.546875 → 23.669922 | 6.223887 → 6.223887 |

每卡 allocated peak 增加 **341,684,224 bytes（0.318218 GiB）**，比例约 **1.016007**；reserved peak 增加 **3,353,346,048 bytes（3.123047 GiB）**，比例约 **1.151996**；同 rank 的 warm 唯一 storage 字节数相同。原始 bytes、差值和比例见 [memory.csv](memory.csv)。每个计时步先重置峰值，本次所有 240 条 measured rank-step 的 after peak 均高于重置后的 before allocated；allocated 峰值属于对应步骤。reserved 可能包含预热遗留的 allocator cache；warm 唯一 storage 不包括 cache、activation、库和 HCCL 占用。

既有历史 TP conservative 同步单组的相同口径均值约 `2.019936 s`，可作为量级上的近似历史支持；其 worker 和计时脚本 SHA 不同，首 40 步循环前三条语句的 AST 相同，不能称为同源码重复实验，也未纳入本组差值计算。

这是一组顺序运行的描述性对照。ROOT 曾在 10:33 观察到 physical 0 外部作业；deferred 之前桌面记录 11:13:14 对 physical 0、11:14:48 对 physical 0–3 的时点查询无进程。sync 计时窗缺少同粒度背景记录，点查询也不能证明整个计时窗主机独占。桌面记录时间与同次 host epoch 相差约 297 秒，不能混用为精确同步窗口。因此不能排除后台、运行顺序或其他时变因素，也不能从本组差值直接宣称普适或严格因果收益。

## 证据边界与文件

TP deferred 与历史 FSDP reference 均有实际终态、四 rank 1000 步连续 loss 和 258-role 恢复审核；TP terminal exit 0 并记录运行后源码 2,661 项检查。四 rank 的 post 观察各含 258 个 role，全元素 finite 观察结果均无 nonfinite，role layout 和绑定与 target 对齐。它们是实际运行后记录及哈希，不是跨运行全权重比较。

两次主运行的实际 dataset 读取索引各覆盖 canonical 6000–9999 共 4,000 个；TP rank 0/2 各读 2,000 个，TP peer 不独立读取。前两步直接检查 prepared 的四字段和 TP 广播；从 step 1503 起恢复原生 get_batch，后续已核实际索引和游标，不能宣称每步四字段 hash 均已直接检查。optimizer groups 在 1500→2500 保持成员和非 step 选项，LR 为 `1e-5`；保留 RNG 恢复证据，但不宣称跨 mesh 随机流等价。

本包公开的是 loss、计时、聚合内存、方法和哈希；不包含私有 YAML、权重或 slot 值、完整 RNG、实际 token 或样本内容。[build_public.py](build_public.py) 只在持有对应私有归档、固定本地 Git 和 PyYAML 的环境中读取输入，逐文件核 SHA 后生成白名单；仅持本包者可复核表格和来源指纹，不能凭本包重算私有原始数据。[SHA256SUMS](SHA256SUMS) 固定当前 v1 文件。

若后续取得 sync40 repeat，它是可选增强，应作为**独立版本化扩展包**添加，引用本包 SHA，不覆盖或回填本包现有结果；其终态、输入身份、前后背景和双时钟证据需另行审核。本包不等待且未纳入重复结果。整理模型为 GPT-6.1 Terra max；当前工具无 Fast selector，未声称启用。
