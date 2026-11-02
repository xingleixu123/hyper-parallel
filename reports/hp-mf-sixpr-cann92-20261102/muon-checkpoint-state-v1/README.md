# Muon checkpoint 非 owner 状态：独立诊断

本证据针对 ordinary native checkpoint 恢复后非 owner Muon momentum 生命周期。正式配置为3层 Xing40（2 Dense + 1 MoE）、S4096、GBS4/LBS1；旧源码 `2dd33e02f46fc5d6810e806b9d7d4717cf10a3e4`，新源码 `f827186ea3416b58371603210140ee32ff08b04f` 包含 e5978afd2817f83eaedf41c609226cd4242bab7c 的四文件 cherry-pick。四臂是125同卡串行 HSDP old/new 与 TP2 FFN replicate old/new：各从同一1500 checkpoint恢复，1501–1570共70步、前20步预热、后50步记录。这不是ABBA，也不是完整1000步验收。

新源码清理前四rank完整local状态逐字段等于此前独立全局CPU验证的目标；依据实际assignment，清理后只移除非owner momentum，model/main alias、owner状态、Adam、组及恢复extra/样本不变。HSDP四rank删除12/12/16/16项（56），TP为8/11/8/11项（38）。每rank实际逻辑slot字节及helper allocated、reserved前后见 [正式短跑摘要](formal-70-summary.json)；allocated变化含allocator等因素，不能直接等同slot逻辑字节。70步实际raw、prepared、sampler、框架RNG摘要，old/new四rank逐项一致；不推断独立kernel RNG。

| 布局 | old median 秒 | new median 秒 | rank0 mean(new−old) | rank0 MAE | rank0 max abs |
|---|---:|---:|---:|---:|---:|
| HSDP | 1.077033 | 1.081759 | +3.08582e−5 | 0.000230380 | 0.00249863 |
| TP，FFN replicate | 1.401750 | 1.401039 | +4.31810e−5 | 0.000246934 | 0.00072217 |

时序每步取4rank最大值，吞吐16384 tokens/update，IQR/p95及各rank峰值见摘要；[50步时序](formal-time50.csv)和[70步四rank标量](formal-scalar70.csv)保留原值。短窗不设1000步阈值，单次差异不能作为性能显著性或因果结论。外卡离散监控无异常，不证明连续独占。

独立 tiny 合成 HSDP2×2 原生 callback 回归中，清理前与共同producer canonical、清理后权威状态及steps2–4与旧A一致；sync和async checkpoint各7个全局tensor、31个group字段经无设备CPU native DCP独立读回，与旧A实际rank切片逐值一致，见 [tiny 对照](tiny-compare.json)、[全局读回](tiny-global-readback.json)和[内存时序](tiny-memory-timeline.json)。这是合成模块，不能替代正式模型1000步验收。旧B产生worker结果但controller因rank identity changed失败，该旧案例不改判为闭合成功。

单卡 MF/HP 满足既定窗口数值条件不证明 Muon 分组或算法等价；相关选择仍在确认。主线完整数值/性能见 [四卡报告](../fourcard-report-v2/README.md)。
