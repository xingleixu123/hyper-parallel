# 单卡完整 1501–2500 结果

完整 1000 步统计、实际样本、producer rc=0、final checkpoint 与独立 CPU 逐叶读回均已验收。运行源码是 PR1477 加 #1420/#1470/#1441 的精确组合；不称裸 PR1477。四卡完整训练矩阵仍待独立验收。

| 指标 | mean(HP−MF) | abs(mean) | MAE | max abs / step | 均值规则 |
|---|---:|---:|---:|---|
| LOSS | -0.00014291787147521973 | 0.00014291787147521973 | 0.0039931757450103762 | 0.028119802474975586 / 2001 | PASS ≤.005 |
| MTPLOSS | 0.00036955329775810241 | 0.00036955329775810241 | 0.0038067377507686614 | 0.024060249328613281 / 2249 | PASS ≤.005 |
| AUXLOSS | -0.00020188993215560912 | 0.00020188993215560912 | 0.0031050047278404235 | 0.016387343406677246 / 2459 | 仅观察，无门槛 |

规则分别检查 LOSS 与 raw MTPLOSS 的 abs(mean(HP−MF))≤.005，不是逐步误差或 MAE≤.005。AUX 只观察，不虚设通过门槛。

LOSS 为主 LM；MTPLOSS 为 .1 系数前 raw 值。AUX 为两个 router、GA2 两个微批平均的 raw sequence aux，每个 router 的训练系数 .0005。目标保持 world1 legacy 微批等权；不叠加 token 目标来改写既有 MF 参考。

1000 步连续 optimizer 进度与未压缩 LR schedule 已独立核对。8000 个捕获的 normalized int32 输入字段完全一致，不扩大为原生 dtype 相同。最终 CPU 读回核验 65 个模型张量、91 个优化器槽、61 个参数名及 643 项序列化优化器字节描述符；61 个 FP32 alias 的身份在 MF1500 恢复时验证，不声称 CPU 读回重建运行时 alias。NumPy RNG 未被原生 DCP 序列化，此域依赖 MF sidecar 恢复且本次 live consumer 未使用；其他已采域见单独验收记录。

source manifest SHA256 `49bffc6761d24e79ab59af1758b028145ba2bcc3c09030fec7b840cc9e005f85`，1542 文件。CPU readback SHA256 `9d0fdd0434377fe0d73053d6c4f94b3b6eb4ebce05b4be773a9b4f14cf1d3ecb`。

完整数值见 [1000 行 CSV](single-1501-2500-pairs.csv)、[根任务独立统计](single-root-statistics.json) 与 [最终验收](single-acceptance.json)。没有发布私有配方、配置字段映射、训练 payload 或 checkpoint。
