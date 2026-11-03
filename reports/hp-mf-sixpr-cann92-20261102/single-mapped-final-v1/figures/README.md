# 图像身份和图注

三幅图均由已核验的单卡 1501–2500 scalar 生成，横轴为训练 step，无平滑、无降采样，均为 1200×880 PNG。红蓝 loss 曲线高度重叠，需结合差值图查看。文件按网站图卡导出的原字节归档。

| 文件 | 图注 | SHA-256 |
|---|---|---|
| `loss-overlay.png` | MF callback loss 与 HP 两个已按 8192 token 归一的 micro loss 之和，每侧 1000 个步点。 | `088f57dd504ddd8496c9c30ddc8192787cfc898a1b507b9d5277a84c1150de6b` |
| `signed-delta.png` | 每步 HP−MF native loss 与零参考线；单步差值是诊断数据。 | `6a2f8812704e4e2472656c0e408c60f61b972839d9f8d2c2613715ce786986b7` |
| `cumulative-mean.png` | 从 step 1501 至当前步的累计有符号均值；末点为完整窗口均值。 | `3b24fea0d0019405b98fa6b4c3cd61afb9591c9ae5a362c51483fc276f22b7cd` |

完整窗口的绝对有符号均值门槛为 0.005；不能将中途累计均值或单步绝对差解释为另设的验收门槛。scalar 原值见 `../paired-loss1000.csv`。
