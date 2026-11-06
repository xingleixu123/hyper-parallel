# 七组完整续训窗口：同数值汇总排版

本目录是从公开数值包原 CSV 重新制作的中文 Matplotlib 汇总排版，**不是网站原始导出**。已有网站 headless 导出另存 `../website/`，本任务未访问浏览器、网站、GitHub、NPU 或容器，也未修改原数值包及网站文件。

每条数值序列严格为 step1501–2500 的全部1000点，没有平滑、下采样、补点、数值插值或数据裁剪；仅用折线连接相邻点。Matplotlib 路径简化关闭，SVG 保留每条完整折线路径。所有原始字段直接取 CSV，raw MTP 包含既有 weighted→raw 反解产生的浮点舍入，图中没有重新除以0.1。PNG具有有限像素；完整数值仍以有SHA的原CSV为准。

| 文件 | 内容 |
|---|---|
| `01_single_raw.png/.svg` | 单卡 MF 与 HP 的 LOSS、raw MTP 原始双面板 |
| `02_single_signed_deltas.png/.svg` | 单卡逐步 HP−MF，零线及 signed mean/MAE/max/正负零数量 |
| `03_single_cumulative.png/.svg` | 单卡精确fsum累计signed mean、±0.005尺度与终值 |
| `04_four_signed_deltas.png/.svg` | 六配置对旧continuous a852的主指标逐步signed Δ |
| `05_four_cumulative.png/.svg` | 六配置累计signed mean与完整窗口终值简表 |
| `06_four_raw.png/.svg` | 旧continuous与六个actual的原始LOSS、raw MTP |
| `07_aux_diagnostics.png/.svg` | 单卡AUX对MF；四卡AUX只对新FSDP496，不设阈值 |

图均为白底16:9，PNG2400×1350；SVG中的文字与折线路径可编辑，查看时需要现有 Microsoft YaHei 或兼容中文字体。未打包微软字体。图表文字按显示精度四舍五入，实际折线和验收复算使用 CSV 对应完整浮点值。图例和统计表独立排在绘图区之外。

## 数据口径

单卡：GBS2/LBS1/GA2，legacy微批等权，实际参考是冻结 MF52f 连续训练，HP从MF1500恢复。四卡：GBS4/LBS1，整步有效token加权，六配置主LOSS与rawMTP都对同一个旧continuous a852；新FSDP只作四卡AUX的完整全rank/micro/router参考。单卡和四卡的样本游标起点、GBS与数学口径不同，不能跨组混比。

| 配置 | DP/TP/CP/EP | GA | 重算范围 | HP源码 |
|---|---|---:|---|---|
| 单卡 | 1/1/1/1 | 2 | 原单卡配置 | 49bffc |
| FSDP全重算 | 4/1/1/1 | 1 | [0,1,2] | 496788 |
| FSDP骨干重算 | 4/1/1/1 | 1 | [0,1] | 496788 |
| EP2 | 4/1/1/2 | 1 | [0,1] | 496788 |
| CP2×DP2 | 2/1/2/1 | 2 | [0,1] | 496788 |
| TP2×CP2 | 1/2/2/1 | 4 | [0,1]，SPfalse/LPtrue | 3292a5 |
| TP2×EP2+SP | 2/2/1/2 | 2 | [0,1]，SPtrue/LPtrue | 3292a5 |

上述mesh参数按实际归档描述；EP和DP等group可重叠，不能机械相乘当总rank数。所有实验模型均2个基础层（1Dense+1MoE）+1MTP、序长4096、fusedmHC开启、clamp与CMHC关闭。TP共享MLP采用合法TP复制及相符序列IO方案，不声称section-aware packed TP实现。

- 单卡源码manifest：`49bffc6761d24e79ab59af1758b028145ba2bcc3c09030fec7b840cc9e005f85`，PR1477+#1420/#1470/#1441。
- 四个TP1源码manifest：`4967881062f63d8144bd8c4214b60c16f9ca7495b7f241a6701d04a630e08d34`，PR1477+#1420/#1470/#1453/#364。
- 两个TP/LP源码manifest：`3292a5b93c7b62d4f97c3edf42572aeaca91a475057bb3540ca6bcf1b9eae245`，在496上加入366/f8ec的显式router副本均值修正。三种源码不混称同一源，也不声称裸PR1477完成这些实验。
- MF实际冻结内容来源：`52f614eeb91c6e873d37bdbb5b880c82926ae9f6`；HP-MS来源`c69f8f70317268337314285f62d81bd5970c791a`。MS2.10.0正式构建来源`6b7c4b0274392eb09ee3a5652ba14e3ed7a270b4`、build20261104091850。MF运行CANN9.1 B243；其预编译extension工具链9.2 B237是单独字段。
- HP训练：Python3.12.14、Torch2.12.0+cpu、TorchNPU2.12.0.post3、Transformers5.21.0、CANN9.2 B237、Ascend910B3。完整原始公开版本/并行信息在数值包`source-version-parallelism.json`。

MF62ab的额外两步回归不属于这七组1000步CSV，不替换图中的MF52f参考，也不在图中声称62ab完整2500训练。

## 验证与复现

`verify_public_csv_stats.py`不依赖Matplotlib，仅用stdlib逐26CSV核SHA、步数和有限值，重新计算差值及math.fsum的signed mean、MAE、max及发生步、正负零数量，并与原statistics.csv逐字段精确比较。14个必需均值规则为`abs(mean(actual−reference))≤0.005`，分别用于LOSS和rawMTP；不是对逐点、MAE或max设0.005。AUX的threshold和pass/fail始终null。统计通过不替代原生保存、样本或CPU状态核验，原wrapper状态见原数值包，图不改写它。

累计第n点为`math.fsum(delta[:n])/n`，n从1至1000，终值精确等完整signed mean；中途累计值只观察，不提前判通过。`plot-data-audit.json`核全部61条数值折线、61000个点、x范围和无y数据裁剪，所有线的float64小端指纹可复算。`CSV-verification.json`保留完整统计、输入SHA；`requirements-freeze.txt`与`render-runtime.json`记录依赖和字体指纹；`manifest.json`/`SHA256SUMS`封存所有交付。

`verify_figure_exports.py`另用stdlib解析最终SVG：61条数值路径均为1个M与999个L，合计61000个顶点，全部文字可编辑且无嵌入栅格；同时核PNG尺寸、中文说明和原公开包42个文件的完整SHA。机械结果在`export-QA.json`。七张实际PNG均已逐图目视检查中文、图例、表格和边界，已审图片SHA及观察记录独立保存在`visual-review.json`；脚本不把机械校验冒称人工目视。

公开脚本没有硬编码本机私有目录。默认取脚本相邻上层的`numerical`，也可提供已解压公开数值包路径：

```text
python -B verify_public_csv_stats.py --archive ../numerical --output CSV-verification.json
python -B plot_presentation_figures.py --archive ../numerical --output .
python -B verify_figure_exports.py --archive ../numerical --figures .
```

如默认Windows系统雅黑路径不可用，使用`--font /path/to/msyh.ttc`指向已有雅黑字体。本次使用已安装任务venv的Matplotlib3.11.2；未安装或更新任何框架，未读取private YAML、checkpoint、真实token、RNG状态或凭据。

本发布副本仅将可移植数据目录改为相邻numerical、网站导出路径改为相邻website，冻结绘图直接依赖；七张PNG／SVG和原CSV数值未改。原绘图包manifest SHA为599e8b42acc7e1b57c5874668376b9cfeb5d9efe5b639583a696c3757c9c0a4e；其source archive SHA指原准备快照，所有26份CSV与当前numerical同字节。
