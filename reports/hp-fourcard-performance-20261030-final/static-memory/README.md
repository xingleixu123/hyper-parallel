# 八组实验的 allocator 峰值附图

**公共归档复现范围：** `plot_memory.py`保留原工作区布局和本机离线Matplotlib runtime调用，是可编辑历史来源；本公共目录不含这些依赖或原父目录。不能将下面原工作区命令视为在本公共目录开箱可运行。跨机器重绘须安装兼容绘图依赖、将输入改到`../site-inputs/strategies/`，并配置输出/字体路径；CSV、figure-data及可编辑SVG可直接复核。网站13视图的Node命令另见根目录commands.json。


[PNG](allocator-peaks.png) · [SVG](allocator-peaks.svg) · [绘图源码](plot_memory.py) · [逐卡绘图值](figure-data.json)

**图注：** 这是一张本地 Matplotlib 绘制的静态附图，展示 8 个隔离组内 18 条策略序列的 72 条卡记录；并非网站导出，也不是逐步显存曲线。每个策略的两根柱分别为计时窗口内四张卡各自的 allocated peak、reserved peak 中的最大值；四个符号分别标出物理卡 4–7 的实际峰值。柱从零开始，标签四舍五入至 0.01 GiB，完整数值见逐卡绘图值。横轴依照源数据中的语义顺序，不按内存排序。窗口为 step 1511–1540，GBS4、LBS1、序列长度 4096；分面标题标记来源、主机以及配对或非同期参考。

18 条是图中的展示序列，不是 18 条互不重复的新实验：EP2 面板的两条历史 FSDP 参考在其他面板也有引用。不同分面之间条件和运行时段不一，CP/EP 参考非同期；配对运行也为顺序观察，不据此推断普遍因果。`reserved` 可能包含 allocator cache；`allocated`、`reserved` 不是全设备物理 HBM，二者没有堆叠，也未相减或解释为 activation。warm 唯一 storage 是独立快照，本图未绘制或伪造成时间序列。

本图数据仅读取同级 [memory-peaks.csv](../site-inputs/strategies/memory-peaks.csv)（SHA256 `8b487ff35b571a74e4a31dc56dfd9a85f3e924b9369f82cef7d3eb797e8495eb`）、[groups.json](../site-inputs/strategies/groups.json)（`41588a6c90e22e22b8919e7ec58f7c1116eb1eaac2d6208ffb3d02f985b092db`）与 [manifest.json](../site-inputs/strategies/manifest.json)（`be2491f3eeafbe79efc2a8a061a36295f0416dc47f00f842fbddf4b57d807146`）；源包 `SHA256SUMS` 的 SHA256 为 `3f9c674ab8c7ce20f4f14d048689da1b8c38d8f11c4f8a345da16ecb5f5db7c6`。`figure-data.json` 保留各组、run、四张卡和源文件指针；其中两条柱高可从四个记录直接复算。

图中文字使用 Matplotlib 自带 DejaVu Sans 英文标签，本中文图注补充说明，未下载字体。绘图采用本机缓存的 Matplotlib 3.11.2 与 SHA 固定的兼容 wheel，版本和摘要见 [environment.json](environment.json)；没有下载软件、访问网站、调用 NPU 或外发。

[cleanup-receipt.json](cleanup-receipt.json) 记录本任务临时解包目录的准确路径与大小。自动审批拒绝了对该目录的递归删除，返回 `blocked by policy`，未提供更具体原因；该目录已保留，没有换工具或路径重试删除。输出清单只覆盖图、源码、说明和验证文件，明确不把保留的临时解包目录当作交付源或纳入清单。
