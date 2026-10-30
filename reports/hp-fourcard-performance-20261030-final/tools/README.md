# 本任务的在线 Edge 绘图副本

本目录只归档 `plot.cjs`、原版 `plot.base.cjs`、`plot.diff` 和本文；未打包本机的离线site-snapshot、examples或测试fixtures。公开复现使用online模式，不能从此四文件目录运行`--snapshot`或声称已独立重跑本地15项准备测试。

原版SHA256：13d43733e714e4449234a79867286237a72f894d30cc5db110aa1bd394e3b4f1；在线副本：5c75e34dedd60c5e10e03c4a560fe3ecf0db305a6a7f13f904a6186b4050ef10。`plot.diff`仅增加公开X输入范围和网站白底卡片截图，不改原source-points、worker或CSV逐点验证。真实13个online视图回执在[验证清单](../website-validation.json)，可运行参数数组在[commands.json](../commands.json)。

需要已安装Node.js、Microsoft Edge和Playwright。选择一个参数数组，在本报告根目录运行并将输出占位符换成新空目录；不含`--headless`的两个interactive视图会保持窗口打开。X轴范围包含全部输入点；网站原生下载和白底卡片截图分开记录，后者不是离线重绘。

绘图输入为带来源的派生JSONL，不是原始私有训练文件；零线是数学参考。零基准相对误差以及对累计均值再次取平均不用于验收。累计曲线终点对应完整1000步signed mean，0.005不作为逐步差阈值。
