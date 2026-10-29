# PR310 精度图：总 issue #304

本目录只公开两张经用户明确授权的精度PNG及其SHA；实际HP/MF YAML和逐步数据保存在本人私仓，不在此目录公开。

- [单卡MF→HP](figures/mf-hp-single-1501-2500.png)：MF连续训练，HP从真实MF1500完整状态独立续训1501–2500。
- [HP四卡](figures/hp-fourcard-1501-2500.png)：相同FSDP1500状态，各配置独立续训1501–2500。

每窗口1000步、4096、2 Dense+1 MoE；单卡GBS2/LBS1，四卡GBS4/LBS1。MTP按唯一base关闭，MTPLOSS为N/A。阈值0.005只约束完整窗口abs(mean signed delta)，图中保留正负与全部离群点，不平滑、不降采样。单卡/FSDP/EP/selective长跑身份33096，CP/TP为1a；图片不代表新增32GiB环境或最新上游523的直接训练结果。

图来自log.javazero.top真实逐步数据。单卡保留原生网站图；四卡是实际网站截图按已观测边界裁切排版，曲线未重绘/缩放。来源与统计、独立修复及实验范围见[总issue304](https://github.com/hf1s/hyper-parallel/issues/304)。
