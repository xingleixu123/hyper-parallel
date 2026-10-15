# 当前固定配方的资源总览与重计算能力

本目录汇总六组独立完成的 200 步实验，比较对象均为固定集成提交 `18602c4524dc664b81b2bad9a7b624455ce1494f`、JT_dev 基点 `f637ef7`、129 的八张 Ascend 910B、CANN 9.2.0-beta.2。**FSDP 的关闭、完全、选择重计算精度通过；EP2、CP2、TP2 精度仍未通过。** 红色斜线资源条不能解释为可交付的优化推荐。

![显存与时间，精度结论分别标注](../../figures/f637-current-resource-overview-zh.png)

| 配置 | 采样截至 allocated 峰值 GiB | reserved 峰值 GiB | 第 21–200 步全 rank MAX 中位秒/步 | allocated 节省 | 步时变化 | 200 步精度 |
|---|---:|---:|---:|---:|---:|---|
| FSDP8 关闭 decoder 重算 | 22.032196 | 27.250000 | 1.155003 | 基线 | 基线 | 通过，差 0 |
| FSDP8 完全 decoder 重算 | 19.392933 | 24.320313 | 1.235180 | 11.9791% | +6.9417% | 通过，差 0 |
| FSDP8 选择算子重算 | 19.875641 | 24.789063 | 1.232111 | 9.7882% | +6.6760% | 通过，差 0 |
| FSDP + EP2 | 20.928654 | 26.275391 | 1.317435 | 5.0088% | +14.0634% | 未通过 |
| FSDP + CP2 | 20.706646 | 25.982422 | 1.596322 | 6.0164% | +38.2093% | 未通过 |
| FSDP + TP2 / SP | 17.294672 | 18.638672 | 1.857553 | 21.5027% | +60.8268% | 未通过 |

每组采用全部八个 rank 的逐步耗时最大值，去掉前 20 步后计算中位数。吞吐为窗口内有效监督 token 总数除以逐步 rank MAX 时间总和，逻辑数据只计一次。显存取全部 200 步采样中的全 rank 累计峰值，包含初始化和 warmup，不能称作保存激活的独占容量，也不包含采样之后的末次 checkpoint 保存过程。原始读取和这些统计由独立审计重新计算，见 [独立审计](f637-resource-overview-independent-review.json)。

这是单次匹配观测，均含输入身份摘要的采集开销，不是无插桩吞吐或端到端性能。完全重算比选择重算多省约 0.483 GiB，单次中位步时差仅约 3.069 ms，不能据此排稳定速度名次。TP2 同时开启 SP/loss parallel，各拓扑在固定 GBS=8 下改变了微批次数；没有把全部耗时变化归因于单一通信或算子。

JT 当前 YAML 可直接配置 off/full/selective。底层已有 checkpoint 和 exclude wrapper、策略函数等接口，但 JT adapter 的安全子区域声明、默认整层的层过滤、任意路径与排除/通信的用户配置映射仍有缺口。MF PyNative 与 Megatron 的当前固定官方提交、精确语义及建议见 [详细能力对照](f637-recompute-flexibility-gap-review-v2.md)。当前 full 覆盖两个主干 decoder 和一个 MTP decoder；off 仍保留独立的 chunked head/CE 重算，两者均已在报告中说明。

[当前版本模块成本复测计划](f637-module-cost-current-plan.md) 区分 norm、dense MLP、完整 MoE、MTP 内部 decoder；只是待执行计划，没有将历史版本的模块消融结果改写为当前测量。新的区域先验证实际调用与完整梯度，再进入 200 步；计时探针和整步资源实验分开。

可复核产物：

- [原始数值与 60 项输入 SHA](f637-current-resource-overview.json)、[CSV](f637-current-resource-overview.csv)、[可编辑 SVG](../../figures/f637-current-resource-overview-zh.svg)。
- [原始证据的公开路径映射](source-map.json)：48 份 rank 指标及 12 份严格/资源报告按 SHA 复用已公开文件，不重复上传原始训练数据。
- [绘图工具](tools/plot_f637_resource_overview.py) 与 [全 rank 统计实现](tools/all_rank_resources.py)。在任务根布局下使用绘图脚本可重算全部数据并出图；输出采用新建方式，已有结果不会被静默覆盖。
- `source/` 是能力审查引用的集成源码快照，文件 SHA、原始路径与转换后的文档链接记录在映射中；没有修改训练源或新增生产补丁。

关联：[中文总问题单 #202](https://github.com/hf1s/hyper-parallel/issues/202)。全部独立问题仍按“一小问题单、一单提交 PR”维护；本目录是证据汇总，不新增合并多个问题的代码 PR。
