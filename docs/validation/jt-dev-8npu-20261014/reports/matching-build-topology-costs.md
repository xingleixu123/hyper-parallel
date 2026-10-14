# 相同 build 观测下的时间与显存

所有行使用同一输入身份采集方式，保留 NPU→CPU 拷贝及哈希开销。时间为去掉前20步后的中位数，显存为全部8rank的峰值 allocated；均为单次200步测量，不能给出重复实验置信区间。

| 配置 | 秒/步 | 相对FSDP步时 | allocated GiB | 节省 GiB | 相对FSDP精度 |
|---|---:|---:|---:|---:|---|
| FSDP8 | 0.937752 | +0.00% | 22.633759 | 0.000000 | PASS |
| EP2 | 1.020075 | +8.78% | 21.183023 | 1.450735 | FAIL |
| EP4 | 1.029181 | +9.75% | 21.322903 | 1.310856 | FAIL |
| CP2 | 1.285255 | +37.06% | 21.058210 | 1.575548 | FAIL |
| CP4 | 1.670697 | +78.16% | 20.217754 | 2.416004 | FAIL |
| TP2, experts replicated | 1.416853 | +51.09% | 17.771236 | 4.862522 | FAIL |

该小模型在8卡FSDP下已经能放下。TP2通过分片稠密层进一步降低峰值，但增加通信和microbatch次数；CP2/CP4减少局部序列激活，也增加注意力通信。当前实测均没有吞吐收益。EP4相对EP2没有继续降低整步峰值，说明不能只按专家数推算最终 allocated；参数、激活和通信临时量共同决定峰值。这些解释是与配置一致的推断，不是独立算子成本分解。

生产版FP32专家权重梯度＋FP32 NS的FSDP基线为1.165173秒/步，allocated 22.282196GiB，相对原配方步时增加24.25%。这是两项选项合用的成本，不能单独归因于NS。该200步基线已通过完整性自检，续训/连续参考及跨拓扑门槛仍待完成。

纯TP同源runtime/build的200步LM/MTP/aux/total和grad norm逐步全部相同。runtime为7.13563秒/步，build为1.41685秒/步；此差异是观测模式的A/B，不是并行策略或优化器精度的速度比。

阶段诊断采样第4–6步×8rank。runtime路径执行梯度/参数/优化器状态采样及布局记录，耗时差主要位于prepare_optimizer_step和step_optimizers_and_schedulers两个外层区间，其内部NS事件基本相同。事件包括当前流等待与host提交间隙；嵌套父子区间不能相加，阶段采样不能代替200步吞吐测量。

原始依据：matching-build-topology-costs.json、tp2-runtime-vs-build-numerical-comparison.json、tp2-stage-profile-runtime-vs-build.json以及各配置comparison.json。完整原始标量和日志在本地归档V9保留。
