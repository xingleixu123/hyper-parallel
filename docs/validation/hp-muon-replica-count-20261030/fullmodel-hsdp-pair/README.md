# Muon 副本数：同源 HSDP2×2 四卡实测（40步）

结论：显式 count=1 的配置语义修复有效；本次代表性整模型对照没有测到性能或显存收益，因此不把它推广为加速建议。这里只实际执行1501–1540共40步，不替代1000步精度验收。

## 固定条件与实验问题

本轮问题：修复显式设置被覆盖后，将优化器去重子组宽度从默认2设为1，实际执行和资源占用是否符合接口语义。hsdp_replica_count 表示子组宽度；1形成单rank组，关闭副本维度的组内去重，不能理解为减少总状态副本。区分性预期：默认组宽2时四rank的momentum/update owner记录为16/16/12/12，显式1时各28；这增加而非减少此维度的owner工作。owner元数据本身不能证明完整NS kernel的计算分配或次数。两臂均为1a39572f加同一 Muon 唯一 overlay a3e5d623，source manifest f829c314；原 native1500 checkpoint、3层(2Dense+1MoE)、MTP关闭、GBS4/LBS1、序列4096、原2500步学习率日程均一致。只改 optimizer replica count，default 与 explicit1 的配置SHA见配对JSON。

125 physical4–7，CANN9.2.0，PTA2.12.0.post3，Torch2.12.0+cpu，Python3.12.14；环境指纹 e14f2704。经原 nsenter host-mount/image-root 协议执行；真实 worker 为UID0/Seccomp0/CapEff000001ffffffffff，不能用 Docker init 安全字段替代它。原始14控制器、namespace、PID/tick、argv、七设备allowlist已存档。

## 测量口径

每臂原生续训40步，1501–1510预热，1511–1540计时。每步计时前后均同步，各rank本地持续时间取最大值；这不等同基于统一时钟的全局makespan。名义吞吐=16384/平均秒数，未删点、未平滑。每步同步后重置peak stats，因此 allocated峰值覆盖该步(含当时存活张量)，reserved可能延续预热缓存；二者均非 npu-smi 实际HBM峰值。

| 配置 | mean s | 总体标准差 s | CV | p90 s | 名义tokens/s |
|---|---:|---:|---:|---:|---:|
| default | 1.168631892 | 0.004000716 | 0.342% | 1.174292762 | 14019.812 |
| explicit1 | 1.182438706 | 0.003941886 | 0.333% | 1.187624914 | 13856.109 |

显式1比default耗时+1.181451%，名义吞吐−1.167656%。只有一组顺序配对，不声称稳定的普适性能排名，也不能从这1.18%净效应分离出通信、NS计算或主机波动的份额。

| rank/物理卡 | default peak allocated B | explicit1 peak allocated B | 两臂 peak reserved B | 两臂 warm唯一storage B |
|---|---:|---:|---:|---:|
| 0/4 | 23543166976 | 23543166976 | 24010293248 | 9642977812 |
| 1/5 | 23543165952 | 23543165952 | 24010293248 | 9642977756 |
| 2/6 | 23543166976 | 23543166976 | 24010293248 | 9642977812 |
| 3/7 | 23543165952 | 23543165952 | 24010293248 | 9642977756 |

实测峰值和warm唯一storage在两臂相同。原始记录显示默认组宽2仍在每rank恢复了全部28个Muon momentum槽，其中非owner也保留槽；显式1则每rank拥有28个参数。本轮没有通过清理/重分片消除已恢复的非owner槽，因此不能把owner逻辑去重直接当成已实现状态内存切分。尚未测量全NS kernel的owner工作量或广播关键路径的独立耗时。

## 恢复、数值与证据边界

两臂重新通过258角色（58state、57parameter、57main、86slot）的完整恢复审计、别名、mesh/group、extra/RNG/loader、13个checkpoint文件身份及实际下一批样本核对。各40步终止正常，完整原始日志/配置/测量点已逐字节归档，两个容器均在确认无活跃消费者后移除，30个bind源均保留。checkpoint payload的全局参考复用已封存真实读回，未声称重新hash全部约19GB payload。

第1501步rank0 LOSS两臂同为7.199586868286133。40步rank0差值显式1−default的有符号均值为-2.4533271789550783e-05，MAE=0.000307464599609375，最大绝对差=0.0023298263549804688，正/负/零=22/17/1。这是短程诊断，不套用完整1000步验收结论。MTPLOSS为N/A，不填造0。

步后232个同位model-state记录10hash相同/222不同；112个实际owner动量对比hash不同。hash只证字节不同，不能给出数值误差或判定数学错误。模块FP32/BF16分支的既有四卡数学/原生恢复试验提供独立边界证据；本40步整模型记录不替代全张量更新oracle。

## 决策与后续

保留issue1364/Draft1366以修复“显式设置被自动推断覆盖”的真实接口语义问题；不以本性能结果宣传加速。停止为追求收益重复该count消融，转向TP2/DP2两个micro的延迟梯度同步：先同状态同样本单更新比较真实完整梯度/裁剪/权重及优化器slot，再决定性能测试。

## 复核与配置

- [逐步原始rank日志](raw/)：两臂各四rank40步，未删点、未平滑。
- [逐步配对CSV](data/step-rank-data.csv)、[各rank损失诊断](data/loss-diagnostics.json)、[性能/显存完整统计](data/pair-summary.json)。
- [运行终止与258-role恢复审计](raw/)、[归档/精确容器清理证据](archive-proof.json)、[版本和数据标识](source-info.json)。
- [复核代码](code/)；需要用户保留的完整本地归档，原始NPU运行命令见私有配置档案，不向公共证据放YAML。
- [两臂实际输入及四rank resolved YAML（仅owner可见）](https://github.com/xingleixu123/mindformers-private/tree/c39ab4fc061064c9a857e9c37e908041e690b3c4/private_review/archives/hsdp-replica-fullmodel-pair-20261030)。15文件逐字节读回，仓库private且collaborator只有owner；未覆盖原MF/HP档案。

此附录为已完成的真实40步证据。原始候选README如写“待执行”仅代表封包时状态；本附录以两臂exit0的实际终止记录为准。绘图网站按用户要求在其他工作完成后处理，原始数值已保留。

## 语义更正（v2）

初稿把count=1写成减少更新工作的假设，方向错误；本修订按冻结代码的_split_replicate_groups和实际owner/槽记录更正。所有原始运行数据、耗时、显存、损失统计、配置和恢复结论不变。[owner/槽逐项证据](owner-assignment-proof.json)仅重组已有实测元数据，未新跑NPU；不据此推断2倍完整NS计算。
