# Xing4 检视整改与完整精度回归

模型 PR [#1439](https://github.com/mindspore-ai/hyper-parallel/pull/1439) 追加提交 `fdce1a9a2285f1ab76f4c73d4e68b620b134eaea`，原六个提交和原检视意见保留，PR 仍为 Draft。增量为 8 个文件、+258/−343；官方 HF 三个 Python 文件未修改。29 个原始检视线程逐项审查，14 个此前尚未回复的线程已补充处置理由，未代替检视者 resolve 线程。

**当前验收使用统一的复合运行源码**，manifest SHA256 为 `7dd9613c60843e646e15f6bb9879f477c201d63824793859e5386bd8059d8f4f`，1513 个文件逐项锁定。它包括模型 fdce、公共修复和冻结依赖，不能称作裸 master，也不能用模型 Git tree 代替完整运行身份。最后查询的 master 是 `435ccc0287056af621c3f5fda0182e27284967e5`。详见 [源码清单](source-manifest.json)、[运行增量归属](review/runtime-dependency-delta.md) 和 [合入依赖表](review/pinned-dependencies-final.md)。

## 本次必要修改

- 配置创建并入 builder；Router、MLP、MoE 的训练适配并入 training.py，删除 configuration.py 和 moe.py 两个额外文件。
- decoder 保留官方 mHC 参数和 norm 对象，通过既有 plan 替换为共享融合实现，删除随后必被替换的中间包装和重复初始化。保留融合 pre 第四项 residual 的梯度路径，以及 post 的布局转换。
- EP 根据公共 expert-major 接口能力检查，不再绑定具体专家类；矩阵执行继续复用公共 GroupedExperts。保留官方 ModuleList 到打包权重的必要校验与可逆映射，不增加另一套专家执行器。
- 删除冗余模型类型检查，保留缺失融合计划的明确报错；撤回 _precision.py 纯文档改动。公共 MTP、MLA、Muon、CP 和重计算计划继续复用。

删除重复初始化会改变相同 seed 冷启动时的随机数消费，不能把“同 seed”当成“同状态”。本轮使用相同实际 checkpoint；没有补随机调用以复刻旧轨迹，也没有固定规约顺序或移植 MF 特制累加。

## 固定条件与比较口径

| 项目 | 单卡 MF→HP | 四卡 HP 自洽 |
|---|---|---|
| 模型 | 1 Dense + 1 MoE + 1 MTP，MLA，实际训练长度 4096 | 相同模型与长度 |
| mHC | 两侧融合，clamp/CMHC 关闭 | 相同 |
| 有效 batch | GBS=2、LBS=1 | 用户已授权 GBS=4、LBS=1 |
| 归一化 | 显式 legacy，保持 MF 微批等权目标 | 显式 token，按整步有效 token 计权 |
| checkpoint | MF 第 1500 步更新后完整状态转换到 HP | 同一纯 FSDP 第 1500 步完整状态，各配置独立恢复 |
| 窗口 | 1501–2500，连续完整 1000 步 | 每配置相同窗口，各 1000 步 |
| 日程 | 保留原 7500 步 LR 日程、1500 步 warmup | 相同；没有为短跑压缩日程 |

MF 固定参考来自 `Somnus2020/mindformers_formal_v1.0` 提交 `52f614eeb91c6e873d37bdbb5b880c82926ae9f6`、HP-MS r1.0.0 `c69f8f70`、MS 2.10.0、CANN 9.1.0 B243；这是已有真实连续 2500 步基线。HP 实测 Python 3.12.14、PyTorch 2.12.0+cpu、torch_npu 2.12.0.post3、Transformers 5.21.0，实际训练设备为 Ascend NPU；CANN 9.2.0 B237 按已归档安装记录和运行挂载锁定，compiler/version.info 本身仅证明 9.2.0，融合算子扩展另有冻结身份。MF 基线不因本次 HP 文件整理而重跑或改标新版本。

数据前缀 `mmap_deepseekv3_datasets_text_document`：
- .bin SHA256 `6b925f342504c3eae80c5bde3163ff5924ad36b262bc6a276839a6f3f817ea3c`
- .idx SHA256 `d535e502d62368e7a195b27a9a717543534ee1eaa9b4f9ce45f4d08d746cb5cd`

原始 mmap 含 930514 个文档，原始文档长度并非都为 4096；冻结的有序预处理截断/填充到 4097，再构造 4096 个训练位置。核验对象是实际输入，不是仅检查 YAML 的 seq_length。

LOSS 为主语言模型损失，MTPLOSS 为乘 0.1 前的 raw mean，均不把两者之和混作 LOSS。四卡采集已归一化的全局指标，各 rank 一致；不同 CP/TP 视图按同一优化步和全局样本对应。单卡计算 HP−MF，主要四卡验收计算待测配置−连续纯 FSDP 基线。最终阈值是完整 1000 步的 `abs(mean(delta)) <= 0.005`；符号保留到均值之后。MAE、最大绝对差、正负步数是诊断项，不另设阈值。

## 恢复与输入证据

单卡核验 MF1500 转换后的 65 个模型张量、91 个优化器槽、61 个 FP32 主权重别名，以及调度器、随机状态和数据进度。1000 步的 8000 个实际语义字段核验一致。HP2500 最终 checkpoint 独立 CPU 读回检查 65 个模型状态、91 个槽、734 个优化器元数据叶及随机状态，cursor=5000；CPU 读回不是 NPU 数值计算替代。

四卡连续 FSDP 基线的 0–2500、step1500 checkpoint、全局读回和逐 rank 记录均保留；checkpoint1500 为 13 文件、24,426,017,832 字节。EP/CP/TP 恢复先通过全局重分片读回；同布局 FSDP 检查全部 rank 的恢复值与 producer 相同，并引用独立源 checkpoint 全局读回。每配置核 4000 个全局样本及 CP 分片完整覆盖、TP 副本数量、实际有效 labels/mask/positions，学习率和步序一致。

masked-off label 的无效位置不声称与原始数据逐字节相同；CP 的有效字段通过冻结数据逐片核验，不将片段 SHA 拼接伪造全序列 SHA。原始记录哈希和范围见 [ledger-provenance.json](statistics/ledger-provenance.json)，逐步值见 [raw-values](raw-values)。

原生 checkpoint 保存 Torch CPU/device 和 Python RNG；NumPy 状态保留于实验 sidecar，原生 DCP 未保存。当前固定训练路径没有活跃 NumPy 随机消费者，不能把此结论推广到任意数据管道。

## 四卡配置范围

FSDP replay 与每个待测配置均从同一连续基线 step1500 恢复。当前 fdce 的 FSDP 是独立 1000 步 replay，**不是重新建立的 fdce 0–2500 连续基线**。除对旧连续基线的主要验收外，还报告各当前配置相对当前 fdce FSDP replay 的同版本差值。

| 配置 | DP / TP / CP / EP | FSDP shard | GA | 重计算与范围 |
|---|---|---:|---:|---|
| FSDP replay | 4 / 1 / 1 / 1 | 4 | 1 | 主干两层和 MTP |
| 仅主干重计算 | 4 / 1 / 1 / 1 | 4 | 1 | 主干两层；MTP 仍训练 |
| EP2 | 4 / 1 / 1 / 2 | 4 | 1 | 主干两层，专家按 EP 分组执行 |
| CP2DP2 | 2 / 1 / 2 / 1 | 4 | 2 | 主干两层，ring 路径 |
| TP2CP2 | 1 / 2 / 2 / 1 | 2 | 4 | 主干两层，Ulysses=2、loss_parallel=true |

TP2CP2 使用已验 plan，部分专家/共享专家参数保持 TP 复制；不是声称任意 TP/EP/SP 组合都已覆盖。FSDP 的分片维度包含 CP，表中 shard 不能机械当成 DP。所有正式配置以私有 YAML 和运行 spec 为准。

## 验证边界与保留材料

本轮 CPU 检查覆盖真实 replacement/provider、HF 部分加载、Muon native leaf hook、公共 MLA callback 和共享实际权重下的 decoder 前向/反向。CPU mHC 桩只证明接线；实际算子和训练精度由 NPU 完整运行补足。另有 4 进程 Gloo 的 DP2×CP2 token/aux/MTP 梯度独立参考，用真实 collectives 验证含空 CP 分片的情形；外部模拟 FSDP 平均，不替代 NPU FSDP。

125 的 backbone 尝试被外部进程占卡打断，未拼入验收；129 首次 backbone 启动因端口占用在设备准入前退出，换隔离端口后重新完整运行。旧中断和失败凭据保留。最终 checkpoint 读回第一版仅因读回器硬编码旧 head 而在打开 checkpoint 前退出；只更新读回器的两处身份绑定后重验通过，未修改产品或放宽数据检查。

采样包含逐步校验开销，且独立任务并行使用不同卡组；这些计时不能作为无干扰性能比较。本轮没有新增性能优势声明。既有 lint 诊断仍存在，不宣称 lint 或仓库 CI 全绿。PR 的 CLA 已通过，review quorum 仍 pending；当前 head 的 Label Gate 等待新鲜外部流水线标签，已按正常流程请求检查，保留 Draft 和真实状态。

原有总 issue #353 的正文完整归档于 [history](history/issue353-pre-final.md)；#304 未改。私有 YAML 不进入此公开包。关键 checkpoint、实际原始日志、命令、源码和逐步统计按任务目录保留；仍被长期容器挂载的旧 checkpoint 暂未删除。

## 完整窗口验收统计

| 比较 / 指标 | signed mean | abs(mean) | 正 / 负 / 零 | MAE | 最大绝对差 @step |
|---|---:|---:|---:|---:|---:|
| HP−MF 单卡 / LOSS | -0.0001479853 | 0.0001479853 | 470 / 530 / 0 | 0.00400050 | 0.03519988 @1835 |
| HP−MF 单卡 / MTPLOSS | -0.0000119462 | 0.0000119462 | 490 / 510 / 0 | 0.00385510 | 0.03009307 @2471 |
| fsdp_full−连续 FSDP / LOSS | +0.0002845216 | 0.0002845216 | 536 / 463 / 1 | 0.00240828 | 0.01817729 @2124 |
| fsdp_full−连续 FSDP / MTPLOSS | +0.0000204408 | 0.0000204408 | 488 / 512 / 0 | 0.00327829 | 0.02093315 @1964 |
| fsdp_backbone−连续 FSDP / LOSS | +0.0002001138 | 0.0002001138 | 531 / 467 / 2 | 0.00238176 | 0.01488414 @2356 |
| fsdp_backbone−连续 FSDP / MTPLOSS | -0.0000456579 | 0.0000456579 | 504 / 496 / 0 | 0.00341879 | 0.02097318 @2289 |
| ep2−连续 FSDP / LOSS | +0.0000845105 | 0.0000845105 | 514 / 485 / 1 | 0.00239396 | 0.01628426 @2263 |
| ep2−连续 FSDP / MTPLOSS | -0.0000398258 | 0.0000398258 | 495 / 505 / 0 | 0.00346399 | 0.02018471 @2320 |
| cp2dp2−连续 FSDP / LOSS | +0.0001192427 | 0.0001192427 | 515 / 485 / 0 | 0.00243374 | 0.01513897 @2400 |
| cp2dp2−连续 FSDP / MTPLOSS | +0.0000890005 | 0.0000890005 | 502 / 498 / 0 | 0.00337114 | 0.02236281 @1906 |
| tp2cp2−连续 FSDP / LOSS | +0.0000976843 | 0.0000976843 | 512 / 488 / 0 | 0.00231052 | 0.01644351 @2448 |
| tp2cp2−连续 FSDP / MTPLOSS | +0.0000412105 | 0.0000412105 | 506 / 494 / 0 | 0.00341216 | 0.02157542 @2431 |

## 同版本 FSDP replay 对照

| 比较 / 指标 | signed mean | abs(mean) | 正 / 负 / 零 | MAE | 最大绝对差 @step |
|---|---:|---:|---:|---:|---:|
| fsdp_backbone−fdce FSDP replay / LOSS | -0.0000844078 | 0.0000844078 | 468 / 531 / 1 | 0.00233877 | 0.01828769 @2124 |
| fsdp_backbone−fdce FSDP replay / MTPLOSS | -0.0000660987 | 0.0000660987 | 501 / 498 / 1 | 0.00330567 | 0.01696115 @2314 |
| ep2−fdce FSDP replay / LOSS | -0.0002000110 | 0.0002000110 | 465 / 534 / 1 | 0.00243982 | 0.01804777 @2416 |
| ep2−fdce FSDP replay / MTPLOSS | -0.0000602666 | 0.0000602666 | 514 / 485 / 1 | 0.00320189 | 0.01923584 @2320 |
| cp2dp2−fdce FSDP replay / LOSS | -0.0001652789 | 0.0001652789 | 476 / 524 / 0 | 0.00233773 | 0.01654881 @2405 |
| cp2dp2−fdce FSDP replay / MTPLOSS | +0.0000685597 | 0.0000685597 | 516 / 484 / 0 | 0.00315844 | 0.02506209 @2480 |
| tp2cp2−fdce FSDP replay / LOSS | -0.0001868373 | 0.0001868373 | 460 / 540 / 0 | 0.00232925 | 0.01629201 @2124 |
| tp2cp2−fdce FSDP replay / MTPLOSS | +0.0000207696 | 0.0000207696 | 490 / 510 / 0 | 0.00333130 | 0.02349304 @2151 |

以上每行均为完整1000步，两个指标各自计算；所有均值阈值通过，差值自然出现正负。实际原始数值保留全部精度，表格仅展示舍入值。

## 图表与复核

网站导出和实时页面状态见 [图表说明](charts/README.md)。数据不平滑；主图、差值和累计均值分别展示，0.005 只约束完整窗口最终均值。

按 CSV 独立复算：

```python
import csv, math
r = list(csv.DictReader(open('raw-values/single-mf-hp.csv', encoding='utf8')))
assert [int(x['step']) for x in r] == list(range(1501, 2501))
for k in ('loss_delta', 'mtp_delta'):
    d = [float(x[k]) for x in r]
    print(k, abs(math.fsum(d) / 1000), sum(x > 0 for x in d), sum(x < 0 for x in d),
          math.fsum(map(abs, d)) / 1000, max(map(abs, d)))
```

四卡按 configuration 分组后使用相同公式；原始 ledger 校验脚本在 verification-scripts，所需原 ledger 哈希在统计文件。私有运行配置与脚本需依实际归档路径及完整 checkpoint，不能仅凭公开指标 CSV 重新训练。

私有配置：[七份实际 YAML 与运行 spec](https://github.com/xingleixu123/mindformers-private/blob/95044304de283b3439a0bd1595313161b0f93016/archives/xing4-fdce-full-precision-20261105/README.md)。发布时仓库为个人私有、唯一 collaborator 为 owner，公开链接不会授予访问权。
