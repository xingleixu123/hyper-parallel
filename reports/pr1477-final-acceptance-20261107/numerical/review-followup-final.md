# PR1439 全部检视：最终后续

2026-11-06 21:06 UTC重新读取在线记录：82 inline、50 reviews、16 conversation，共148项，按ID/body/path/reply/updated_at比对无新增、删除或修改；44线程原处置矩阵继续适用。#1439原提交与远端线程状态未修改，#304/#353未修改。

原[逐项矩阵](review-disposition.md)冻结的是#1477原head eb112147，表中待处理字样是当时状态。当前独立后续如下：

| 来源 | 独立问题／PR | 实际范围与结论 |
|---|---|---|
| T31/T32 | mindspore-ai#1456／[JT Draft#363](https://github.com/hf1s/hyper-parallel/pull/363) | 非空swiglu_limit/compute_dtype静默接受但不消费的契约；两行拒绝未支持选项。本次两项为空，未叠进精度源码，不是本次验收必需。正式支持应复用公共JT#356。 |
| T40/T44及router域归属 | [JT#338](https://github.com/hf1s/hyper-parallel/issues/338)／[Draft#364](https://github.com/hf1s/hyper-parallel/pull/364) | 复用DS4.1公共sequence AUX和resolved CP/TP placements，将router统计移出CP文件；本轮source496和3292包含。统计模块证据与原生LP倍率证据分开，不冒称原12case显式TP平均即LP验证。 |
| 本轮实际原生因果实验 | [JT#365](https://github.com/hf1s/hyper-parallel/issues/365)／[Draft#366](https://github.com/hf1s/hyper-parallel/pull/366) | 显式完整TP副本AUX在loss_parallel下被原生SUM重复累加；CP/TP与EP/SP均旧2×/新1×，非LP梯度字节不变。两TP整网1000步和完整状态在3292组合中通过；不改Muon或规约顺序。 |

单卡需要1470→1477→1441；四卡需要1470→1477→1453，再按触发范围叠364和366。1453已含1441的多foundation绑定，四卡不重复挑1441。1420已用于运行，但没有独立证据证明本次入口必触发，列导入顺序条件项。packed MLP分片351/1429、reader拓扑285、Muon冗余动量1431、swap hook1459等写清触发，本次不把它们当精度前置。1477已经替代旧模型路线1439/1413，且含1425 CP增量，不再重复合入。

三新增PR均Draft、各自单个增量commit，不改原review历史。当前可信外部CI尚未回写，label-check failure按[current-pr-ci.json](current-pr-ci.json)保留；本地／NPU验收不冒称已具备合入绿灯。
