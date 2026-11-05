# 最新 master 与依赖核验

本轮开始为 master 6c8ccfac（#1386 公共MTP合入）。其 components/modules/mtp.py 与 components/losses/mtp.py 和已验证冻结组合571d逐字节一致，故无需重复cherry-pick，也不因同内容合入而重跑MF基线。

提交前新查到 master 7b1ea07e8c557551f10104d7479b55d325e15543，合入 #1434，逐文件审查了相对6c8ccfac的增量：

- DS4.1 packed graph输入和attention元数据；
- RuntimeInputAdapter.prepare_graph_inputs（默认原样，仅图模式调用）；
- MoEMonitor.last_mean_aux_loss从Python浮点改为设备上标量Tensor；source last_aux_loss已detach；
- Trainer及logging增加micro-step-end和日志指标转交。router_aux_loss被加入loss_dict日志输出，未加入本次训练模型foundation/MTP反向目标；
- 不改变sync_and_update_expert_bias/on_step_end的计数归约或更新数学；
- 与本次六个生产文件无重叠，不包含本次int64持久计数或FP32 buffer保护。

实际NPU训练仍使用保留的完整依赖组合+六文件overlay，每份1514文件逐项锁定：单卡0b2f30d2501c5cf8b701015200bed5b4ed46e9a95153befd4575370643aa0b2c，四卡2b08a0f01a77cc078730275463a9072b65e8f6b095bb7b21a5952a228f55d0ae。不能将该结果表述为整棵7b1ea07e或裸#1439单提交的验收。

当前PR保持已有四个提交，追加一个提交；没有rebase、amend或复制#1386/#1434来改变检视历史。合入顺序与条件依赖以dependency-final.json为准。#1425已关闭且公共CP功能包含于#1439；#1455不再是Xing4的必要依赖。

#1203依然开放，其路由计数/偏置策略不满足本模型padding排除、重算排除、非EP和未中心化sign更新契约，不能当作现成等价实现直接换入。
