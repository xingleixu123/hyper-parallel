# JT单卡200→400：总loss阶段已达标

第99组独立200步已完成：首步绝对差9.5367431640625e-7，全部200步最大0.00276947021484375，
平均0.000021147727966308594，0步超过0.005。当前门槛为总loss首步≤0.001、每一步≤0.005。
按用户约定已停止进一步训练，MTP/Aux专项待决定；原始MTP仍有1步差0.009670257568359375。

- [完整中文总报告](总问题-总loss达标.md)：配置、完整状态恢复、消融优先级、正确性、独立issue/PR、复现和限制。
- [第328步专家归约因果及完整复验](阶段记录-328专家归一化及分阶段定位.md)。
- [99组完整实验清单](evidence/completed-cann92-inventory.csv)：另3组完整teacher干预不计独立验收。
- [第99组200步逐步审计](evidence/frontier328-expert-independent200-audit.json)。
- [329/330原生统计及48项模型](evidence/frontier328-expert-native-statistics-comparison.json)。
- [原总issue完整归档](总问题-306阶段原文归档.md)与[历史README](README-历史截至306阶段.md)：保留早期结论、负面消融和固定证据链接。

![200步全部分项误差](evidence/frontier328-expert-independent200-loss-errors.png)

最后四组使用JT_dev e21302f9＋PR192集成181fab5f，包含PR180；实际Ascend910B3单卡，
HP CANN9.2.0-beta.2、MF CANN8.5.0，MF用户原包未改。所有运行均从同次MF200完整模型和
28份Muon动量、18＋18份AdamW矩恢复。最后一个因素只改变实际专家down输入的归约顺序，
无teacher数据、步号或坐标特判。它是旧归档兼容因素，不是已证明的HP通用数学错误。

目录保留源码、配置、未截断标量指标、日志、原生统计/hash、失败记录和清理回执；
不含用户训练数据、模型或优化器大数组。环境路径需按复现机器映射，初始状态需来自同次MF200。
已清理的328/329边界数组需用对应入口重新生成，不能直接运行依赖这些数组的分析。

`文件校验.json`按实际Git blob字节列出hash。`scripts/latest-jt-f637`对应冻结的旧工具层，
`scripts/tools-e213`对应最新入口；本阶段源路径映射见`evidence/frontier328-final-source-manifest.json`。
本目录为追加式实验历史，不重写旧提交或permalink；数值通过不等于产品PR已合入或CI已绿。
