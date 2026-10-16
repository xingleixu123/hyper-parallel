# JT 单卡200→400：三项loss同时达标

第100组独立完整200步通过：总loss首步≤0.0005，原始MTP/Aux首步各≤0.001；三项每一步≤0.005。
最大误差分别为1.90735e-6、3.17891e-6、1.29733e-6，全部200步梯度范数精确一致。

- [最终完整报告、优先级、配置、正确性及清理](总问题-三项loss达标.md)
- [完整200步审计](evidence/frontier396-shared-independent200-audit.json)
- [独立原生观测保真与396/397边界](evidence/frontier396-shared-native-statistics-comparison.json)
- [单因素完整因果链](evidence/frontier396-causal-chain.json)
- [最新MF master PyNative、Megatron与FP64正确性对照](阶段记录-396归一化正确性与行业对照.md)
- [100组实验清单](evidence/completed-cann92-inventory.csv)
- [29份本阶段源码清单与执行范围](evidence/frontier396-final-source-manifest.json)
- [独立诊断程序](scripts/shared-down-pr/README.md)
- [历史总loss阶段](总问题-总loss达标.md)和[此前README](README-历史总loss阶段.md)

只包含可编辑报告、源码、配置、原生IR/统计、逐步指标和完整hash，不含用户数据或完整模型/梯度数组。
新归约因素用于复现指定旧MF图，不代表HP普通归约是数学错误；产品默认保持独立判断。
目标数组已清理，重新运行单步replay需要先按采集脚本再生，不能用哈希代替数组。
