# PR1477 单卡与六种四卡布局：完整窗口数值与状态证据

实际完成范围、原 wrapper 状态见 [matrix-summary.json](matrix-summary.json)。本目录仅由严格 allowlist 在全部七组真实1501–2500窗口、样本、native2500及独立CPU状态回执闭合后生成。这里没有从first10外推的值、图表或性能比较。

每组 `cases/<layout>/primary-pairs.csv` 保留1000行真实标量，`statistics-and-state-summary.json`列signed mean、正负/零数量、MAE、最大绝对差及step，并单列原始controller/CPU/closure/audit SHA。必需指标为 `abs(mean(actual-reference)) <= 0.005`，分别检验主LM LOSS与未乘0.1的raw MTPLOSS；不把LM+.1MTP总loss当主LOSS。

单卡参考原MF连续2500并从canonical MF1500恢复，数学目标为legacy微批等权。六种四卡从同一旧FSDP1500恢复；主LOSS/rawMTP参考旧连续a852的1501–2500，数学目标为整步有效token加权。四卡AUX只对新source496 FSDP完整全rank/micro/router采集；`AUX-pairs-vs-new-FSDP.csv`为观察项，没有AUX阈值。单卡原MF raw AUX单独按其legacy口径保留，不能与四卡AUX混比。

源码身份严格区分：单卡49bffc（1477+1420/1470/1441）；四组TP1为496788（1477+1420/1470/1453/364）；两组TP/LP为3292a5（在496上叠366/f8ec显式router副本均值）。细节、版本与并行/重算元数据在 [source-version-parallelism.json](source-version-parallelism.json)。不称裸1477通过，不把这三种源码混成同一实验。

单卡已完成MF参考仍为冻结52f。较新formal MF62ab已完成三案原生两步恢复／兼容与同源同设备重复控制，结合精确六文件和活跃GMM调用合同，未发现使52f长基线失效的已确认数学变化；不宣称62ab完整1000步通过，见[有限审查](mf-upstream-semantic-probe.md)。本数值子包不生成图片；上级[网站导出](../website/README.md)已在线后台验证并导出，[汇总图](../figures/README.md)只将相同原值重新排版。普通Edge可见交互核验受URL安全限制，外部CI亦未宣称通过。

两组FSDP原wrapper非干净退出由有界post-native完成审计支持，原false状态保留。TPCP/SP的实际原rc及是否需要有界审计按各自新原始回执记录，不以其他配置的rc代替。TP0 reader应推进到10000，TP1保留原生nonreader6000；四份extra按真实角色闭合，不伪造全rank10000。

CPU摘要覆盖全局65模型/91槽、四rank实际局部hash、643原生optimizer option bytes、3组及4extra的scheduler/RNG/游标；单卡对应四文件，其余13文件。FP32 main alias只核活体证明，CPU没有重新构造对象/alias。公开只保留RNG域与闭合结果，不含RNG状态内容。

[保留与只读封印摘要](checkpoint-retention-summary.json)来自小位置索引和seal回执，不公开实际私有路径。后续chmod只读改变ctime单独记录，原CPU阶段identity/SHA闭合证明不重写；本包没有重扫tensor/metadata。

[历史44线程/148项处置](review-disposition.json)和[表格](review-disposition.md)保留原head快照；新365/366、统计域364及其他独立根因链接见 [attribution-links.json](attribution-links.json)。原4ae审计包和11f7 synthetic原生AUX梯度包冻结保留，以固定commit链接复用；12case的显式TP均值不冒充原生LP/FSDP证明，ContextVar观察器失败不被归为产品bug。

配置仅指向JSON中私有归档链接，不附任何YAML、完整字段映射、模型/optimizer权重、真实token输入、原生RNG、凭据或环境全dump。执行 `python -B verify_final_package.py` 可离线重算封印、全部主指标和review覆盖；无需Torch、网络、NPU或checkpoint。

[最新review后续与必要PR](review-followup-final.md)区分原head处置矩阵及三个独立增量；原44线程／148 timeline没有被改写。
