# 40步观察透明性与单卡探针前置状态

40步off对照 `r129-f637-module-off-40-v1` 已完整完成：8rank各40步，退出0、197.626085秒后释放。对同源FSDP8基线前40步，LM、MTP、aux、总loss及梯度范数误差均为0；输入、初态、学习率与计数检查通过。严格200步比较器仍保留唯一未满足项“Final acceptance requires exactly 200 consecutive steps”，不将40步作为新的200步验收。

前3步共9个模块边界计数正确、off无重放，8rank的计数hook与callback均移除。累计allocated/reserved峰值22.032196/27.25GiB，21–40步全rank MAX中位1.155995秒；这是控制组观察，尚不能给出norm/dense/MoE重算成本收益。原有off/full/selective200步结果仍独立保留。

单卡探针v1 `r129-f637-pta-levels12-1npu-v1` 仅映射129的0号卡，6.296041秒后退出并释放。coordinator错误解析venv Python软链接，子进程变成系统Python，`import torch`失败；没有RMSNorm或Linear结果、没有level2结果，不能得出任何一致性结论。该问题属于本任务诊断启动器，未混入产品修复PR。已准备v2修正并增加真实子进程检查，执行前仍须满足20分钟间隔。

`completed-control-and-probe-v1.tar.gz`保留两次执行的130份原始文件及释放凭据，配套manifest逐SHA/大小核验并逐成员回读。`fixed-pta-probe-v1-execution-inputs.tar.gz`保留v1的22份精确执行输入，供检查失败原因；两类包用途不同，不是重复传输归档。原始训练样本未上传。

用户最新指示恢复：125/129只使用空闲卡，不停止他人；实验结束确认释放后至少1200秒再启动下一轮。短失败也按该间隔处理。先解决主LM loss完整200步0.005对齐，达标后先汇报，再由用户决定MTP/aux专项是否继续。
