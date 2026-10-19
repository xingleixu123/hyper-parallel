# 125/129 FSDP8：完整200步主loss逐值一致

两机各一次、同源544526f7的完整200步×8rank对照中，主LM最大绝对差为 **0**，超过0.005的步骤为 **0/200**。四项loss、grad norm、LR和四项data counters共10个字段均零差；CPU初态记录、runtime与逐步输入指纹一致。这一结果支持继续在125/129调度本任务，不改变其他并行配置仍未达标的事实。

![完整200步对照](authored/figures/control200-main-lm.png)

已完成[独立数据与报告复核](independent-review-summary.json)：183份原始文件SHA复核，10字段逐rank逐步重算；从原始日志重建数据并独立重绘，CSV、摘要、来源表、PNG、SVG均逐字节一致。8rank是归约后的同值标量，不能当成8次独立重复。

环境为CANN9.2.0-beta.2、同一固定镜像和集成验证源码544526f7；544526f7不是上游JT_dev分支头。两次recipe仅save_ckpt true→false及输出目录有差。MTP/aux是已有带系数日志及系数还原观察，不构成其全并行专项通过；本报告不推断性能收益或optimizer内部字节一致。

125的远端raw与本机canonical scheduling原文各自保留SHA，解析JSON一致。原strict直接路径检查的FAIL/weighted PASS/unweighted UNVERIFIED历史保持原样；新的显式映射检查闭合两种表示，未冒充原strict通过。

[详细中文报告及复现方法](authored/README_ZH.md)、[200行CSV](authored/data/control200.csv)、[绘图源](authored/tools/plot_control200.py)与[原始作者封版manifest](authored/manifest-v1.json)完整保留。authored子目录是独立审核前的原封版，正文里的“待审核、未发布”及review-status是当时状态；本页与独审摘要给出后续实际审核结果。该包可从CSV重绘；从raw重新验全部原始证据仍需任务私有保留原文，未复制checkpoint或大张量供体。
