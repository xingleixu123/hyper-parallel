# 新formal master的有限影响审查

本轮完整单卡参考保留实际52f614ee身份。新formal master62ab670a的精确六文件净差异仅重入heavy-op bridge与GMM实例生命周期等内容；当前原生resolved路径为non-reentrant、MC2关闭、router-only exclude。新重入/MC2桥不执行，不推广到这些未测配置。

会执行的GroupedMLP差异为同一GroupedMatmul(split_item=3,group_type=0)从每次构造移到实例属性；两次调用八个位置实参、transpose默认False和list返回取output[0]合同不变。实际MS2.10 primitive源码只将同八操作数与四attrs送同pyboost入口，没有Python tensor/参数状态写入。新helper的Tensor直返用于桥；未做逐调用返回type hook或底层compiler缓存测量。

旧52f/a1、新62ab/a1、同源同设备的旧52f/a2均从原MF1500完整状态实际运行1501–1502，三次原生/controller均clean0。237项加载值、38 FP32 masters、83 alias、实际样本、适用RNG与LR/原日程逐项闭合；1501完整原生观测行及三raw loss exact。old-old第一次更新后也有4 model/29 optimizer-only hash差，新-old为5/27；最早model均mHC mapping_proj，expert/GMM权重/槽及FP32 master当时未变。全FQN、计数和原三loss见[公开小投影](mf-two-update-source-projection/README.md)。

同源重复本身非逐位稳定，故不能把一次new-old hash非exact单独归因于persistent GMM；也不能用loss接近证明正确、从SHA推断误差幅度或用一次重复估计variability envelope。结合当前适用路径和未改变的算子/实参/输出合同，没有已确认事实使52f长基线失效，继续复用该冻结基线，停止低信息重复长跑。

**62ab仅完成上述有限兼容和控制检查，没有执行新62ab完整2500或1000步验收。** 未覆盖其reentrant桥VJP、MC2启用或其他配置，不补造通过声明，不改规约顺序、阈值、学习率或模型配置。小投影内published:false保留其本地导出时的历史快照，实际发布身份由上级Git commit及manifest记录。
