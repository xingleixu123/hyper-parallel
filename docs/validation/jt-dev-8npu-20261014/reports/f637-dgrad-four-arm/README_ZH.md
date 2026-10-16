# CP2 精度定位：四臂 Linear 输入梯度实验

**仅将本地 BF16 Linear 的输入梯度改为 FP32 计算，未使 CP2 与 FSDP8 整体更接近。** 同配方跨布局比较中，三步 LM 最大差减小，但 MTP、aux 最大差增大；首步采样梯度的不同坐标数也增多。因此本实验保留为排除性证据，没有据此新增生产修复 PR。现有 CP2/TP2/EP2 完整200步仍未满足0.005；本目录不改变该结论。

![原生与FP32输入梯度配方分别作跨布局比较](../../figures/f637-dgrad-four-arm-zh.png)

## 实验设计与边界

四组均在129、同一固定镜像 `sha256:5f9e677f439d11ab4e7c888f54b142335a27eee7411cbd2ce894a4798a70146d`、CANN9.2.0-beta.2、集成源码 `18602c4524dc664b81b2bad9a7b624455ce1494f` 上独立启动。JT_dev 基点为 `f637ef7`，包含#180。数据、初始化、GBS8、序长4096、LR和其他数值配方与当前200步参考一致；四份真实parser配置已逐字段校验，仅允许相应窗口/保存目录等计划差异。

| 臂 | 独立case | 布局 | Linear输入梯度 |
|---|---|---|---|
| A | `r129-f637-dgrad-fsdp8-native-3` | FSDP8 | 原生BF16 |
| B | `r129-f637-dgrad-cp2-native-3` | FSDP+CP2 | 原生BF16 |
| C | `r129-f637-dgrad-fsdp8-fp32-3` | FSDP8 | FP32运算后转回BF16 |
| D | `r129-f637-dgrad-cp2-fp32-3` | FSDP+CP2 | FP32运算后转回BF16 |

C/D只处理符合条件的本地普通BF16 Linear操作数：前向仍用原生实现，输入梯度以FP32执行 `dY @ W` 后转回BF16，权重梯度使用原生BF16重放算法。上游dY会改变，不能说权重梯度的数值保持不变。DTensor、router、GMM与attention的其他计算不在此干预范围。HF32逐次观察均为false。FSDP8每步1个microbatch、CP2每步2个；C共576次、D共1152次实际 `derivative_calculated` 事件，不能仅看“安装成功”就称实际生效。

观察覆盖首步8个唯一逻辑输入；首步之后清除batch身份，后两步记录值为null。每个样本按输入SHA、模块/调用/阶段和全局坐标关联。dtype、逻辑全局尺寸须相同，允许CP局部序长4096→2048；重复、缺失、错误坐标或签名会拒绝。观察只采样stride128，不能推出全张量相等。非Linear观察集合不是整张反向图的完整覆盖。

## 三步全部8个rank的标量差异

| 对照与含义 | LM最大绝对差 | 日志MTP最大差 | 还原MTP最大差 | 日志aux最大差 | 还原aux最大差 | 梯度范数最大差 |
|---|---:|---:|---:|---:|---:|---:|
| A−B：原生配方跨布局 | 0.000398635864258 | 0.000127553939819 | 0.000425179799398 | 4.77448338643e-8 | 0.000477448338643 | 0.00443840026855 |
| C−D：FP32 dgrad配方跨布局 | 0.000271797180176 | 0.000285863876343 | 0.000952879587809 | 8.98871803656e-8 | 0.000898871803656 | 0.00579071044922 |
| A−C：FSDP8布局内干预 | 0.000745773315430 | 0.000329971313477 | 0.00109990437826 | 1.77824404091e-8 | 0.000177824404091 | 0.00152778625488 |
| B−D：CP2布局内干预 | 0.000697135925293 | 0.000309467315674 | 0.00103155771891 | 3.33384377882e-8 | 0.000333384377882 | 0.00158882141113 |

MTP/aux“还原”值分别为日志值差除以0.3/0.0001，没有另外采集加权前张量。不同指标的最大值可能出现在不同步。完整逐步数值、total loss、输入/初态及计数检查在[标量对比](analysis/f637-dgrad-four-arm-scalar-comparison.json)。原严格比较器未修改，每组仍保留 `Final acceptance requires exactly 200 consecutive steps`，三步不能标作200步验收通过。

原生观察臂A/B相对各自既有完整200步参考的前三步，全部8rank loss与梯度范数差均为0，说明在本次短窗口中未观察到采集改变原生数值。C/D改变了数值配方，不使用此“透明性”结论。A−C/B−D也不能替代C−D来声称跨布局精度改善。

## 首步采样边界

| 同配方跨布局 | 前向组/坐标 | 前向不同坐标 | 梯度组/坐标 | 梯度不同组 | 梯度不同坐标 | 梯度最大绝对差 |
|---|---|---:|---|---:|---:|---:|
| A−B | 72 / 1,747,200 | 0 | 70 / 1,138,944 | 66 | 193,767 | 3.4332275390625e-5 |
| C−D | 72 / 1,747,200 | 0 | 70 / 1,138,944 | 66 | 200,825 | 3.814697265625e-5 |

LM head原始call0的输入梯度差在C−D仍存在：3/10,240个采样坐标不同，最大5.960464477539063e-8；A−B为4/10,240、同样最大值。两组MTP decoder输出梯度均有3/10,240不同，最大3.637978807091713e-12。未记录MTP `final_layernorm` 的完整输入/反向边界，不能据此将它认定为唯一根因，也不能把hook回调记录顺序称为整图因果顺序。

CP本地loss在 `before_cp_sum` 的标量反向系数与匹配FSDP样本一致；这是按真实逻辑输入逐batch匹配的结论，不是将CP根节点导数统一乘2。CP根节点各副本可能有不同局部token权重。采样激活的前向相同，也不代表根loss标量每个batch完全相同。

见[全部142组对比及四种对照](analysis/f637-dgrad-four-arm-activation-comparison.json)、[原生loss导数](analysis/f637-dgrad-native-loss-derivatives.json)、[FP32配方loss导数](analysis/f637-dgrad-fp32-loss-derivatives.json)和[独立复核](analysis/f637-dgrad-four-arm-independent-review.json)。

## 资源约束、性能与后续选择

四轮退出码均为0、8rank各完成3步、释放后无本任务残留PID；每轮约137–142秒。原始主机时钟receipt显示相邻轮释放→启动间隔分别约1222.239、1221.270、1271.974秒，另有本地wall/monotonic守卫和空卡复查。没有停止他人实验。这是四臂运行时的历史间隔策略。用户随后明确取消固定20分钟等待；后续就绪实验经空卡和释放确认后连续运行，不停止他人任务。

这些短跑有首步高频梯度采集、CPU复制及FP32临时矩阵，不能作生产吞吐/显存收益排名。例如词表head仅FP32 dY与W两项约5.22GiB，再计FP32 dX约5.30GiB（序长4096、词表152064、hidden5120的静态张量容量估算，不是实测峰值）。当前结果不支持承担该成本进入生产。已有完整200步资源比较仍以[资源总览](../f637-resource-overview/README_ZH.md)为准。

后续优先在固定PTA中区分行位置/分区引起的算子舍入差，准备单卡level1/2及真实RMSNorm/Linear的窄范围对照。level2尚未经过本任务硬件验证，也不能把更晚版本的level3 API套用到当前镜像。[候选的官方源码边界](analysis/fixed-pta-precision-candidate-boundaries.md)。模块重计算成本则按已有计划另行实测。任何数学配方变化都需重建匹配的FSDP0–200、续训200–400及独立连续400参考。

## 原始证据和复算

[四臂原字节包](four-arm-inputs.tar.gz)为32,429,346字节，SHA256 `847448eeee007ae513fc9521af499ceb03c4aed58cb0ca6a45755e90e3949231`，解包共554文件、206,589,224字节。包含四臂全部520份原始run/receipt文件、各自130文件的原始采集manifest、请求/配置及冻结工具。不是只有图表或经过截断的指标。原始训练数据文本未包含在内。包的[完整大小/SHA清单](four-arm-inputs-manifest.json)已无损回读核验。

在Python3环境中恢复到尚不存在的目录（包与恢复脚本须同目录）：

```bash
python restore_inputs.py /absolute/path/to/new-evidence-root
cd /absolute/path/to/new-evidence-root
python tools/analyze_dgrad_four_arm_scalars.py
python tools/analyze_dgrad_four_arm.py --output evidence/f637-dgrad-four-arm-activation-comparison.json
python tools/audit_dgrad_short_run.py runs/r129-f637-dgrad-cp2-native-3 runs/r129-f637-dgrad-cp2-fp32-3 --reference-manifest evidence/f637-dgrad-cp2-native-3-evidence-manifest.json --candidate-manifest evidence/f637-dgrad-cp2-fp32-3-evidence-manifest.json --output evidence/dgrad-d-recomputed.json
```

绘图另需matplotlib，中文字体可用Microsoft YaHei；工具以恢复目录为任务根。对比工具按单个逻辑batch处理，避免构建全部百万坐标字典。恢复工具拒绝已有目标、越界路径、重复/额外成员及SHA不一致文件。本机已实际恢复全部554文件并复算四组标量，结果字节一致，见[恢复及清理验证](analysis/f637-dgrad-package-restoration-verification.json)。原A/B观察透明性审计所用的完整参考另见[FSDP200证据](../f637-fsdp8-r4-complete)与[CP2完整200证据](../f637-cp2-200-complete)。

本目录的分析JSON、独立审计、CPU/前置检查和固定源码材料在`analysis/`与`references/`中保留。原始材料的CRLF和空白按SHA保留，局部Git属性仅对这些证据及生成SVG设置相应空白检查边界，不改训练源。关联[中文总issue #202](https://github.com/hf1s/hyper-parallel/issues/202)；14个已确认独立问题继续各对应中文小issue及单提交草稿PR。
