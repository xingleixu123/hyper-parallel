# 第229步 Muon 归一化：原生 Dump 与独立消融

## 当前结论和验收

总 loss 尚未达到目标。第91组独立完整200步只增加2D NS的epsilon字面量类型转换；201–230每一步全部48个FP32/BF16模型张量均精确匹配MF，原229参数首差消失。完整201–400窗口最大误差0.239593505859375、平均0.009304693937301635，66步超过0.005；首步误差9.5367431640625e-7满足0.001。局部逐位一致和最大值改善不代替全程达标。

当前只验收总loss：201首步≤0.001、201–400每步≤0.005。通过后先与用户交互，再决定MTP/Aux专项工作。下表分项作为诊断记录。累计91组独立完整训练无总loss通过，3组完整参考值干预诊断不计验收。

## 配置和实验来源

125服务器Ascend910B3，单卡、4K Arrow、Muon＋AdamW；HP CANN9.2.0-beta.2、JT_dev固定f637ef7ee4d4b416800716f1ebaaae7709b8a06a，带PR192的集成提交9213a97147020fb75e7a98a40f7e7b636f98d349；MF仍使用指定压缩包、MindSpore2.7.2、CANN8.5.0。MF原包文件未修改。完整MF200模型、28项Muon动量和18＋18项AdamW矩恢复；两边独立继续训练，不读取MF未来状态改变HP训练。

- [短程配置](configs/hp-jtf637-epsilon230.yaml)、[200步配置](configs/hp-jtf637-epsilon400.yaml)、[30步入口](scripts/latest-jt-f637/capture_hp_epsilon230.py)、[200步入口](scripts/latest-jt-f637/capture_hp_epsilon229.py)。
- 对照为`hp-jtf637-ce-tail230/400`，唯一新增因素是[epsilon类型因子](scripts/latest-jt-f637/hp_ns_epsilon229_factor.py)。其他既有数值兼容因素、配置、恢复、输入、49个冻结来源及实际调用均审计。
- epsilon由`float(torch.tensor(1e-7,dtype=torch.bfloat16).float())`得到`1.0011717677116394e-7`，只作用于2D默认归一化。3D和外部提供的归一化输入保持原行为。没有步号、参考数值或训练坐标特判。
- 每步26次命中，8种实际形状；200步共5200次。完整运行前30步指标及梯度范数与独立短程逐值相同。

## 从实际训练逐步排除输入和状态

CE尾部兼容因素后，227/228全部48项模型精确，229只有MTP `q_b_proj.weight`不同。两侧29步实际探针的全部指标、输入及恢复与各自无探针基线一致，并关联独立模型哈希。

| 229目标参数相关边界 | 完整元素数 | 比较结果 |
|---|---:|---|
| 228权重 | 37748736 | 全部逐位一致 |
| 228 Muon动量 | 37748736 | 全部逐位一致 |
| 229裁剪后梯度 | 37748736 | 全部逐位一致 |
| 229更新后Muon动量 | 37748736 | 全部逐位一致 |
| 229更新后权重 | 37748736 | FP32不同15803359，BF16不同118163 |

权重最大绝对误差7.56140798330307e-6。差异在NoPE部分；RoPE部分相同。将完整真实状态与梯度送入原HP、原MF Muon更新，分别精确复现各自真实权重与动量，说明本次无需更新后QK clip就可复现差异。该重放属于诊断，不计为独立训练。

[实际状态/梯度审计](evidence/frontier229-state-comparison.json)、[原生重放边界](evidence/frontier229-muon-boundaries.json)。

## MindSpore原生Dump的证据

按用户建议核对[2.7.2官方Dump文档](https://www.mindspore.cn/tutorials/zh-CN/r2.7.2/debug/dump.html)，保持原归档更新函数，使用`MINDSPORE_DUMP_CONFIG`在进程启动前配置。先对单参数原生重放的第0次迭代保存统计/IR，再精确选择两个归一化融合算子；没有插入源码TensorDump，也没有打开整网所有步骤的大张量Dump。

- [统计配置](configs/frontier229-mf-builtin-stat-dump.json)：单图、迭代0、统计值与最终IR。
- [张量配置](configs/frontier229-mf-builtin-tensor-dump.json)：`Default/GraphKernel_Sqrt_Add_Div_Cast_split-op0`和`op1`，输入与输出，只有6个数组，共301990664字节。这里迭代0是第229步单参数重放的第1次执行，不是训练全局第0步。
- 两次Dump运行均退出0，完整权重SHA为`c2099d3c74a33b5490ac42fdeb31305cbd53bdcbc28eb4473dc7af04aa202013`、动量SHA为`b61436e25662814a5c89add5863dd4d10fa644045537687278f3ae9f5584f948`，与实际MF229精确一致。
- 原生NoPE平方和为FP32 `7.337770966842072e-6`，RoPE为`4.452397479326464e-5`。原生输入与HP输入全部一致。原生NoPE归一化输出与原HP存在201015个BF16差异；只改epsilon的HP重放后，两部分的完整归一化输出和最终参数更新全部精确匹配MF。
- [原生DVM文本](evidence/frontier229-builtin-native-dvm-0.txt)第416/612行与对应Adds指令直接显示，旧图实际相加的FP32常量为BF16提升后的`1.00117e-7`，不是标准FP32的`1e-7`。该事实由原归档函数来源及融合指令确认，不是根据loss反推的拟合常量。

[原生Dump完整边界审计](evidence/frontier229-builtin-boundaries.json)、[最终执行IR](evidence/frontier229-builtin-ms_output_trace_code_graph_0.ir)、[实际算子执行序](evidence/frontier229-builtin-ms_execution_order_graph_0.csv)、[统计数据](evidence/frontier229-builtin-statistic.csv)。

统计＋SHA1能力也在当前MindSpore2.7.2原生重放上实际验证：同样两个算子的6条输入/输出记录，CSV仅1412字节，包含SHA1、Max/Min和L2Norm，没有NPY/NPZ；连同IR/PB/执行序总共251826字节。完整模型更新与实际MF仍精确一致。这是能力验证而非235训练结果，见[能力凭据](evidence/frontier235-stat-hash-capability.json)和[原始统计表](evidence/frontier235-stat-hash-probe.csv)。统计字段打印精度有限；SHA1需使用同一算法口径，当前MindSpore对超过20480元素的张量使用分块组合哈希，不能直接与Python整块SHA1比较，详见[235阶段](阶段记录-235全局范数与统计哈希.md)。确需坐标和误差分布时再选择完整张量。

**观测边界限制：** Sqrt和epsilon相加仍在融合内部；CPU对原生平方和开方不是原生DVM内部标量的直接观测。早期源码TensorDump额外cast出来的BF16 norm/denominator也不能当作真实融合FP32分母，因此不能用它们对范数算子做精度排名。本结论依据原生融合IR、完整边界一致和独立单因素干预；不把一个标量的观测解释扩大成所有归约实现都已一致。

## 独立200步消融及影响排序

| 指标 | 对照最大 / 平均 | epsilon 最大 / 平均 | 对照 / epsilon 超 0.005 步数 |
|---|---:|---:|---:|
| 总 loss（当前验收） | 0.322234153748 / 0.0100560808182 | 0.239593505859 / 0.0093046939373 | 63 / 66 |
| 主 LM | 0.26727771759 / 0.00832351446152 | 0.207327365875 / 0.00701682448387 | 46 / 44 |
| MTP × 0.3 | 0.0550689697266 / 0.00286719292402 | 0.0633435249329 / 0.00312373369932 | 28 / 33 |
| MTP 原始值 | 0.183563232422 / 0.00955730974674 | 0.21114508311 / 0.0104124456644 | 96 / 98 |
| Aux × 1e-4 | 0.000135737041757 / 5.99555535009e-06 | 0.000120555406995 / 4.86205227696e-06 | 0 / 0 |
| Aux 原始值 | 1.35737041757 / 0.0599555535009 | 1.20555406995 / 0.0486205227696 | 120 / 115 |

两组首步总loss误差相同。新增因素的最大值、平均值下降，但超差步数从63增至66，首次总loss超差由272提前到263，MTP分项也退步。不能概括成稳定全面改善，更不能以短程全模型精确宣称训练已对齐。

[30步模型/配置/恢复/调用审计](evidence/frontier229-epsilon-independent30-audit.json)、[200步逐步误差与完整保真审计](evidence/frontier229-epsilon-independent200-audit.json)、[91组索引](evidence/completed-cann92-inventory.csv)。

1. **继续定位优先**：原229首差已经消除。[双边35步观测](evidence/frontier235-modelhash-audit.json)保真通过：231–234每步48个FP32/BF16模型张量仍精确匹配；235首次出现29项FP32、27项BF16模型差异，同一步首次出现梯度范数差。下一步只取234/235的梯度哈希与统计值，判断反向计算或更新路径。模型哈希一致不推出优化器状态一致，范数一致也不推出完整梯度一致。
2. **中：旧图兼容因素**：epsilon类型有原生源码/指令、完整更新重放和独立短程证据。全窗口仍远不满足标准，超差步数增加，暂不作为HP默认产品修复提交。
3. **较低：全程未受益的因素**：此前CE尾部因素虽消除226首差，但总loss平均和超差数退步，保留因果与负面记录，不将其排在已确认必要产品错误之前。
4. **不单独追逐**：仅正负零位差、被BF16舍入消除的表示差；没有可证实的有效更新影响时，不因此新增PR。

## 哪个实现符合当前契约

旧归档MF图模式的实际类型提升应由原生IR界定；为逐位复现该归档，使用其BF16 epsilon提升是有依据的兼容路径。HP在FP32归一化中保持配置epsilon的FP32值也具有明确的精度意义，不能因为未复刻旧图的隐式舍入就判成通用错误。

[MF PyNative固定提交](https://gitcode.com/mindspore/mindformers/blob/5089da308ebcc146d0cbe32180c209c4f1d39d77/mindformers/pynative/optimizer/muon.py)的`_normalize_newton_schulz_input`显式使用FP32范数、`clamp(norm,min=eps)`后转BF16，源码校验与保存位置见[219阶段](阶段记录-219归一化首差与消融.md)。[Megatron固定集成源码](https://github.com/NVIDIA/Megatron-LM/blob/3914483fd325e053797aff3fa431d4a3141139aa/megatron/core/optimizer/emerging_optimizers.py)调用NVIDIA Emerging Optimizers；[其0.4.0官方实现](https://docs.nvidia.com/nemo/emerging-optimizers/0.4.0/_modules/emerging_optimizers/orthogonalized_optimizers/muon_utils.html)要求FP32输入，单卡默认`functional.normalize`，并支持FP64归一化。这些实现并不要求旧图的`norm+BF16(epsilon)`契约。不能为了追旧MF而默认改变HP的现代数学契约；必要产品修改仍应独立中文子issue和PR。

## 失败实验、证据恢复和存储

- 第一版显式FP32 norm探针未把归一化结果转回BF16，改变了后续NS精度，最终权重未通过原生保真。它不作为有效因果证据或独立训练。补上BF16转换的第二版完整状态通过，但仍需要原生Dump避免将额外cast标量误认为融合内部量。
- 失败探针曾因输出报告路径替换遗漏，覆盖原重放的JSON元数据；原数组、源码、终端日志没有覆盖。失败元数据单独保留，原元数据从原终端JSON及未变源码恢复，并显式标记重建。恢复后原重放边界审计再次通过。见[路径恢复凭据](evidence/frontier229-report-path-recovery.json)。
- 失败探针的7个大数组经路径、结束状态和归属检查，并保留每文件SHA、shape、dtype及失败报告后清理，实际释放604008448字节；四个小标量、源文件、日志和有效原生数组保留。[清理凭据](evidence/frontier229-invalid-probe-cleanup.json)。
- 125临时内存盘接近满额后，已检查125/129实际文件系统及容器挂载。将终止且未被读取的`mf-grad205-state`177文件归档到125根分区`/var/tmp/xxl-jt-alignment-20261013/mf-grad205-state.tar`，逐文件SHA核验完整32,616,437,760字节归档后，移除原位置NPY、保留JSON索引与`ARCHIVED.json`，释放tmpfs32,616,681,472字节。这是跨盘迁移，不计成全机空间净清理。[归档/恢复映射](evidence/frontier229-mf205-cold-archive.json)。
- 迁移后125的tmpfs剩余38,456,762,368字节、根分区87,085,334,528字节。129只读检查时`/home`约354GiB、tmpfs约951GiB可用，作为后续必要时的备用；未往129写实验，未触碰zth目录。

总issue仍开放：[217](https://github.com/hf1s/hyper-parallel/issues/217)。已有中文链路#215→PR213、#187→PR192、#216→PR180保持；本阶段全部归档兼容因素和负面实验都记录，不据此新建混合兼容因素的产品PR。
