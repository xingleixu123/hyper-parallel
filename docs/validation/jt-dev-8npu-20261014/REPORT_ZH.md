# JT八卡验证：问题排序、独立PR与证据总览

## 总体目标与当前结论

本单是 JT DeepSeek V3 八卡验证的**总问题单**，统一关联所有已拆分的小问题及独立 PR，维护优先级、复现证据、精度曲线、显存/性能、重计算与续训验收状态。每个已确认的独立缺陷或接口需求各用一个小问题单和一个单提交 PR，两个方向均能回到本单；不把多个缺陷混入同一个修复提交。

**当前总体尚未验收通过；最新固定配方的FSDP基线与续训已通过。** JT_dev `f637ef7`加已检视修复源`18602c4`，FSDP1–200和checkpoint200恢复后的201–400分别对独立连续400参考：8rank全部LM、加权及按系数还原的MTP/aux、总loss和梯度范数最大差均为0。当前配方EP2、CP2、TP2均已完整跑完200步，但三者精度均失败；完全和选择重算均完整200步×8rank与off基线逐步一致，其他合法组合仍待完成同配方200步验收；旧配方的跨拓扑超阈值和重算零误差证据单独保留，不能直接转作当前配方结论。

**当前执行顺序以用户最新目标为准：先解决主LM loss的完整200步对齐。主loss全部达到0.005以内后先停下来汇报，由用户决定是否继续MTP/aux专项；已有MTP/aux观测保留，但不把它们作为当前主loss排查的额外优化目标。**

当前阶段推荐优先检视#194的训练目标归一化修复和#198的源参数释放修复。全部14个已确认独立问题均有中文小问题单、独立单提交PR与原生父子关联；持续补充真实证据，问题保持开启，PR保持草稿。

## 按影响排序的问题、子问题与独立 PR

排序依据依次是训练目标正确性、运行阻断、资源浪费、重算与恢复语义、精度配置和诊断可靠性。表中的“测试通过”不代表问题已合入或完整 CI 已通过；所有小问题保持开启，PR 保持草稿。

| 排序 | 问题与影响 | 小问题单 | 独立单提交 PR | 已有证据及剩余边界 |
|---|---|---|---|---|
| Top1 | 当前配方EP2、CP2、TP2完整200步精度失败，组合矩阵仍待完成 | 本总单的整体验收缺口 | 根因确认后继续拆分 | 旧配方全rank误差超阈值，见历史数据；EP2为0.007523/0.015433/0.138989，CP2为0.011062/0.022786/0.204605（LM/还原MTP/aux）；TP2完整200步为0.021448/0.024670/0.155838；余下矩阵待完成 |
| Top2 | 梯度累积按微批有效 token 归一化，改变整步目标 | [#203](https://github.com/hf1s/hyper-parallel/issues/203) | [#194](https://github.com/hf1s/hyper-parallel/pull/194) | 实际 Trainer 的不等长监督样例，正确梯度 14、旧行为 10；TP2×EP2 LM 误差显著下降但仍未过 0.005 |
| Top3 | FSDP 分片后仍持有完整源参数，显存不能释放 | [#204](https://github.com/hf1s/hyper-parallel/issues/204) | [#198](https://github.com/hf1s/hyper-parallel/pull/198) | EP2/CP2 分别减少 9.422/10.125 GiB；同拓扑 200 步 loss/梯度零误差 |
| Top4 | 测试调度器覆盖上层NPU可见列表，使设备分配越界 | [#223](https://github.com/hf1s/hyper-parallel/issues/223) | [#224](https://github.com/hf1s/hyper-parallel/pull/224) | 单提交2文件；无设备回归10测试/18子测试通过、两轴检视通过；不等同实际多卡训练验证 |
| Top5 | CP 分块输出损失的支持与回归覆盖 | [#205](https://github.com/hf1s/hyper-parallel/issues/205) | [#199](https://github.com/hf1s/hyper-parallel/pull/199) | 功能已由上游 #186 合入；#199 收敛为独立回归，新基点 CPU20项/25子测试及129四卡短步ST通过；每组2步，不代替200步验收 |
| Top6 | 完整重计算漏掉独立 MTP 解码器 | [#206](https://github.com/hf1s/hyper-parallel/issues/206) | [#195](https://github.com/hf1s/hyper-parallel/pull/195) | 实际重放及覆盖用例通过；当前配方full200步×8rank全部loss/梯度范数零误差，两个主干层和一个MTP层均包裹；selective第三轮完整200步×8rank全部loss及梯度范数零误差，未拼接旧前缀 |
| Top7 | 重计算重放重复提交专家路由 token 计数 | [#207](https://github.com/hf1s/hyper-parallel/issues/207) | [#196](https://github.com/hf1s/hyper-parallel/pull/196) | 保留必要张量运算，只约束持久副作用；均匀重复计数未必改变 sign 型 bias 更新，不能虚构旧 loss 漂移 |
| Top8 | checkpoint 续训后累计样本/token 指标清零 | [#208](https://github.com/hf1s/hyper-parallel/issues/208) | [#197](https://github.com/hf1s/hyper-parallel/pull/197) | 最新固定配方201–400×8rank累计/逐步计数与独立400参考一致；各loss与梯度最大差0；集成结果不全部归因于本PR |
| Top9 | 专家局部 BF16 权重梯度舍入不能由后续 FP32 归约恢复 | [#209](https://github.com/hf1s/hyper-parallel/issues/209) | [#200](https://github.com/hf1s/hyper-parallel/pull/200) | 默认关闭的 FP32 主梯度累加选项；CPU 与真实 2 卡累积/共享权重/重算 ST 通过，200 步组合验证仍待完成 |
| Top10 | Muon NS 精度无法显式配置并从 checkpoint 恢复 | [#210](https://github.com/hf1s/hyper-parallel/issues/210) | [#201](https://github.com/hf1s/hyper-parallel/pull/201) | 保留 BF16 默认；CPU、8 卡和 2 卡 ST 通过；可选 FP32 成本须匹配配方评估 |
| Top11 | Trainer向NPU写入无效CUDA属性，不能关闭预先开启的HF32 | [#227](https://github.com/hf1s/hyper-parallel/issues/227) | [#228](https://github.com/hf1s/hyper-parallel/pull/228) | 单提交2文件；真实TorchNPU选项及3项CPU测试/2项子测试通过，两轴检视通过；默认HF32关闭，未证明是CP/EP漂移根因，硬件训练未验 |
| Top12 | 非整除普通 Shard 重建返回错误形状或通信不一致 | [#211](https://github.com/hf1s/hyper-parallel/issues/211) | [#214](https://github.com/hf1s/hyper-parallel/pull/214) | 优化器无关的 Gloo 复现；入口一致拒绝不支持布局，457 项 UT/41 项子测试、2/4 进程 ST 通过；Ascend两卡各2项、四卡各1项通过，原提交未改；不代替200步训练验收 |
| Top13 | CPU 后端返回字符串设备索引，JT 模型构建报 TypeError | [#218](https://github.com/hf1s/hyper-parallel/issues/218) | [#219](https://github.com/hf1s/hyper-parallel/pull/219) | 单提交3文件；真实 CPU 路径修复前5失败，修复后20测试/7子测试通过；不代表125 NPU发现问题已修复 |
| Top14 | 资源MAX仅覆盖DP+CP，漏报其他TP/PP rank的耗时或显存峰值 | [#225](https://github.com/hf1s/hyper-parallel/issues/225) | [#226](https://github.com/hf1s/hyper-parallel/pull/226) | 单提交2文件；受控回归修复前4项子测试失败、修复后5测试/7子测试通过；旧69步前缀确认耗时及reserved漏报；新完整200步有51步时间漏报；真实8进程CPU/Gloo资源偏差64→0，NPU/PP训练通信仍待回归 |
| Top15 | FSDP动态派生的重算wrapper被覆盖报告漏计 | [#229](https://github.com/hf1s/hyper-parallel/issues/229) | [#230](https://github.com/hf1s/hyper-parallel/pull/230) | 真实三处区域报告为一处；单提交2文件，CPU62测试/2子测试、两轴检视及lint通过；仅修报告，NPU训练与性能无新增结论 |

目前优先推荐审阅 #194 的训练目标修复和 #198 的源参数释放修复：前者有明确的数学反例，后者有显著且可解释的实测显存收益。#200/#201 是显式可选数值配方，尚无足够证据默认启用。#214 的边界是明确拒绝尚不支持的普通非整除分片，不是实现其收集算法。

已固定的独立提交：#194 `bad2550`、#195 `00ae8a7`、#196 `d1f5213`、#197 `bd18261`、#198 `b948684`、#199 `e9fc94d`（旧提交 `d86b74e` 保留备份）、#200 `630d543`、#201 `59a03c6`、#214 `453618a`、#219 `86930dd`、#224 `a37d6bc`、#226 `b58fed7`、#228 `453fe57`、#230 `b540051`。各 PR 的完整 SHA、文件范围和定向测试以对应 PR 及回读凭据为准；集成测试含多个修复，不能全部归功于单一 PR。

## 当前固定环境

当前 JT_dev 为 `f637ef7ee4d4b416800716f1ebaaae7709b8a06a`，包含用户指定的 #180 及上游 #186；集成训练源为 `18602c4524dc664b81b2bad9a7b624455ce1494f`。最近核对为2026-10-16 05:28（UTC+8），上游未变。训练源集成#194–#201相关修复及CPU测试隔离补充；#214/#219/#224/#226/#228/#230有独立验证，不冒称全部进入当前训练源。改动源码或数值配方时需重新固定匹配参考。

| 项目 | 固定值 |
|---|---|
| 当前完整训练 | 129，8×Ascend 910B3 64GiB；125实际空闲时可使用 |
| CANN | 9.2.0-beta.2 / V100R001C12B086 |
| Python / Torch / TorchNPU | 3.12.14 / 2.12.0+cpu / 2.12.0；PTA Git `fa0f83fe49d309dcbc31e264e9e6ed6e5dc49d2d` |
| Transformers / datasets | 5.13.1 / 4.8.5 |
| 镜像 | `jt-clean:hp-910b-20261009`，ID `sha256:5f9e677f439d11ab4e7c888f54b142335a27eee7411cbd2ce894a4798a70146d` |
| 用户镜像原文件 | `/home/jt_image/four-images-clean-20261009/HP-910B/hp-910b-arrow.tar.gz`，保留不删 |
| 代码导入 | `/workspace/hyper-parallel` 的挂载editable checkout，启动时核验，不使用镜像内旧代码 |
| 模型 / 数据 | 2层主干＋1层MTP，8专家/top2，hidden5120、vocab152064；Arrow4710行，seq4096、GBS8/MBS1 |
| 数值配方 | BF16参数、FP32归约和主参数，专家FP32权重梯度及Muon NS FP32；不是全模型FP32 |
| 固定训练项 | seed42、dataloader seed1234、constant LR0.0001、整步token目标、MTP0.3/aux0.0001、chunk loss4096 |

训练入口为 `examples/training_demo/jt_deepseek_v3`。下列原始证据目录含完整YAML、解析配置、启动代码、初态/输入哈希、逐rank标量和退出/释放凭据；原训练文本不上传。复现需要同时保留这些身份与数值记录。


## 统一验收标准

1. 固定源码、镜像、CANN、数据及优化器配方，先跑 **FSDP8 的 step1–200**，保存 step200 checkpoint；另起进程加载后跑 **step201–400**，再与独立连续 step1–400 参考比较。
2. 固定全局 batch8、微批1、序列4096、seed42、输入顺序、学习率与初始化。合法 TP、CP、EP 及混合策略各跑 200 步，与同精度配方的 FSDP8 比较。改变精度选项必须重新建立基线、续训与连续参考。
3. 检查全部 8 rank、全部 step 的最大绝对差，LM、MTP、aux 均须 ≤0.005；同时检查日志加权量和按实际系数还原的未加权量。日志 MTP 系数为 0.3，aux 为 0.0001，不能用系数掩盖原始分量误差。
4. 初始化/输入身份、步数、rank 覆盖、LR、有限值、退出码、完成标记、恢复计数另外验收。输入哈希相同不能替代 loss 验收；运行完成和自比较零误差也不能替代独立参考。
5. 关闭、完整、选择重算各跑 200 步；模块选择消融注明步数和测量边界。报告 allocated/reserved 峰值及预热后中位步时，数据采集开销保持一致。
6. 每个修复以独立最小回归、相应分布式用例和规范/验收条件两轴检视作为证据；保留首次失败与修复后结果。开放问题不能仅靠 PR 已创建或某个标签关闭。

## 当前FSDP基线与续训：两个独立200步窗口已通过

基线 `r129-f637-fp32-fsdp8-0-200-r4` 完成1–200，保留step200 checkpoint；新进程 `r129-f637-fp32-fsdp8-200-400-r2` 从该checkpoint恢复至400。比较参考是从初始化独立连续跑400步的 `r129-f637-fp32-fsdp8-continuous400-r3`。

| 全8rank、全200步最大绝对差 | LM | 日志及还原MTP | 日志及还原aux | total loss | 梯度范数 | 结论 |
|---|---:|---:|---:|---:|---:|---|
| 基线1–200 vs 连续参考1–200 | 0 | 0 | 0 | 0 | 0 | 通过 |
| 续训201–400 vs 连续参考201–400 | 0 | 0 | 0 | 0 | 0 | 通过 |

源码、镜像、数值配方、初始化、输入、LR、逐步和累计计数、退出码/完成标记均核验。还原MTP/aux由日志除以0.3/0.0001得到，没有额外采集加权前张量；梯度结论只涉及范数。

![当前FSDP基线对独立参考，全部rank最大误差](figures/f637-fsdp8-baseline-independent-zh.png)

![当前checkpoint200续训对独立参考，全部rank最大误差](figures/f637-fsdp8-resume-independent-zh.png)

[基线严格比较](reports/f637-fsdp8-continuous400-r3-complete/f637-fsdp8-baseline-independent-comparison.json) · [续训严格比较](reports/f637-fsdp8-continuous400-r3-complete/f637-fsdp8-resume-independent-comparison.json) · [独立连续参考全部证据](reports/f637-fsdp8-continuous400-r3-complete)。


## 当前TP2完整200步

TP2第四轮`r129-f637-fp32-tp2-200-r4`完整完成200步×8rank，退出0，517.637332秒后释放八卡，无本任务残留。与FSDP8基线的输入、初态、计数、源码、镜像及数值配方检查通过；独立数值与资源复算也确认原始证据有效，**训练精度FAIL**。LM/还原MTP/aux最大差分别为**0.0214481354/0.0246699651/0.1558382064**，首次超过0.005分别是111/45/19步。旧中断轮共同1–69步的全部loss和梯度范数与本轮差0，未拼接前缀。

![TP2完整200步全部rank精度误差](figures/f637-tp2-r4-200-raw-zh.png)

| 当前配方完整200步 | LM最大差 | 还原MTP最大差 | 还原aux最大差 | 验收 |
|---|---:|---:|---:|---|
| EP2 | 0.007523060 | 0.015432835 | 0.138989417 | 未通过 |
| CP2 | 0.011061668 | 0.022786458 | 0.204604585 | 未通过 |
| TP2 | 0.021448135 | 0.024669965 | 0.155838206 | 未通过 |

TP2 allocated/reserved峰值17.294672/18.638672GiB，allocated较FSDP8节省4.737524GiB（21.5027%）；逐步8rankMAX、去前20步后的中位步时1.857553秒（+60.8268%），监督token吞吐9957.676842/s（-37.7972%）。实际布局DP4×TP2，开启SP/loss_parallel、每步两个微批并保持路由/专家权重TP复制；不是仅改tp_size。显存下降方向符合分片预期，但本次没有吞吐提升；通信/矩阵尺寸的各自份额尚无profile证明。单次观测含输入采集开销，精度FAIL的配置不能推荐为已验收方案。

rank0有51/200步漏报最慢TP lane时间，最大1.213431毫秒；本轮allocated/reserved没有rank0峰值漏报，与旧69步记录分开。离线汇总已校正，进一步补充#225/PR#226的真实修复前证据；该PR原单提交不变，真实NPU修复后通信成本仍待测。

76份原始文件SHA全部核验，传输重复包两端清理，checkpoint200保留。当时设备实验按释放后至少20分钟及空卡复查策略执行；FSDP/CP2×原生/FP32输入梯度的四臂三步诊断保持LR、数据和0.005门槛，不替代200步验收。

[详细配置、复现和资源解释](reports/f637-tp2-r4-complete/README_ZH.md) · [严格全rank比较](reports/f637-tp2-r4-complete/f637-tp2-r4-strict-comparison.json) · [数值独立复算](reports/f637-tp2-r4-complete/f637-tp2-r4-precision-independent-review.json) · [资源与释放独立审计](reports/f637-tp2-r4-complete/f637-tp2-r4-resource-independent-review.json)。

## 当前CP2完整200步

`r129-f637-fp32-cp2-200`完成200步×8rank，退出0，462.81秒后释放全部8卡，无本任务设备PID残留。启动满足前轮释放后的全局20分钟间隔，源`18602c4`、镜像、CANN9.2.0-beta.2和归一化数值配方与FSDP基线一致。76份原始文件大小/SHA逐一核验，两端重复归档已删除；全部输入、48项记录初态、计数和LR检查通过，精度门槛失败。

| 完整200步、全8rank最大绝对差 | LM | 日志MTP（×0.3） | 按系数还原MTP | 日志aux（×0.0001） | 按系数还原aux |
|---|---:|---:|---:|---:|---:|
| CP2 对同配方FSDP8 | 0.0110616684 | 0.0068359375 | 0.0227864583 | 0.0000204605 | 0.2046045847 |

LM最大差出现在第148步，MTP第98步，aux第172步。总loss最大差0.0140056610，梯度范数最大差0.6418290138；未加权量是日志标量按真实配置系数还原，不是额外采集的加权前张量。这里确认的是跨拓扑精度失败，尚未确定新的独立根因；继续在Top1追踪，不能用资源统计修复解释这些数值差异。

![CP2完整200步精度未通过；MTP和aux按配置系数还原](figures/f637-cp2-200-raw-zh.png)

| 同主机同配方资源观测 | FSDP8 | CP2 |
|---|---:|---:|
| 每步全部rank MAX、去20步后中位秒/步 | 1.155003 | 1.596322 |
| 监督token/s，实际有效token只计一次 | 16008.4066 | 11586.1031 |
| 全程累计峰值allocated GiB | 22.032196 | 20.706646 |
| 全程累计峰值reserved GiB | 27.250000 | 25.982422 |

allocated减少1.325550 GiB，约6.02%；中位步时增加约38.21%。CP2在本配置中将逻辑序列分段由4096降为2048，dense FSDP仍为8路，优化器步由1个变为2个微步；参数与优化器持久状态没有CP倍数的分片收益。通信及微步变化是需要关注的成本，但本轮没有算子级归因。精度未通过，且只测了一次，这些数据只用于诊断，不能推荐为达到精度目标后的最优方案。

[严格精度比较](reports/f637-cp2-200-complete/f637-cp2-200-strict-comparison.json) · [完整分析](reports/f637-cp2-200-complete/f637-cp2-200-complete-analysis.json) · [76份采集清单](reports/f637-cp2-200-complete/f637-cp2-200-complete-evidence-manifest.json) · [原始8rank记录](reports/f637-cp2-200-complete/raw/runs/r129-f637-fp32-cp2-200) · [全rank资源观测](reports/f637-cp2-200-complete/f637-cp2-200-all-rank-resources.json) · [匹配参考资源](reports/f637-cp2-200-complete/f637-fsdp8-r4-all-rank-resources.json) · [SHA及清理凭据](reports/f637-cp2-200-complete/f637-cp2-200-complete-local-verification.json) · [完整结果独立复算检视](reports/f637-cp2-200-complete/f637-cp2-200-independent-evidence-review.json)。

CP2结束后已完成当前配方FSDP8完整重计算200步，详见本单最新结果；selective和TP2现已完成；TP2精度仍失败，接续优先做当前配方单变量短步定位，再补齐其余并行矩阵。该阶段采用释放后至少20分钟及空卡复查策略；后续按本单末尾的最新资源策略执行。

## 当前EP2完整200步

`r129-f637-fp32-ep2-200`在满足全局20分钟间隔后，完成200步×8rank，退出0，407.33秒后释放全部8卡。76份文件大小/SHA逐一核验，两端重复压缩包已删除。固定源仍为`18602c4`、CANN9.2.0-beta.2；本轮运行完整，源码/配方/镜像、输入、初态、计数与LR等完整性检查无不一致，但精度门槛失败。

| 全8rank、完整200步最大绝对差 | LM | 日志MTP（×0.3） | 还原MTP | 日志aux（×0.0001） | 还原aux |
|---|---:|---:|---:|---:|---:|
| EP2 对同配方FSDP8 | 0.0075230598 | 0.0046298504 | 0.0154328346 | 0.0000138989 | 0.1389894169 |

加权MTP单项小于0.005，不能掩盖还原分量超阈值；LM也超过0.005。原始误差与完整性检查均由未修改的严格比较器重新计算，根因仍在定位，不能把这些差异归因于新资源统计修复。

![EP2完整200步未通过；MTP/aux由日志按实际系数还原](figures/f637-ep2-200-raw-zh.png)

| 同主机、同配方资源观测 | FSDP8 | EP2 |
|---|---:|---:|
| 逐步全部rank MAX、去20步后中位秒/步 | 1.155003 | 1.317435 |
| 监督token/s，实际有效token只计一次 | 16008.4066 | 14034.3284 |
| 累计峰值allocated GiB | 22.032196 | 20.928654 |
| 累计峰值reserved GiB | 27.250000 | 26.275391 |

本次allocated减少1.103542 GiB，约5.01%；中位步时增加约14.06%。FSDP已分片整模型状态，且该小层数模型的embedding/输出头占保存模型载荷57.29%、路由专家仅占13.89%，不能期待EP2使整步显存减半。额外通信可能增加耗时是配置/源码预期，此轮未做算子归因；单次观测且精度失败，尚不能称为达到目标后的并行收益。

离线资源工具补齐了逐步全rank MAX、单次数据计数和累计峰值边界。独立两轴检视发现的输出覆盖、路径别名及非法计数问题已修正，9项CPU反例通过；输出使用排他创建，输入解析与SHA共用同一字节快照，严格loss比较器及训练源码未改。工具复核不替代多卡训练验证。

[完整运行分析](reports/f637-ep2-200-complete/f637-ep2-200-complete-analysis.json) · [严格精度比较](reports/f637-ep2-200-complete/f637-ep2-200-strict-comparison.json) · [76份采集清单](reports/f637-ep2-200-complete/f637-ep2-200-complete-evidence-manifest.json) · [8rank原始证据](reports/f637-ep2-200-complete/raw/runs/r129-f637-fp32-ep2-200) · [EP2逐步全rank资源](reports/f637-ep2-200-complete/f637-ep2-200-all-rank-resources.json) · [FSDP同窗口资源](reports/f637-ep2-200-complete/f637-fsdp8-r4-all-rank-resources.json) · [离线工具两轴复核](reports/f637-ep2-200-complete/all-rank-resources-two-axis-review.json) · [完整结果独立复算检视](reports/f637-ep2-200-complete/f637-ep2-200-independent-evidence-review.json)。

该阶段12组独立问题/PR的中文正文、单提交、双向链接及原生父子关系已全部回读核验（当前已增至14组，见首节），[25项核验记录](reports/f637-ep2-200-complete/github-chinese-full-link-audit.json)。本轮另清理54,872,986字节已用完的本机/129临时输入、CPU源码副本及专用venv，[清理凭据](reports/f637-ep2-200-complete/global-resource-metrics-cleanup.json)；最终源码、原始证据、当前step200 checkpoint和用户镜像保留。后续CP2已按间隔完成，结果见前文；此处保留EP2完成时的阶段记录。

## 当前重计算200步对齐

选择重算第三轮`r129-f637-fp32-fsdp8-selective-200-r3`完整完成200步×8rank，退出0，运行394.861022秒并释放八卡，无本任务PID残留。与关闭重算FSDP8基线相比，LM、日志加权及按系数还原的MTP/aux、总loss和梯度范数最大绝对差全部为**0**，输入和初态检查也通过。由此当前固定配方的off/full/selective三种重算均完成200步精度对齐；这不替代EP/CP/TP矩阵验收。

![选择重算完整200步精度对齐，全部8rank误差为0](figures/f637-fsdp8-selective-r3-200-raw-zh.png)

| 同129、同配方、同输入采集开销 | 关闭重算 | 完全重算 | 选择重算 |
|---|---:|---:|---:|
| 去前20步后，逐步8rank MAX中位秒/步 | 1.155003 | 1.235180 | 1.232111 |
| 全200步累计allocated峰值GiB | 22.032196 | 19.392933 | 19.875641 |
| 全200步累计reserved峰值GiB | 27.250000 | 24.320313 | 24.789063 |
| 有效监督token/s，逻辑数据只计一次 | 16008.407 | 14969.751 | 15011.921 |

选择重算相对关闭节省2.156555 GiB allocated（9.7882%），中位步时增加6.6760%，监督吞吐下降6.2248%。完全重算再多省0.482708 GiB，而本次步时仅比选择重算多0.2491%；单次观测不能证明稳定速度差异。在当前两主干层加一MTP层配置中，选择重算尚未显示明显更好的性价比。模块消融继续按实际配方/步数注明，旧结果不直接转作当前结论。

76份原始文件SHA核验完成，重复归档已清理；精度与资源由两轴独立复算。MTP/aux为日志按0.3/0.0001系数还原，不是额外采集的加权前张量；梯度结论限于范数。此后TP2-r4已完成200步，见本单新增完整结果；这些已完成轮次遵守当时的20分钟间隔并检查空闲。

[完整中文说明](reports/f637-fsdp8-selective-r3-complete/README_ZH.md) · [严格比较](reports/f637-fsdp8-selective-r3-complete/f637-fsdp8-selective-r3-strict-comparison.json) · [76份原始文件清单](reports/f637-fsdp8-selective-r3-complete/f637-fsdp8-selective-r3-complete-evidence-manifest.json) · [资源逐步结果](reports/f637-fsdp8-selective-r3-complete/f637-fsdp8-selective-r3-all-rank-resources.json) · [数值独立审计](reports/f637-fsdp8-selective-r3-complete/f637-fsdp8-selective-r3-precision-independent-review.json) · [资源独立审计](reports/f637-fsdp8-selective-r3-complete/f637-fsdp8-selective-r3-resource-independent-review.json)。

两次旧重试分别完成109/48步后因外部实验使用卡4/卡5而仅停止本任务，原始严格结果仍为未完成；第三轮从step0独立跑满200步，**未拼接前缀**。旧记录与匿名归属审计保留：[第一轮109步](reports/f637-fsdp8-selective-prefix109/README_ZH.md)、[第二轮48步](reports/f637-fsdp8-selective-r2-prefix48/README_ZH.md)。启动前约10分钟各次样本均空闲，手动接续最大间隔约94.565秒，不冒称自动观察器的90秒约束已通过；最终仍由原守卫两次当前空卡检查。当前step200 checkpoint保留。

![当前完全重算200步全部rank误差](figures/f637-fsdp8-full-200-raw-zh.png)

## 当前资源总览与重计算接口

六组完整200步的资源结果已从48份原始rank日志、9600行逐项复算，60项输入SHA及图表数字经独立审计通过。绿色重算配置的精度已通过；红色斜线的EP/CP/TP仍未通过0.005，图中资源下降不能当成已验收的优化推荐。

![当前配方的显存与耗时总览，分别标出精度结论](figures/f637-current-resource-overview-zh.png)

当前FSDP完全/选择重算分别节省约11.98%/9.79% allocated显存，步时增加6.94%/6.68%；两者仅3.069ms的单次中位时间差不能排稳定速度名次。图中每组均为独立完整200步；耗时先逐步取8rank MAX再去前20步，显存含warmup且截至训练采样，不含末次checkpoint保存。保留相同输入采集开销，非无插桩或端到端性能。

JT底层已有checkpoint、exclude及策略接口；当前YAML的off/full/selective可以直接使用。按层安全子区域尚缺JT adapter声明，任意模块路径、排除和通信组合也未完整映射。full覆盖两个主干decoder和一个MTP decoder，off仍保留独立chunked head/CE重算。MF PyNative与Megatron按最新固定官方提交逐项核对；模块成本复测已定义精确FQN与pre-FSDP安装点，尚未执行的新策略不冒称通过。

[中文详细说明、数据、SVG及独立审计](reports/f637-resource-overview/README_ZH.md) · [MF/Megatron能力与接口缺口](reports/f637-resource-overview/f637-recompute-flexibility-gap-review-v2.md) · [当前版本模块复测计划](reports/f637-resource-overview/f637-module-cost-current-plan.md)。

| 当前完整200步策略 | allocated峰值GiB | reserved峰值GiB | 中位秒/步 | 监督token/s | allocated节省 | 步时变化 |
|---|---:|---:|---:|---:|---:|---:|
| FSDP8 off | 22.032196 | 27.250000 | 1.155003 | 16008.407 | 基准 | 基准 |
| FSDP8 full | 19.392933 | 24.320313 | 1.235180 | 14969.751 | 11.98% | +6.94% |
| FSDP8 selective | 19.875641 | 24.789063 | 1.232111 | 15011.921 | 9.79% | +6.68% |
| EP2，精度未通过 | 20.928654 | 26.275391 | 1.317435 | 14034.328 | 5.01% | +14.06% |
| CP2，精度未通过 | 20.706646 | 25.982422 | 1.596322 | 11586.103 | 6.02% | +38.21% |
| TP2，精度未通过 | 17.294672 | 18.638672 | 1.857553 | 9957.677 | 21.50% | +60.83% |

本轮尚未观察到TP/CP/EP吞吐收益，精度未通过的配置只作诊断。模块策略先以同配方40步off分别对照六个decoder norm、一个dense MLP和两个完整MoE，精确验证目标与重放，再筛选候选进入200步。进入/退出allocated差不是保存激活量；嵌套模块时间和单模块节省量不可相加。具体FQN、pre-FSDP接入和旧工具限制见前述模块计划。


## 补充验证：资源统计 PR #226 的真实八进程 CPU 通信

在固定基点`f637ef7`与单提交修复`b58fed7`上，使用真实八进程Gloo，分别检查DP2×CP2×TP2、DP2×CP2×PP2对应的进程组边界及DP8 world控制组，每组两步。最终v2两侧各48条观测、合计1152个指标值：基点64项资源偏差（两种分组各32）逐项符合旧代码的组内MAX，修复版全部为0；数据、loss、累计计数及world控制均无非预期差异。

通信函数和进程组未mock，时钟及显存读数使用人工哨兵。这证明CPU/Gloo的归约域修复，不是TP/PP模型训练、NPU/HCCL性能或200步验收。两轴独立检视均通过，原始逐rank数据、两个版本的精确Git归档指纹、退出码与无设备容器移除凭据一起公开；PR #226保持原来的一个提交。

[中文详细记录和复现入口](reports/pr226-gloo-8cpu/README_ZH.md) · [基点结果](reports/pr226-gloo-8cpu/raw/evidence/pr226-gloo-8cpu-v2/base-summary.json) · [修复结果](reports/pr226-gloo-8cpu/raw/evidence/pr226-gloo-8cpu-v2/head-summary.json) · [全部原始文件SHA](reports/pr226-gloo-8cpu/pr226-gloo-evidence-manifest.json)。

## 其他独立修复及回归边界

#228真实TorchNPU选项及CPU定向测试通过，修复预先开启HF32后Trainer不能关闭的问题；镜像默认HF32关闭，尚未证明是本轮漂移根因。[复现、修复前后与两轴检视](reports/npu-precision-switch/README_ZH.md)。#230修复FSDP动态派生wrapper被报告漏计；62项CPU测试/2项子测试及两轴检视通过，只修报告，不能称改变了训练性能。[原始证据](reports/checkpoint-wrapper-inventory/README_ZH.md)。

#214保持一个提交，在Ascend完成两卡FP32/BF16及四卡Partial回归，普通非整除Shard的行为为明确拒绝不支持布局。#199功能由上游#186提供，该PR保留独立回归并完成四卡短步ST。各PR的测试范围与剩余限制见顶部排序表及对应中文小问题；短回归和CPU通过不代替200步训练或完整CI。


## 最新精度定位：Linear输入梯度FP32未整体改善CP2

2026-10-16（UTC+8）完成四组独立3步：A=FSDP8原生、B=CP2原生、C=FSDP8 FP32 dgrad、D=CP2 FP32 dgrad。全部8rank完成、退出0并释放，各轮间隔至少1200秒；固定源码/镜像/初态/输入及实际反向分支均有原始记录和独立审计。

| 同配方跨布局的全8rank三步最大绝对差 | LM | 还原MTP | 还原aux |
|---|---:|---:|---:|
| A−B：原生Linear dgrad | 0.000398635864258 | 0.000425179799398 | 0.000477448338643 |
| C−D：FP32 dgrad后转BF16 | 0.000271797180176 | 0.000952879587809 | 0.000898871803656 |

LM最大差缩小，但MTP、aux分别变为原来的2.241/1.883倍。首步1,747,200个前向采样坐标均相同；1,138,944个梯度采样坐标中的不同数量从193,767增至200,825，最大差3.4332275390625e-5→3.814697265625e-5。C实际576次、D实际1152次FP32输入梯度计算（CP2每步两个microbatch）。

![四臂诊断，三步并非200步验收](figures/f637-dgrad-four-arm-zh.png)

**结论：不据此提交Linear FP32输入梯度的生产修复PR。** A/B采集相对原生200步参考的前三步loss/梯度范数差0；C/D是不同数值配方。各布局内的A−C/B−D效应单独保存，不能混作跨布局收益。首步只观察stride128的有限模块边界，不能认定完整张量相同或唯一致因算子；200步门槛未放宽。

[详细设计、四种比较、限制与复算方式](reports/f637-dgrad-four-arm/README_ZH.md) · [原始四臂无损包及SHA](reports/f637-dgrad-four-arm/four-arm-inputs-manifest.json) · [独立逐值/执行审计](reports/f637-dgrad-four-arm/analysis/f637-dgrad-four-arm-independent-review.json)。原字节206.59MB压缩为32.43MB，本机恢复554文件并重算后逐SHA核验，临时恢复副本已清理。

模块成本四组已通过固定镜像parser4/4与CPU7项合约。新增off控制组完整40步×8rank对原基线前40步的各loss/梯度范数误差均为0，9个模块边界的前三步计数和hook移除已验证；norm/dense/MoE成本消融尚未执行，暂优先排查主loss。单卡PTA探针v1启动因诊断器自身venv路径问题失败，6.296秒后释放，没有算子结论；v2正在前置验证，下一轮仍遵守20分钟间隔。[原始130文件、40步审计及探针失败说明](reports/f637-module-off-and-probe-preflight/README_ZH.md)。


## 待完成项、资源与数据管理

- [x] 14个独立问题各有中文小issue及单commit草稿PR，原生父子关系和正文双向链接回读通过。
- [x] 当前FSDP0–200、checkpoint200→400与独立连续400的两个200步窗口通过。
- [x] 当前off/full/selective均完成200步精度与全rank资源复算。
- [x] MF PyNative与Megatron最新固定源码能力对照完成，接口及限制可追溯。
- [ ] TP/CP/EP及合法混合策略全部200步满足0.005，按确认的独立根因继续拆issue/PR。
- [ ] 完成当前配方模块成本消融，筛选有价值的策略后做200步回归。
- [ ] 数值配方若改变，重建对应FSDP基线、续训与独立连续400参考。
- [ ] 当前提交的完整CI、维护者评审及最终回归；PR创建不等于验收或合入。

按用户随后给出的**最新指示恢复每轮实验至少20分钟间隔**：只使用125/129空卡，结束并确认释放后至少等待1200秒，再次检查空卡后才启动下一轮。单轮最长900秒，当前定位优先使用单卡600秒预算；不停止其他人的实验，不长期占用八卡。启动前双重空卡检查、释放确认及本任务所有权保护均保留。此前取消间隔的记录仅为历史，已被本指示替代。[最新策略](reports/f637-module-off-and-probe-preflight/resource-policy-20261016-restored-20min.json)与[部署凭据](reports/f637-module-off-and-probe-preflight/fixed-pta-probe-guard-deployment-v1.json)。CPU与文件检查单独记录，不混入NPU性能数据。

当前129的step200 checkpoint约27.97GB保留作续训及状态定位。每轮复制证据后核验大小和SHA，及时删除两端重复归档及已用完的临时源码/venv。125五份旧逐步参数探针已无损gzip并读回校验，删除未压缩副本释放约6.14GB，原始证据仍保留。用户镜像、共享数据和其他任务文件不删。[已有清理凭据与CPU分母探针边界](reports/cp-denominator-cpu-and-cleanup/README_ZH.md)。

## 历史证据与版本边界

旧`ad1c553/ec2e16b/521d3a8`的长跑、模块消融、初期诊断以及历次让卡均保存在[此前完整报告的固定快照](https://github.com/xingleixu123/hyper-parallel/blob/cc6c1b3a76c4d6aa5559ad2b12d6b4b47087cdd6/docs/validation/jt-dev-8npu-20261014/REPORT_ZH.md)和[阶段历史索引](reports/phase-history-through-20261015.md)。旧配方MLP+norm200步零误差不转作当前结论；失败或中断记录不删除、不拼接。

#198在旧同拓扑200步对照中EP2/CP2分别节省9.422/10.125GiB且loss/梯度范数零误差，这是该修复的定向历史证据，与本轮资源表分别记录。#194目标反例、所有PR的完整SHA、最小复现、修复前后测试与两轴检视均保留在对应小issue/PR。与#145旧源码单卡100步及#190的MF单卡对齐范围不同，本单固定FSDP8、200步和独立恢复参考。
