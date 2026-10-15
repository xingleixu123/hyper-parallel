## 最新八卡基线执行与让卡记录（2026-10-15）

在全局间隔超过20分钟、129八卡连续两次确认空闲后，使用集成源 `18602c4524dc664b81b2bad9a7b624455ce1494f` 启动 `r129-f637-fp32-fsdp8-0-200-r2`。8个rank均记录了step1–31，loss有限、输入身份一致；step31累计248个样本、614332个监督token。随后0–3卡出现外部作业，事件时的UID、PID命名空间及控制组均与本任务容器不同，刷新进程清单后身份差异仍在。本任务只释放自己的容器；退出137、无残留本任务设备PID，整轮172.14秒。**未完成200步，也没有checkpoint200，不能开始本轮200–400续训或声明精度达标；冲突轮次不用于性能收益比较。**

[31步全rank核验与归属摘要](reports/f637-fsdp8-r2-interrupted/f637-fsdp8-r2-interrupted-analysis.json)、[设备释放凭据](reports/f637-fsdp8-r2-interrupted/r129-f637-fp32-fsdp8-0-200-r2.receipt.json)、[逐文件校验记录](reports/f637-fsdp8-r2-interrupted/f637-fsdp8-r2-interrupted-local-verification.json)。采集的68份文件全部校验，重复压缩包两端已删除；原始进程信息本地保留，公开版本仅披露必要身份差异。此轮不能反推此前#199中断的进程归属。

总单、9个小问题及9个独立单提交PR共19项已回读核对，[双向关联与提交数](reports/f637-fsdp8-r2-interrupted/github-chinese-full-link-audit.json)完整。守卫诊断另修正了非UTF8进程信息解码与证据大小边界，10项CPU检查和规范/验收条件两轴复审通过；实际未知PID判断、只停止本任务容器及全局20分钟间隔均未改变。125的基线、续训和连续参考配置均已准备，下一轮重新检查空卡。

## 上游更新与资源进展（2026-10-15）

上游随后合入 #186，JT_dev 推进到 `f637ef7ee4d4b416800716f1ebaaae7709b8a06a`，包含 CP 分块损失支持、MLA 布局调整和 `enable_chunk_loss`/`chunk_loss_size` 配置。此前 `521d3a8` 的基线与续训证据属于固定旧轮次，不能改写为新基点结论。该轮连续400参考在 step168 出现外部设备进程，本任务已释放容器，保留中断日志，不计为完整参考，也没有停止他人任务。新基点已集成完成；接下来按0–200→checkpoint200–400→独立连续400的顺序重建匹配参考。

原 #199 与 #186 的生产功能重叠，现已保留上游实现并收敛为 #205 的独立回归测试：两个文件、单提交 `e9fc94d376ca7380c13a5e09c0020b34a75dc1d2`。新基点 CPU 20项测试、25项子测试通过，两轴检视通过；新基点四卡 ST 已尝试但中断，尚待完整回归，不沿用旧 NPU 通过结论。所有资源操作继续遵守空闲检查和全局20分钟间隔。

上一轮公开证据：[详细中文总报告](https://github.com/xingleixu123/hyper-parallel/blob/3781c38e91aa73baf84942ff3d4cd459ead758a2/docs/validation/jt-dev-8npu-20261014/REPORT_ZH.md)、[本次 CPU 原始日志](https://github.com/xingleixu123/hyper-parallel/blob/3781c38e91aa73baf84942ff3d4cd459ead758a2/docs/validation/jt-dev-8npu-20261014/reports/pr199-f637/pr199-f637-cpu-ut.log)、[隔离容器凭据](https://github.com/xingleixu123/hyper-parallel/blob/3781c38e91aa73baf84942ff3d4cd459ead758a2/docs/validation/jt-dev-8npu-20261014/reports/pr199-f637/pr199-f637-cpu-receipt.json)、[源码与单提交重写记录](https://github.com/xingleixu123/hyper-parallel/blob/3781c38e91aa73baf84942ff3d4cd459ead758a2/docs/validation/jt-dev-8npu-20261014/reports/pr199-f637/pr199-f637-rewrite.json)。以上链接固定到证据提交，不随分支后续更新漂移。

## 最新基点组合回归与 CPU 测试隔离补充（2026-10-15）

最新上游 `f637ef7` 的集成源 `18602c4524dc664b81b2bad9a7b624455ce1494f` 已通过 **82项测试、120项子测试，36.98秒**。最初组合运行出现8项失败：`torch_npu` 导入会把 checkpoint 默认设备改成 NPU，而 CPU 重计算回放会查询该设备的 autocast 能力。一个 CPU 张量的最小复现证实，PyTorch 原生与 Hyper 两条路径都受影响；显式设为 CPU 后两者梯度均精确通过。

修正仅落在相应 PR 的 CPU 测试 fixture：先登记 `addCleanup` 恢复旧设备，再设为 CPU。全部 loss、梯度、重放和路由计数断言保持原样，生产设备推断未改。#195 独立回归21项测试/16项子测试通过（31.02秒），#196 为2项/4项（31.96秒），#200 为13项/14项（0.78秒）。三个独立 PR 各自仍为一个提交；规范与验收条件两轴增量复审通过，修改文件 pylint 修正前后均零告警（本机缺 Torch，明确排除 import-error/no-member）。#200 第一次独立运行因测试路径不存在而未收集测试，修正入口后的结果单独记录。

本轮 CPU 容器未挂载 NPU，完成后已移除；这些结果不代替 NPU ST。旧源码与数值配置已保存在独立备份分支，新基点源码和 0–200、200–400、连续400 三份配置已部署并校验。第一次八卡0–200启动前发现外部任务占卡，因此没有启动。其后八卡空闲并通过两次复查，于2026-10-15 16:30:07+08:00启动新基点FSDP0–200；约30.32秒后3号卡出现外部进程，本任务容器已退出137、全部设备释放，未产生有效训练步。此轮记为资源冲突中断，下一轮最早16:50后重新检查。仍按空闲资源、全局20分钟间隔推进，不停止他人实验。

[回归汇总与限定条件](reports/cpu-fixtures/cpu-fixture-validation.json)、[组合回归原始日志](reports/cpu-fixtures/latest-f637-cpu-fixtures-ut.log)、[修正前失败日志](reports/cpu-fixtures/latest-f637-cpu-ut.log)、[最小复现记录](reports/cpu-fixtures/checkpoint-device-minimal-npu-import.container.log)。

## 两台服务器的匹配环境与中文图（2026-10-15）

125 的独立 Docker 环境中已核对到与 129 相同的镜像 ID `5f9e677f439d11ab4e7c888f54b142335a27eee7411cbd2ce894a4798a70146d`。两台训练源码均为 `18602c4524dc664b81b2bad9a7b624455ce1494f`，树 `825a2bbc68e5d72f2ebaa4af85ae0066d61cce97`；125 以新建隔离 venv 完成 editable 安装、真实配置解析与 4710 行数据读取，未挂载 NPU。两台 Arrow 文件 SHA256 同为 `cb7e146388ff5de09cafe50bb57c060265038ecabd2f343fa9114c05f3b678f1`。旧源码保存在各自备份分支，未删除共享镜像或数据。

总单与 PR 使用的图新增中文版本，并保留英文原件与原始数值。每张图标明历史来源、完整/未加权分量及实际比较对象；中文化不改变验收结论。14张图的生成输入、输出哈希和脚本见 [绘图记录](reports/chinese-figure-rendering.json)。MF 官方入口与镜像已核对到 `5089da308ebcc146d0cbe32180c209c4f1d39d77`，Megatron 到 `0101df91e5f86acb26a03bd8ad2361f3113476a6`；重计算相关实现变化已逐文件核对，详见 [能力对照](reports/recompute-reference-comparison.md)。

## #199 四卡短回归的中断证据（2026-10-15）

独立源码 `e9fc94d376ca7380c13a5e09c0020b34a75dc1d2` 在129的空闲0–3卡运行。约67.78秒后，资源守卫报告四个 PID 不属于当时容器进程清单，随后只释放本任务容器，退出137。事后相关 PID 已退出，现有记录不足以确证其外部归属或排除检测时序问题；下一轮需补齐事件时的进程归属证据。没有停止其他任务。

rank0 已输出8种配置各2步的16条损失记录，其中包含新增的 `chunk_size=48`、TP1×CP4 和 TP2×CP2×SP。观测到的 LM/MTP/aux 日志最大绝对差分别为0.000682831、0.000119448、9.10368e-8，最大梯度相对L2为0.0140618。**这些仅是中断前的局部观察：没有全部4rank的pytest完成摘要、退出0或完整结束标记，因此不能记作ST通过，更不能代替200步验收。** 参考 [局部分析](reports/latest-f637-preparation/pr199-f637-st4-partial-analysis.json)、[资源释放记录](reports/latest-f637-preparation/r129-pr199-f637-st4.receipt.json) 和 [逐文件哈希](reports/latest-f637-preparation/pr199-f637-st4-interrupted-evidence-manifest.json)。重复采集压缩包已校验后删除。

## 总体目标与当前结论

本单是 JT DeepSeek V3 八卡验证的**总问题单**，统一关联所有已拆分的小问题及独立 PR，维护优先级、复现证据、精度曲线、显存/性能、重计算与续训验收状态。每个已确认的独立缺陷或接口需求各用一个小问题单和一个单提交 PR，两个方向均能回到本单；不把多个缺陷混入同一个修复提交。

**当前总体尚未验收通过。** 最高优先级仍是 TP/CP/EP 相对 FSDP 的 200 步误差超过 0.005。六项修复集成的旧配方已做到 FSDP 续训及 full/selective 重算全 rank 零误差；新生产 FP32 专家梯度＋FP32 NS 配方必须使用自己匹配的参考，不能继承旧配方结论。

在固定旧基点 `521d3a8` 轮次中，129 已完成 FP32 配方 FSDP 0–200，再从该 checkpoint 续训 201–400；每段 8 rank 的全部 200 步、退出/完成标记、有限值记录完整，step201 累计样本恢复为 1608，累计监督 token 为 3720095，200→201 边界检查通过。**该固定轮次的独立连续400参考在step168因资源冲突被释放，未完成。当前只确认基线/续训运行完整性与计数连续性，不声明新配方续训loss对齐通过。** 续训收尾检测到外部 PID，该轮时间暂不用于性能对比。

## 按影响排序的问题、子问题与独立 PR

排序依据依次是训练目标正确性、运行阻断、资源浪费、重算与恢复语义、精度配置和诊断可靠性。表中的“测试通过”不代表问题已合入或完整 CI 已通过；所有小问题保持开启，PR 保持草稿。

| 排序 | 问题与影响 | 小问题单 | 独立单提交 PR | 已有证据及剩余边界 |
|---|---|---|---|---|
| Top1 | 完整 TP/CP/EP 矩阵 200 步精度仍不达标 | 本总单的整体验收缺口 | 根因确认后继续拆分 | 下表保留全 rank 最大误差；短跑、改 loss 权重或改变参考配方均不能替代验收 |
| Top2 | 梯度累积按微批有效 token 归一化，改变整步目标 | [#203](https://github.com/hf1s/hyper-parallel/issues/203) | [#194](https://github.com/hf1s/hyper-parallel/pull/194) | 实际 Trainer 的不等长监督样例，正确梯度 14、旧行为 10；TP2×EP2 LM 误差显著下降但仍未过 0.005 |
| Top3 | FSDP 分片后仍持有完整源参数，显存不能释放 | [#204](https://github.com/hf1s/hyper-parallel/issues/204) | [#198](https://github.com/hf1s/hyper-parallel/pull/198) | EP2/CP2 分别减少 9.422/10.125 GiB；同拓扑 200 步 loss/梯度零误差 |
| Top4 | CP 分块输出损失的支持与回归覆盖 | [#205](https://github.com/hf1s/hyper-parallel/issues/205) | [#199](https://github.com/hf1s/hyper-parallel/pull/199) | 功能已由上游 #186 合入；#199 收敛为独立回归，新基点 CPU20项/25子测试通过，NPU尚待完整回归 |
| Top5 | 完整重计算漏掉独立 MTP 解码器 | [#206](https://github.com/hf1s/hyper-parallel/issues/206) | [#195](https://github.com/hf1s/hyper-parallel/pull/195) | 实际重放及覆盖用例通过；集成 full/selective 各 200 步零误差，成本见下文 |
| Top6 | 重计算重放重复提交专家路由 token 计数 | [#207](https://github.com/hf1s/hyper-parallel/issues/207) | [#196](https://github.com/hf1s/hyper-parallel/pull/196) | 保留必要张量运算，只约束持久副作用；均匀重复计数未必改变 sign 型 bias 更新，不能虚构旧 loss 漂移 |
| Top7 | checkpoint 续训后累计样本/token 指标清零 | [#208](https://github.com/hf1s/hyper-parallel/issues/208) | [#197](https://github.com/hf1s/hyper-parallel/pull/197) | 旧配方恢复计数和连续参考完全相同；129 新配方恢复边界正确，完整数值对照待连续参考 |
| Top8 | 专家局部 BF16 权重梯度舍入不能由后续 FP32 归约恢复 | [#209](https://github.com/hf1s/hyper-parallel/issues/209) | [#200](https://github.com/hf1s/hyper-parallel/pull/200) | 默认关闭的 FP32 主梯度累加选项；CPU 与真实 2 卡累积/共享权重/重算 ST 通过，200 步组合验证仍待完成 |
| Top9 | Muon NS 精度无法显式配置并从 checkpoint 恢复 | [#210](https://github.com/hf1s/hyper-parallel/issues/210) | [#201](https://github.com/hf1s/hyper-parallel/pull/201) | 保留 BF16 默认；CPU、8 卡和 2 卡 ST 通过；可选 FP32 成本须匹配配方评估 |
| Top10 | 非整除普通 Shard 重建返回错误形状或通信不一致 | [#211](https://github.com/hf1s/hyper-parallel/issues/211) | [#214](https://github.com/hf1s/hyper-parallel/pull/214) | 优化器无关的 Gloo 复现；入口一致拒绝不支持布局，457 项 UT/41 项子测试、2/4 进程 ST 通过；Ascend 补丁回归待运行 |

目前优先推荐审阅 #194 的训练目标修复和 #198 的源参数释放修复：前者有明确的数学反例，后者有显著且可解释的实测显存收益。#200/#201 是显式可选数值配方，尚无足够证据默认启用。#214 的边界是明确拒绝尚不支持的普通非整除分片，不是实现其收集算法。

已固定的独立提交：#194 `bad2550`、#195 `00ae8a7`、#196 `d1f5213`、#197 `bd18261`、#198 `b948684`、#199 `e9fc94d`（旧提交 `d86b74e` 保留备份）、#200 `630d543`、#201 `59a03c6`、#214 `453618a`。各 PR 的完整 SHA、文件范围和定向测试以对应 PR 及回读凭据为准；集成测试含多个修复，不能全部归功于单一 PR。

## 统一验收标准与执行顺序

1. 固定源码、镜像、CANN、数据及优化器配方，先跑 **FSDP8 的 step1–200**，保存 step200 checkpoint；另起进程加载后跑 **step201–400**，再与独立连续 step1–400 参考比较。
2. 固定全局 batch8、微批1、序列4096、seed42、输入顺序、学习率与初始化。合法 TP、CP、EP 及混合策略各跑 200 步，与同精度配方的 FSDP8 比较。改变精度选项必须重新建立基线、续训与连续参考。
3. 检查全部 8 rank、全部 step 的最大绝对差，LM、MTP、aux 均须 ≤0.005；同时保存并检查日志加权量和未加权量。日志 MTP 系数为 0.3，aux 为 0.0001，不能用系数掩盖原始分量误差。
4. 初始化/输入身份、步数、rank 覆盖、LR、有限值、退出码、完成标记、恢复计数另外验收。输入哈希相同不能替代 loss 验收；运行完成和自比较零误差也不能替代独立参考。
5. 关闭、完整、选择重算各跑 200 步；模块选择消融注明步数和测量边界。报告 allocated/reserved 峰值及预热后中位步时，数据采集开销保持一致。
6. 每个修复以独立最小回归、相应分布式用例和规范/验收条件两轴检视作为证据；保留首次失败与修复后结果。开放问题不能仅靠 PR 已创建或某个标签关闭。

## 当前长跑精度数据

以下是 125 六项修复集成源 `ec2e16b` 的旧配方，所有误差取全 rank、全 step 最大值：

| 策略，均 200 步 | LM 最大差 | 日志 MTP 最大差（×0.3） | 日志 aux 最大差（×0.0001） | 结论 |
|---|---:|---:|---:|---|
| FSDP 0–200 对连续参考 | 0 | 0 | 0 | 通过 |
| FSDP checkpoint200→400 对连续参考 | 0 | 0 | 0 | 通过，含原始分量和累计计数 |
| 完整/选择重算对关闭重算 | 0 | 0 | 0 | 通过 |
| TP2，复制专家 | 0.014084816 | 0.006210327 | 0.0000172808 | 未通过 |
| EP2 | 0.014165878 | 0.003723621 | 0.0000124761 | 未通过 |
| EP4 | 0.015380383 | 0.005264282 | 0.0000185563 | 未通过 |
| CP2 | 0.022237778 | 0.005699873 | 0.0000214768 | 未通过 |
| CP4 | 0.013643265 | 0.006476164 | 0.0000156689 | 未通过 |
| TP2×EP2，整步 token 分母修复后 | 0.012481689 | 0.004909277 | 0.000013115 | 未通过 |

例如 CP4 未加权 MTP/aux 最大差为 0.021587213/0.156689202，TP2 为约 0.02070109/0.17280778。未加权 aux 也未达标。新生产 FP32 配方的 3 步集成通过仅证明短跑路径；早期诊断包装器的 200 步仍失败且梯度布局不同，不能当作生产补丁验收。

![FSDP 基线逐步对齐](figures/reviewed-fsdp8-baseline-zh.png)

![FSDP 续训逐步对齐](figures/reviewed-fsdp8-resume-zh.png)

![CP4 未加权 200 步误差](figures/reviewed-cp4-off-200-raw-zh.png)

![TP2 未加权 200 步误差](figures/reviewed-tp2-replicated-experts-build-200-raw-zh.png)

## 数值定位进展与未确认事项

- 梯度累积分母错误已单独修复，但不能解释全部剩余漂移。源参数释放前后同拓扑 200 步 loss/梯度范数完全一致，显存修复没有改变数值轨迹。
- TP 首步 Q/KV 投影和输出投影完整输入共 4608 个哈希块相同，首次已定位的输出偏离在第一层输出投影。局部投影使用 FP32 后，采样最大差从 0.015625 降至 0.0009765625，后续 MLP 仍放大残差；该消融改变数值配方，不能直接与旧 BF16 基线混比。
- EP/CP 的四个 NS 参数从单矩阵变为矩阵批次2。逐矩阵 NS 和 CPU FP64 范数的三步消融仍有非零差异；首步 28 个被观测参数哈希相同不代表全部专家参数相同。
- CP 首步仅有采样边界证据，不能称已定位完整根因。固定 512 位置 loss 分块使部分反向边界一致，但 embedding 梯度相对 L2 仍约 0.00532；只跑一步，不能替代 200 步验收。
- 重算灵活性评估结论：当前 JT YAML 尚未提供 MF 式任意模块路径/层范围、排除及通信重算配置。已有模块消融使用实验包装器；不能将 `selective` 名称视作相同接口。

## 显存、吞吐与重计算性价比

同一 build 观测口径、预热20步后中位时间如下，保留输入身份采集开销。这里的小层数配置尚未从并行切分取得吞吐收益，内存收益也随策略而异；不能外推完整 DeepSeek 模型。

| 策略 | 秒/步 | allocated 峰值 GiB |
|---|---:|---:|
| FSDP8 | 0.937752 | 22.633759 |
| EP2 | 1.020075 | 21.183023 |
| EP4 | 1.029181 | 21.322903 |
| CP2 | 1.285255 | 21.058210 |
| CP4 | 1.670697 | 20.217754 |
| TP2，复制专家 | 1.416853 | 17.771236 |
| FSDP8，FP32 专家梯度＋FP32 NS | 1.165173 | 22.282196 |

#198 的同拓扑前后比较：EP2 从 30.605194 降至 21.183023 GiB，CP2 从 31.183506 降至 21.058210 GiB；约 0.1% 的步时变化不足以证明吞吐收益。#200+#201 合用步时增加 24.25%，这是组合成本；此前约 6.8 倍的 NS 成本归因混入运行时观测，已撤回。

重算独立40步消融：full 节省约3.095 GiB、步时增加9.13%；selective 节省2.407 GiB、增加8.50%；单独 MLP 节省0.853 GiB、增加1.91%；只重算 MTP 未降低整步峰值、时间增加约2.99%。MLP＋norm 另有200步全 rank 零误差，节省约1.165 GiB、步时增加约2%。模块事件时间为含子调用的时间，不能嵌套相加；进入/退出 allocated 差也不能当作保存激活大小。

![源参数释放的显存收益](figures/source-parameter-lifetime-zh.png)

![模块重算成本比较](figures/recompute-module-cost-zh.png)

[同口径时间与显存原始记录](https://github.com/xingleixu123/hyper-parallel/blob/2f5ba55838338dcb5c37c9cd44a958586da2e89a/docs/validation/jt-dev-8npu-20261014/reports/matching-build-topology-costs.md)；[重算成本与口径](reports/recompute-cost-analysis.md)；[MF/Megatron 固定源码对照](reports/recompute-reference-comparison.md)。

## 固定环境、复现入口与证据

此前已测固定基点 `JT_dev@521d3a8a9aa62353327093ab1258ca2c22006e8b` 已合入 #180（head `5de3cceab094bbed18915d1a9c9231ca220629ca`）。旧八卡测量从 `ad1c553`＋#180 开始，四项修复集成源 `6e762be`、六项源 `ec2e16b`；129 该历史轮次整合源 `a3450812a3bb737e74b8433928acc6fc18374433`，再应用 #200/#201 生产补丁，补丁 SHA256 `803107108b505cd7ab7eafb7c625eb125ae0679e97cb1482f0914f3b3213f784`。

硬件：8×Ascend 910B3 64 GiB。CANN 9.2.0-beta.2 / V100R001C12B086，Python 3.12.14，Torch 2.12.0+cpu，TorchNPU 2.12.0，transformers 5.13.1。129 镜像 `jt-clean:hp-910b-20261009`，ID `sha256:5f9e677f439d11ab4e7c888f54b142335a27eee7411cbd2ce894a4798a70146d`，从 `/home/jt_image/four-images-clean-20261009/HP-910B/hp-910b-arrow.tar.gz` 校验后载入。镜像内原代码较旧，实际导入的是挂载并验证过的 editable checkout。

模型：两层主干＋一层 MTP、8个路由专家，序列4096、全局 batch8、微批1、seed42、固定 LR0.0001，Muon/AdamW 原有优化配方。真实 Arrow 数据4710行，原数据留在实验机，公开证据只包含配置、标量和身份哈希，不上传训练文本。

训练入口为 `examples/training_demo/jt_deepseek_v3`，使用证据中的完整 YAML、8进程启动及固定环境；逐步标量、输入哈希和退出标记必须一起采集。可复现配置示例：[CP4 200步](https://github.com/xingleixu123/hyper-parallel/blob/2f5ba55838338dcb5c37c9cd44a958586da2e89a/docs/validation/jt-dev-8npu-20261014/configs/reviewed-cp4-off-200.yaml)、[TP2 复制专家200步](https://github.com/xingleixu123/hyper-parallel/blob/2f5ba55838338dcb5c37c9cd44a958586da2e89a/docs/validation/jt-dev-8npu-20261014/configs/reviewed-tp2-replicated-experts-build-200.yaml)。[配置、原始标量、曲线和脚本索引](https://github.com/xingleixu123/hyper-parallel/blob/2f5ba55838338dcb5c37c9cd44a958586da2e89a/docs/validation/jt-dev-8npu-20261014/README.md)、[完整中文报告](https://github.com/xingleixu123/hyper-parallel/blob/2f5ba55838338dcb5c37c9cd44a958586da2e89a/docs/validation/jt-dev-8npu-20261014/REPORT_ZH.md)、[CP4 全步比较](https://github.com/xingleixu123/hyper-parallel/blob/2f5ba55838338dcb5c37c9cd44a958586da2e89a/docs/validation/jt-dev-8npu-20261014/reports/reviewed-cp4-off-200-comparison.json)。各小问题提供其专用最小复现和回归命令。

与 #145 的旧源码、单卡参考、100步问题以及 #190 的 MF 单卡对齐范围不同；本单固定 FSDP8、200步和恢复参考。尚没有能解释全部跨拓扑200步偏差的单一最小根因，后续证据确认后继续拆单。

## 待完成清单与资源约束

- [x] 已确认9个独立问题，各有中文小问题单与独立单提交 PR，正文相互关联。
- [x] 旧六项修复配方 FSDP0–200、200→400 与连续参考全 rank 零误差；关闭/完整/选择重算200步已完成。
- [x] 129 固定旧基点 `521d3a8` 的 FP32 配方 FSDP0–200 及200→400按顺序运行完成，数据完整、计数连续；未完成独立连续400参考。
- [ ] 最新 `f637ef7` 基点的 FSDP0–200、checkpoint200→400及独立连续400步完成；逐 rank 加权和未加权 loss 全部≤0.005。
- [ ] 新配方 TP/CP/EP 及合法组合200步全部达到相同门槛，剩余根因分别修复并建单。
- [ ] #214 的 Ascend/HCCL 补丁回归。
- [x] 完成 MF / Megatron 重算配置能力对照，明确当前 JT 与 MF 式模块/层范围、排除及通信配置的差距。
- [ ] 按最终配方复核关闭/完整/选择重算200步及资源收益；模块选择消融按其实际步数报告。
- [ ] 外部维护者评审、当前提交对应的完整 CI、最终回归与合入确认。

仅在125/129获准的空闲设备上运行，优先保证其他实验；任何两轮 NPU 实验之间全局至少间隔20分钟，等待期间不占卡，冲突只释放本任务容器，不停止他人进程。八卡训练每轮有900秒上限，不常驻排队占卡。清理仅限本任务已核验无用的临时数据和重复压缩包，checkpoint200 保留到续训与连续参考比对完毕；原始镜像、共享数据和其他任务文件不删除。
