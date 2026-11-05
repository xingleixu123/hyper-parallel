# PR #1439：0854496d 检视复核与整改

受检文档：`PR1439-检视意见-0854496d.md`，2026-11-05 08:07 更新，共 1088 行。对照受检提交 `0854496de0004a01c04678d3f49983d2bf14ea27`、当日主仓 `a9fa47d64c46d61dd49fc8e56c97799a6aea4a88`、20 条 inline comments、14 次 reviews、5 条 conversation comments及10个 review threads。本报告区分代码缺陷、已知依赖、未证实性能假设和不适用建议。

最终模型提交 **f96dd5b091f75ba4ec827f73e079b3b21e9174db**，公共 hook 提交 **455edbda957e1650d6f36c1618694f2c1bf248cd**。公共问题 [#1454](https://github.com/mindspore-ai/hyper-parallel/issues/1454) / Draft [#1455](https://github.com/mindspore-ai/hyper-parallel/pull/1455)；专家契约 [#1456](https://github.com/mindspore-ai/hyper-parallel/issues/1456) 随 [#1439](https://github.com/mindspore-ai/hyper-parallel/pull/1439) 修复。主仓依赖仍需按表合入，未宣称 bare base 独立运行。

## 本轮改动边界

1. **专家替换契约**：仅接受确切的官方 Xing4 MLP 或训练 MLP；拒绝覆写前向、未知子类、异构激活策略及修改过的 SwiGLU 函数。正常模型仍复用公共 GroupedExperts，未重写计算。
2. **公共 post-step hook**：给 ChainedOptimizer 补标准注册/移除能力；在所有叶优化器成功后调用，保留 closure 参数、异常和返回行为。Xing QKClip 改为注册 hook，移除实例 step 改写。公共能力单独提交，不把依赖 PR 的改动混入 #1439。
3. **EP fix_router**：透传公共接口，默认 False。仅用于基准测试的分发均衡；不替代安全补零，不作为精度验收配置。

官方三个源文件保留原名、原字节。#1439 保留三个提交，前两个不变，本轮并入第三个提交；不新增 UT/ST/recipe 到产品 PR，外部探针及复现文件归档。发布前后核验原有检视记录，不自动 resolve、dismiss 或删除。两个产品PR均为Draft；原#304不改。

## 逐项判断

| 文档项 | 判断及处理 |
|---|---|
| §3、P0-5：Cui 与 hf1s 的结构意见 | 接受需要明确答复。用户后续明确采用 transformer/ 下原样官方文件、adapter/ 下训练差异，故不再移回大 modeling 文件。将训练语义、融合替换和并行边界逐项答复。 |
| P0-1 / §6.1：公共 MTP 缺失 | 成立，是合入前置。#1386 尚未合入受检 master；产品 bare base 不能构建该训练入口。实跑组合有此能力，二者不能混称。已有 build_model 顶层导入和构造后类型检查，不会成功静默变成 HF 推理训练。 |
| P0-2 / §6.2：CP 回调缺失 | 成立。#1425 的 rotary_fn/attention_interface 是前置；该 PR 含旧 Xing40 接入，需提取/协调公共部分而非整包叠加。 |
| P0-3 / G2 / N3：可选 SwiGLU | 部分成立。裸 base 无扩展 API；#1420 只改名，不提供该函数。#356 的函数和 GroupedExperts 策略消费必须成套。目标配方两项为空，不能用默认 smoke 宣称可选策略已验收。按需 import 是正常可选依赖，启用时现有路径会 ImportError，不是默认必须导入全部后端。 |
| §12.4：carrier 策略无人读 | 对裸 base 成立，对实际组合不成立。#356 的两个公共文件与冻结运行时逐字一致，消费 limit/dtype；不能删除这些字段或另写第三套数学。部分应用依赖仍存在风险，依赖清单明确成套要求。 |
| P0-4 / G4 / §7.1：mHC 公共 pair 一定错误 | 不成立。公共 source-first 公式为 Hᵀ@streams，Xing 官方为 H@streams。MF formal 同时有两种明确模式。非对称 NPU 探针验证 Xing 转置实现其公式，不能给公共模块全局翻转方向。见下面实测。 |
| §7.2：Sinkhorn 算术完全等价 | 不成立。官方与 DSv4 的 epsilon 位置、投影 dtype 和 clamp 有差异；需按训练配置/算子契约解释，不以轴顺序近似证明全部浮点相同。 |
| §7.3 / §13.5：clamp 未触发便忽略 | 不采纳。有限样本不证明未来不触发；配置明确 clamp 关，显式要求 clamp 则报不支持，不静默忽略。 |
| P0-6 / §11.3：没有入库 UT，必须补并 squash | 测试证据应可复核这一点成立。本轮用户明确逻辑代码-only、保留三个提交，优先于通用模板。外部测试不等于 CI 通过，结果按版本归档。 |
| N4：无来源/许可 | “无来源”不成立，原 PR 描述已有官方固定来源链接；本轮再保存原始 README，其 metadata 为 apache-2.0。官方文件原本没有项目 Apache 头，按用户原样保留要求不伪加 Huawei 版权、不改字节。来源/哈希和许可记录归档。 |
| N1 / G3：ChainedOptimizer 缺少 hook | 成立。实际 MoEMonitorCallback.register 会失败；当前 Trainer 手工 on_step_end 不受此缺口影响，不能说历史训练失效。补公共 API，Xing 使用它；不重复注册 Trainer 的 MoE monitor。 |
| N2 / §14.2：直接换公共 compute_mtp_loss | 条件不满足。#1386 的 full-logits + valid_tokens 可以值/梯度相等，但默认 sequence 分母不同，且不提供 chunkCE/TP loss parallel/CP 归约接口。公共通用 MTP 裁短 S−k，当前填充尾部参与 router 计数/aux；直接替换会改变训练语义。保留必要接线，不扩展公共库来制造行数下降。 |
| N5 / §9.2：fix_router 替换 padding | 不成立。固定路由改真实 token 目的地；安全补零只加零权重槽并排除统计。接受 EP 暴露公共 benchmark 参数；不替换 padding。 |
| G1：公共 ModuleList packing | 有复用空间，但没有找到可直接替换且覆盖此权重布局/状态的已有接口。将模型拓扑原样搬到公共目录不构成正确抽象。保留必要转换，修复源 forward/策略校验缺口；不按估计行数引入新框架。 |
| G5：跨包私有名公开 | 是独立 API 清理建议，不是已证实的本轮错误。现有 MLA 扩展点保留，其余不顺带改所有调用者。 |
| G6 / §9.4：混合 HF-native GC 与 regions | 未证实收益或正确性缺陷。当前 region provider 覆盖主干和 MTP，已有 NPU 结果。新增混合调度需独立设计/对照，不凭代码可达性推导性能提升。 |
| G7：MergeModulelist | 复用在途 #1442 的保维转换；普通维度可用既有 HF fallback，单例维度显式拒绝，避免静默 squeeze。 |
| G8：重复 MoE utils | 确有不同公共实现，Xing 使用 Trainer 实际消费的 components 版本。公共去重涉及其他调用者，不混入模型修订，也不宣称已解决。 |
| §5.1 / §13.9：删除 Qwen 别名 | 不采纳。该行是 base 已有上下文，git diff 只新增 Xing 两行。删除它会造成无关回归。 |
| §6.4：公开 HF MTP 权重直接恢复 | 仍有限制。官方发布权重缺少目标训练 MTP 的六项 mHC 参数，单纯改名不能生成它们。本任务从完整 MF checkpoint 恢复，不能宣传任意 HF 权重全状态直载。 |
| §6.5 / §12.9：三次 post_init 就初始化三遍 | 不成立。Transformers 初始化标记与已保存构造探针证明已初始化模块会跳过；调用次数不等于参数重置次数。 |
| §13.12：original_inv_freq 共享 | 不成立。公共 register_rebuildable_buffer 对源执行 clone；传同一临时 Tensor 不代表持久 storage 共享。 |
| §7.4：chunk loss 与 TP loss parallel 冲突 | 不成立。当前按是否持有完整词表走互斥分支；单卡 chunkCE，TP 走模块/Trainer 安装的 loss parallel，并未同时执行两个 loss。 |
| §8.3：排除 MTP 又包裹其子层矛盾 | 不成立。exclude 限制自动 DecoderLayer 发现；显式 wrap_modules 后加入 MTP depth。已有 EP/FSDP NPU 结果需按运行版本引用。 |
| §9.1：mHC 20 轮 Python Sinkhorn | 不成立。实际 sinkhorn 调 _Sinkhorn.apply → NPU custom op；文档误把另一个 sinkhorn_knopps 的循环当作此路径。不能据此给性能排序。 |
| §9.3：复用实际路由 indices 作 aux | 不采纳。实际路由有 correction bias，seq aux 用未加 bias 的 affinity Top-K；MF 参考明确如此。复用会改变目标。 |
| §9.5 / §12.5：Muon 两条 row key 是死代码 | 不成立。实际公共 SwiGLUMLP 替换会产生 mlp.linear_fc1.weight 与 shared_experts.linear_fc1.weight，探针已触发并验证逻辑分块。默认配方未触发不等于接口死代码。 |
| §10.1：替换窗口约双份显存 | 参数量推导可用于估算，尚非实测泄漏。当前逐专家 copy 避免整批 stack 副本；不在所有权未证明时 resize_(0)，否则破坏外部别名。控制并发的峰值测量仍单列。 |
| §10.2：MTP expand/mean 可直接删 | 暂不采纳。流展开进入真正 mHC decoder，不是可消去的纯 expand→mean 恒等式；会影响残差混合与梯度。 |
| §13.15 / §13.28：任意 CP 分布和 PP | 不宣称支持。当前验证连续 CP token 分片及既定 FSDP/TP/EP/recompute；任意非连续输入、PP 需另有接口与测试，不在本轮补未验证能力。 |
| §13.16/17/19/22/27：删官方文件、换目录、加 recipe/CPU fallback | 与最新明确范围冲突。不删除 tokenizer，不移动官方源，不加入本轮排除的 UT/ST/YAML，不为 CPU 测试加产品后端判断。 |
| §14.6：优化器校验要移到模型 | logical-layout 依赖优化器选择和替换后的参数布局，在 builder 中校验有明确作用；普通模型并不总用此 Muon 分组。未发现移走有功能收益。 |
| §14.6：del labels/kwargs/num_recomputations | shift_labels 已是预移位权威目标；router 的 is_recomputing 已阻止重算计数。不能据 ignored 参数名推导漏算，也不能为 num_recomputations≠1 新增错误。外部自建 optimizer 不自动具备 Xing builder 注册的 QKClip。 |
| §11.3：label-check 来自许可/提交数 | 缺证据。实际 annotation 是等待新 head 的 pipeline-passed；当前仍不能宣称 CI green。按当前 SHA 重新触发/观察，不把推测写成门禁根因。 |

## mHC 的独立证据

129 物理卡 5、Ascend910B3，Torch 2.12.0+cpu / PTA2.12.0.post3 / CANN9.2.0 B237。形状 [1,16,4,512]，BF16 输入、FP32 phi/alpha/beta、seed107，20次 Sinkhorn。真实公共 pre 输出满足 max|H−Hᵀ|=0.741175，避免近均匀初始化掩盖方向。

| 对照 | 输出 max|Δ| | 输入梯度 max|Δ| |
|---|---:|---:|
| 公共原样 vs source-first 独立公式 | 0.0001221 | 0.0078125 |
| 公共转置 vs destination-first 独立公式 | 0.00001526 | 0.00000763 |
| Xing _post vs 公共转置 | 0 | 0 |
| 公共原样 vs 转置 | 1.625 | 7.234375 |

phi/alpha/beta 三组梯度均另存。官方 pre 与公共 fused pre 本身有 BF16/FP32、epsilon 与算子差异，官方对照不是逐位等价；不能把“更近”作为模型正确性的依据。完整公式、四项梯度、输入和原始张量见 `mhc/方向与GateA核验.md`。

Gate A 单列：不带融合 plan 的目标 builder 明确拒绝；直接 from_config 不做 dtype 准备会发生 BF16 activation/FP32 weight 不匹配，按已确认 precision 配置准备后 CPU 前反向成功。它不证明默认任意构造、NPU融合与官方参考完全相同。

## 验证与历史证据的边界

- CPU：固定旧 SHA 与最终候选 SHA 的专家红→绿反例、meta/eager 官方类、激活策略一致性；公共 EP factory 默认 false 的接线；hook 的 closure、移除、异常、state_dict、主权重时序。测试文件独立归档，未混入逻辑 PR。
- NPU：公共 mHC 非对称前反向；公共 hook 的真实 BF16/FP32 更新与独立 QKClip 数学 oracle；专家旧/新42项张量逐位相同。最终 f96dd5b 专家文件和455edbda公共hook均已分别重新上NPU；42项固定fixture/输出/梯度逐位对照由主代理读取原始Tensor再次确认。
- 训练器：最终 f96dd5b + 公共455edbda叠加在571d冻结组合上，从MF1500状态跑完1501–1505。65模型项、91槽、61主权重别名、scheduler1500和sampler3000核验；10微批40字段与MF及原HP全相等。train_iters=2500、LR decay horizon7500/warmup1500未改，短跑只提前停止。独立主代理已复算原始ledger和对照值。
- 已有完整单卡1501–2500结果归属于冻结组合571d9cb9；LOSS abs(mean)=0.000022688388824462892，MTPLOSS=0.00008995433151721955。四卡100步归属于d7802906；不将本轮小探针冒充新的完整1000步验收。
- 原要求四卡1500→2500长程、独立性能/峰值显存测量仍未完成。当前短程证据和历史长程证据分别引用。

## 复现索引

- `candidate-final.json`：最终提交、四份源哈希、改动范围。
- `modules/candidate-report.md`、`modules/check-*.json`、`modules/npu/`：专家语义与公共 EP 接线。
- `muon/review.md`、`muon/probe-manifest.json`、`muon/npu/`：公共 hook、逻辑矩阵规则、MTP loss 适用条件。
- `mhc/方向与GateA核验.md`：mHC 源/目标轴、NPU原值和构造边界。
- `lint-summary.json`：同环境、同工具、同禁用项的前后诊断差异，不把局部 lint 当全部 CI。
- `pr-before.json`、`guard-before.json`：变更前 PR/评论快照。发布后另有逐 ID/body/state 与 thread 校验收据。

## 本轮五步原始损失（集成回归）

| step | MF LOSS | HP LOSS | HP−MF | MF MTPLOSS | HP MTPLOSS | HP−MF |
|---|---:|---:|---:|---:|---:|---:|
| 1501 | 7.8989529610 | 7.8989086151 | -0.0000443459 | 7.9833512306 | 7.9839789867 | 0.0006277561 |
| 1502 | 6.8772039413 | 6.8765656948 | -0.0006382465 | 6.9048018456 | 6.9039314985 | -0.0008703470 |
| 1503 | 6.9835929871 | 6.9823341370 | -0.0012588501 | 7.0637860298 | 7.0632392168 | -0.0005468130 |
| 1504 | 6.4562149048 | 6.4569423199 | 0.0007274151 | 6.6977767944 | 6.6972419620 | -0.0005348325 |
| 1505 | 7.0971899033 | 7.0970497131 | -0.0001401901 | 7.2344670296 | 7.2333896160 | -0.0010774136 |

这里只是五步恢复集成回归，0.005的正式判据仍约束完整1000步有符号均值。未将短窗代替长窗。逐步原值见 [CSV](single-1501-1505.csv)，主代理复核见 [JSON](root-verification.json)。

## 可在线复核的本轮附件

[版本与四文件哈希](candidate-final.json) · [主代理独立复算](root-verification.json) · [五步原值CSV](single-1501-1505.csv) · [mHC实测统计](mhc-result.json) · [最终专家42项对照](grouped-final-comparison.json) · [源契约反例](contract-cases.json) · [eager/meta测试](official-class-cases.json) · [EP开关接线](fix-router-cases.json) · [同环境lint](lint-summary.json)

NPU独立探针：[公共hook](hook-device-probe.py)、[分组专家](grouped-device-probe.py)、[mHC方向](mhc-device-probe.py)。探针使用已记录的组合依赖、源哈希和NPU环境，不承诺裸master可以执行；本目录不包含训练YAML或checkpoint。源参数与实际输入的原始Tensor证据保留在本任务本地归档。