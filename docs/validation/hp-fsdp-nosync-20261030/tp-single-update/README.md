# TP 单步 owner-aware 数值证据候选

本材料是**待审核的公开候选 v2**，未发布；展示两次完整模型 TP 单步的有界数值证据，不是训练精度、显存或性能验收。`compared` 只表示比较已完成，没有设置或放宽数值通过阈值。ROOT 的语义审核（SHA-256 `499a94f55ce6662e9eca253f619efe086c89bbd8e1df9e38ce6ac987e6eb22e2`）认可继续进行后续实验，同时明确 `final_training_acceptance=false`。

两臂从相同 native 1500 步恢复后，各执行一次 step 1501；策略差异是梯度同步时机，固定 TP2/DP2、GBS4、LBS1、序列长度 4096、原模型和优化器配置。全部源码以原 Git head `1a39572fef23819a2d2651b4bd53047a05a4892f` 为基底，唯一源码覆盖为 scheduler 文件 SHA-256 `51abc1d50805960b50c334c8f5a5b5c4a5b2a5e0d7984686bae3d2e553499fe2`；2,661 项实际源码清单 SHA-256 `0f168e2ce3279cb315fc27d60d06693a17a5626a228fbf777146dda9da3e977d`。实际模型参数、优化器 slot 值、完整 RNG、YAML、训练样本和每 rank 原始文件均不在本候选中。

## 首次断言与 owner 依据

首次 CPU 全量比较在 coverage 阶段以 `replicated region bytes differ` 退出 1，后续 native 阶段未开始。第一个定位项为 `model.hc_head.hc_fn.momentum_buffer`；原断言把 live Muon momentum 的同一区域一律视作逐字节相同，条件过宽。这个比较器错误本身不能证明训练错误。首次失败的完整私有 ZIP SHA-256 `ef584916a696def8ce7f847252a061acb12b17f74d89d16cb9bf3d1a22dbed98`，其错误输出 SHA-256 `28a75940a8e6e41da71ff69e1243f528839663e1be95adb19026df6a2fb03fa6`；失败记录未被覆盖。

随后实际进行了四 rank 的 NPU **restore-only、零更新** owner 缓存观察，完成初态 258 role 审计和物理映射审查。每 rank 记录 28 个 Muon FQN；不能仅从 slot hash 是否变化推断 owner。ROOT 按固定源码依赖、两旧臂/观察的初态、参数组与成员顺序、layout、mesh/PG、环境以及缓存生命周期条件审核后，允许将后续观察**有条件地关联**到两次旧执行；该观察不是旧进程现场录像。条件回执 SHA-256 `9f3aa3af415ce0ca80c1f7c0efb950ba2fde3df3ee266123e886411f7b442f31`，绑定 SHA-256 `de7e8915ec56d2e48e14a174501c32bb41dfe925a0833ebb285e3cb241412eea`。

每臂 201 项 coverage 包括 57 gradient、57 model、1 buffer、86 slot 角色。每臂 92 个 gradient、92 个 model 和 1 个 buffer 重复区域，以及 108 个普通 slot 重复区域，继续使用严格逐字节检查；19 个 Muon FQN 的 38 个重复区域按上述有条件的 owner 缓存语义处理，其 nonowner 后值均匹配各自初态。两臂共有 76 条此类 nonowner slot 记录；这个例外不适用于其他 slot。空 local shard 单独计数，并非缺失。

## 已完成的 CPU 比较

最终 CPU 终态 exit 0、无 OOM；同 rank 跨臂比较包括 804 个 role/rank 行，201 项逻辑 coverage；native 读取覆盖 144 个全局 tensor、680 个 optimizer byte 条目、258 个 source role，生成 1,152 个初态到后值的 arm/rank/role delta。记录中非有限元素为 0、缺失梯度为 0。804 和 1,152 都是行数，包含 rank 和重复区域，不表示同样数量的独立参数。

| 类别 | 两臂同 rank 最大绝对差 | sync 相对 native1500 最大绝对更新 | deferred 相对 native1500 最大绝对更新 |
| --- | ---: | ---: | ---: |
| preclip gradient | 4.430394619703293e-05 | 不适用 | 不适用 |
| model | 6.386660970747471e-06 | 1.4302786439657211e-05 | 1.4537479728460312e-05 |
| buffer | 0 | 0.0010000020265579224 | 0.0010000020265579224 |
| optimizer slot | 1.1376803740859032e-05 | 0.002625559864100069 | 0.0026255419361405075 |

两臂之差与每臂对初态的更新是不同统计；例如 router correction bias 的 64 个值在两臂间相同，但相对初态都发生变化。四 rank 报告的裁剪返回 norm 跨臂绝对差为 `2.4557113647460938e-05`，scale 绝对差为 `1.817941665649414e-06`。最大相对 L2 可能受很小分母影响；这是描述性统计，不是自动精度判据。最大差的 FQN 和 rank、空分片及其余聚合数见 [aggregate.json](aggregate.json)。

通用 source-region cache 的 `target_verified=false` 如实保留。另一个 delta 读取器根据实际 target shape、placement、mesh 从 native 全局 tensor 取片，验证初始片段 SHA，并核验 post SHA；独立摘要逐条关联 1,152 条初始/后值 SHA。1,104 条目标区域也出现在通用 cache 的区域表，另 48 条未列在该表，但由独立 delta 路径验证；没有把通用 cache 标记改成 `true`。完整 CPU 原始结果的私有 27 文件 ZIP SHA-256 `24a782795f0c9ab4cfbe0f807dc8386febaa295b25e4beaf955949888ca8372f`。

## branch_alpha 标量诊断

五个 `branch_alpha` FQN、rank 2/3、两臂共有 20 个标量实例。根据初态 L2 与 SHA 唯一约束符号，在内存核验 60 个 model/momentum scalar；私有 post 样本 SHA 也经校验。一个独立 CPU FP32 舍入轨迹给出 40/40 slot 位模式匹配，20/20 参数后值与初态一致且该轨迹的参数预测一致；20/20 的保守更新上界低于半个 FP32 ULP，最大上界/半 ULP 比为 `0.9172806749120356`。存在 subnormal 初态。

这些结果只说明一个可行的舍入解释；它不是 NPU 融合 kernel、FMA 或 FTZ 执行轨迹的证明，也不是独立全模型优化器 oracle。原私有补充材料 SHA-256 `f3b8da000ef60dcccb8e86646ed9f5fbd49e90993b1c6d2d56a6e2537b9bd11b`，其中有实际标量，不作为公开文件复制。此处只提供 [scalar-diagnostic.json](scalar-diagnostic.json) 聚合和不输出原值的重算代码。

## 来源核验和复现边界

[aggregate.json](aggregate.json) 列明完整 CPU 终态、各阶段 JSON、失败归档、owner 观察、ROOT 两项审核及摘要的 SHA-256，另记录执行比较器、native delta 和读取器的 SHA。固定 Git blob 的优化器依赖 SHA 已与上述 actual source manifest 条目复核。摘要本身的校验清单见 [SHA256SUMS](SHA256SUMS)。

[tools/rebuild_public.py](tools/rebuild_public.py) 在持有私有只读结果和 ROOT 条件关联回执的环境中，校验回执、owner 绑定及同包审计 SHA 后生成白名单聚合；[tools/recheck_scalar_private.py](tools/recheck_scalar_private.py) 需额外持有四份私有 rank 报告，只在内存中使用标量样本并输出聚合。两者均可用 `python -B <工具> --help` 查看所需路径参数，`--out` 必须是新文件。它们不下载或包含训练数据。

[tools/original_compare/owner_compare.py](tools/original_compare/owner_compare.py) 及同目录四个依赖/算法文件与实际运行代码逐字节一致，供持有原始私有输入、owner 观察和审核回执的人员核对或复算；需要 NumPy，其中 `owner_compare.py --help` 列出输入。`owner_evidence.accept_review` 还会读取同目录的原始 [source-dependency-audit.md](tools/original_compare/source-dependency-audit.md)，其 SHA-256 `f3a115b210975a24319ac1e5aad8103953884cdb00d68f9d0936d5f610bb29df` 与 ROOT 条件回执一致；该审计是形成时的待审核历史快照，后续 ROOT 回执才作了有条件认可。文内两个固定 Git 源摘录及校验引用也随原文附带。代码为核对原运行身份保留必要 run ID 和配置摘要，不含运行数据；运行产生的逐角色结果仍须作为私有资料保存。`native_delta.py`、`regions.py` 提供实际 native 切片及更新比较算法；完整 native DCP 读取还依赖未在本包分发的环境相关读取器，已记录其源 SHA，不声称单凭此包即可重跑完整 native 读取。仅持本公开包的读者可以核对公开文件与所列来源指纹，但无法独立重算未公开的私有原始数据。

## 后续结果接口

本包不填入或预先判定 TP40/1000 的数据：`aggregate.json` 中相应字段为 `pending_unpopulated`，只表示本包尚未纳入这些结果。计划接入同一 source 的 sync40 与 deferred 首 40 步（10 步预热、30 步计时）作同期比较；deferred 后续在同一进程组继续至 2500 步，形成 1000 步损失序列。若与既有原 1a、FSDP4、selective recompute 的千步基线比较，结论涉及 TP + full recompute + deferred 的组合，不能归因于 no-sync 单项。实际结果、版本、口径和回执必须另行核对；原有 0.005 训练验收标准不在此修改。
