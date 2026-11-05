# PR #1439：1105 更新检视的复核与增量

检视输入为 hp-pr-1439-review-20261105/README.md 的 16:39 更新版，SHA256 801e2f3ff305cc470da3a3830f98c57fbb6c977383187926feb1753285b97eb0。本次比首轮增加132行、删除6行；新增第零章已逐项核对，旧章节按前轮复核保留。

## 提交及独立问题

| 范围 | 提交 / 问题 | 必要修改 |
|---|---|---|
| 现有 #1439 追加第6提交 | e4013cfb2539d81709015cada6a788517a7a28e0；[初始化问题 #1466](https://github.com/mindspore-ai/hyper-parallel/issues/1466) | 4文件，+11/−14。使用HF受保护的nn.init初始化路由和mHC状态；复用原FP32 helper名、去掉额外别名；补充必要契约注释。 |
| 独立公共loader补丁 | 5fcd3a8a940c767b0da1b26835822071900c91ae；[问题 #1467](https://github.com/mindspore-ai/hyper-parallel/issues/1467) | 基于master c108cf94，1文件+4/−2。保存过滤前missing集合用于初始化，过滤后集合用于strict和报告。 |
| 原有激活策略 | [hf1s#356](https://github.com/hf1s/hyper-parallel/pull/356)、[#1420](https://github.com/mindspore-ai/hyper-parallel/pull/1420) | 继续复用，未重复创建激活实现或能力协议。 |

#1439保留原五个提交，不amend/rebase/强推，官方HF三个源文件不改字节。公共修复不混进模型PR；新模型提交也没有加入公开HF冷启动的缺键豁免规则。UT/ST依用户本轮范围留在仓外证据，不宣称库内已有测试或外部CI通过。

## 更新检视逐项决定

| 条目 | 复核结果 | 处理 |
|---|---|---|
| 0.3 FP32偏置与int64归约 | 已有修复和公共MoEMonitor复用成立。 | 保留；注释说明forward已排除重计算。不把任意局部重算误计都等同全局统一倍数。 |
| 0.4 持久计数与HF strict冷加载 | 缺键风险确实存在；MF1500/原生FSDP step5恢复不是官方HF冷启动。 | 保留persistent schema；CPU真实DCP反例显示直接改nonpersistent会丢弃旧非零计数，反向加载也缺键。未采用“两行严格更优”的建议。 |
| 0.4 深入检查发现的公共缺口 | ignore missing在HP loader中同时删掉初始化目标，strict可能返回带未初始化存储的模型。 | 独立#1467及公共补丁，按实际缺失项初始化，不放宽其他缺键。 |
| 0.4 深入检查发现的模型缺口 | HF部分加载触发初始化时，Xing直接zero_/fill_绕过HF guard，覆写已加载bias/mHC状态。 | #1466在#1439追加修复，改用nn.init.zeros_/constant_。不新建一套初始化器。 |
| 0.5 SwiGLU在裸master不可用 | 非空策略依赖尚未合入的完整#356；#1420仅提供导出文件改名，并非全部策略实现。默认空策略不触发该条件。 | 保留已验证的条件能力与明确失败。partial检查是锁定#356实现的过渡保护，不冒称通用能力协议。公共构造器当前没有packed activation注入接口；新协议需协调既有#356，而非重复提PR。 |
| 0.6 helper重命名 | 原函数体本身已支持buffer，新增名字不是行为需要。 | 改回preserve_fp32_parameters并更新说明，移除冗余alias，不改函数算法。 |
| 0.7 专家forward | 原透传签名确为ModuleList替换验证所需。 | 保留并明确docstring，复用已有20项NPU功能断言。 |
| 0.9 测试、门禁与Draft | 未上库UT是用户已明确的提交范围；仓外结果不能等同CI绿。 | 保留Draft，单独核当前head门禁。没有删除或自动resolve检视。 |

[冷加载/DCP兼容性分析](cold-load-review.md) · [真实HF初始化分析](hf-ignored-missing-audit.md) · [SwiGLU与helper复用核查](activation-contract-review.md)

## 模型初始化红绿对照

使用真实Transformers PreTrainedModel/from_pretrained、safetensors、Xing4Router及mHC模块，同一adapter.initialize代码：

| 情况 | 7f4a3fd7旧版 | e4013cfb候选 |
|---|---|---|
| 缺count，已载bias=3、weight=7 | count=0，bias被清0，weight=7 | count=0，bias=3，weight=7 |
| count和bias同缺 | 都初始化为0 | 都初始化为0 |
| count=19已提供 | 保留19 | 保留19 |
| 缺hc_fn，已载scale=2、base=4 | scale变0.125，base变0 | 保留2、4 |

固定seed随机初始化全部state tensor SHA256逐项一致；已有CPU24项、14子测试通过。[脚本](hf-guard-regression.py)、[旧输出](hf-guard-candidate.log)、[新输出](hf-guard-hf-init-stage.log)、[CPU说明](hf-guard-review.md)。

129物理卡5受控专项22项通过，包含原20项和新增固定seed/真实HF部分加载。新增from_pretrained用例先在CPU加载，再转NPU复核；不能称整个HF加载在NPU执行。首次新增fixture未走create_config而缺少配置字段导致构建失败，已修正fixture并保留失败记录。[NPU报告](npu-followup-report.md)、[22项日志](attempt2/stdout.log)。

## 公共loader红绿及边界

CPU以master c108cf94完整源码stage验证，模型遵循HF guarded nn.init契约；实际调用load_pretrained_weights读取safetensors，legacy与DCP两路径均覆盖：

- ignored parameter/buffer旧版保持未初始化哨兵71/71；新版本分别初始化3/5。
- 已加载兄弟参数/缓冲保持原值；meta/to_empty后的nonpersistent scratch经现有rebuild保持17。
- 完整权重加载不改值；strict下其他未忽略缺键仍报错。
- tied alias解析后不重复初始化，init_calls=0。
- 真实HP DTensor缺失项在两后端都明确要求DTensor-aware initialization，未静默使用空存储；这是一项从静默跳过变为拒绝的有意行为变化。
- 既有加载测试16项、18子测试通过。独立stage缺项目pytest标记注册带来6条UnknownMarkWarning，未为消除警告修改产品。

[公共回归详情](loader-regression.md)、[真实文件探针](loader-regression-probe.py)、[边界探针](loader-boundary-probe.py)。

另在NPU张量上实际执行before/after finalizer：旧count=[71,71]，新count=[0,0]，已加载weight/bias均保持，strict报告missing=[]。[直接数值日志](attempta4/stdout.log)。该验证为单卡模块边界，不是多rank DCP恢复验收。

公共补丁沿用HF对nn.init的逐张量保护；不会自动保护自定义_init_weights直接zero_/copy_等绕过guard的操作，Xing的这种行为由模型补丁修正。不引入全量权重快照或逐层额外副本。

## 版本、证据与验收范围

CPU：Python3.12.3、Torch2.12.0+cpu、Transformers5.22.1。NPU：Python3.12.14、Torch2.12.0+cpu、PTA2.12.0.post3、Transformers5.21.0、CANN9.2 V100R001C12SPC001B237、910B3。NPU四文件叠加源共1514文件，manifest ea0bb3500e6b4276fddaa97475918c56501efe2c309e320f56030a88c4d3f233，运行前后核验相同。新master c108cf94已包含#1422数据流水线改动，本轮没有将其冒认为旧训练组合的一部分。

[提交/四文件绑定](xing-commit-binding.json)、[公共补丁绑定](loader-commit-binding.json)、[负责人独立复核](root-evidence-verification.json)、[NPU文件哈希](npu-evidence-sha256.json)。

控制器源自旧模板，原controller.json中的四个SHA展示字段未更新，不能用于本次代码身份结论。实际source manifest、worker/test及控制器文件的独立hash已核对；设备cgroup、PID namespace、npu-smi对应卡5。本说明保留原日志，没有回填或覆盖旧字段。详情见NPU报告。最后两次容器均已移除，卡5空闲。

5个相关文件同配置pylint前后均24条、退出22、新增0；不是lint全绿。语法编译、git diff --check通过。[lint对照](lint-comparison.json)。

**本轮不重跑MF/HP完整1000步或四卡训练。** 本次修改影响部分加载初始化；随机初始化逐位一致，持久状态schema、并行plan、forward/backward和优化器数学未改。历史5/100步及完整1000步各自保留原提交归属，不将旧曲线改标新head。[上一轮训练与性能报告](https://github.com/xingleixu123/hyper-parallel/blob/c7f67b63abcdd3a9b0c00bf2f3876aa634cce73a/reports/xing4-readme-review-20261105/README.md)。

公开官方Xing4权重冷启动还涉及训练MTP缺失mHC参数及映射边界；本轮两个修复不表示完整官方checkpoint已可直接恢复当前MF训练状态。原MF1500 checkpoint、原生FSDP step5及真实数据/配置保持归档；YAML仍仅在私仓。总跟踪更新#353，#304不修改。
