# 本次代码、issue 与 PR 提交覆盖核对

核对日期：2026-10-18。结论限定于**本轮已经确认的 HP 产品修复及提交清单**：没有发现未提交的正式训练源码文件，必要产品修复都有对应 issue/PR；本次补齐了转换工具 PR #166 缺少专门子 issue 和详细描述的关联遗漏，新增 #247。

**提交齐全不代表代码全部修完或可合入。** PR #166 的缺槽保护/独立回归与 Draft #245 的接口缺口仍未完成，必须明确保留为待办，不能因已有 PR 就宣布完成。

## 已确认产品修复

| 问题 | 对应 PR | 当前状态 / 精确范围 |
|---|---|---|
| #152 router 计算精度 | [#148](https://github.com/hf1s/hyper-parallel/pull/148) | 已合入，FP32 router GEMM；不等同所有 gate 参数副本始终 FP32 |
| #216 token 恒等梯度 | [#180](https://github.com/hf1s/hyper-parallel/pull/180) | 已合入，按全局有效 token 均值推导比例 |
| #187 K-shift 环境接管 | [#232](https://github.com/hf1s/hyper-parallel/pull/232) | 已合入，保留调用者环境；旧替代 #192 仍 Draft，不重复叠加 |
| #215 已有 FP32 update 单分片写回 | [#213](https://github.com/hf1s/hyper-parallel/pull/213) | 已提交、未合入；条件修复，不声称原生默认 Muon 普遍有 FP32 update |
| #242 JT 原投影存储等价性 | [#243](https://github.com/hf1s/hyper-parallel/pull/243) | 已提交、未合入；只恢复 Q/latent-KV、expert gate/up 原逻辑边界 |

当前 JT 的 FP32 rotary 已在 [#186](https://github.com/hf1s/hyper-parallel/pull/186) 合入的 `dfae97cd4` 中存在，旧 #156 不属于当前待补代码；4K Arrow 及相关运行时已有 [#136](https://github.com/hf1s/hyper-parallel/pull/136) 合入。没有为这些已有基线功能重复提交修复。

## 转换、诊断与候选配方

| 问题 / 用途 | 对应 PR | 状态及未完成项 |
|---|---|---|
| [#247 完整 checkpoint 转换](https://github.com/hf1s/hyper-parallel/issues/247) | [#166](https://github.com/hf1s/hyper-parallel/pull/166) | PR 已存在；本次补专门子 issue 与中文详细正文。缺槽拒绝、占位区分、元数据/范围校验及固定 head 的独立回归尚未完成，不标成全部修好 |
| #240 共享 down/FP64 诊断 | [#241](https://github.com/hf1s/hyper-parallel/pull/241) | 已提交 examples 只读工具；不是公共 optimizer 数值替换 |
| #210 NS 精度能力 | [#201](https://github.com/hf1s/hyper-parallel/pull/201) | 已有单提交 Draft，可配置能力；不是 BF16 默认错误的修复，不修改其它八卡工作流 |
| #244 旧 MF 显式复现 | [#245](https://github.com/hf1s/hyper-parallel/pull/245) | Draft，不推荐通用产品合入；忽略部分配置、子类初始化与实际 runtime 校验三项缺口尚未修复 |
| #187 旧环境策略提案 | [#192](https://github.com/hf1s/hyper-parallel/pull/192) | Draft，已由 #232 另一方案处理；原样合入会重新覆盖调用者设置 |

旧 [#137](https://github.com/hf1s/hyper-parallel/pull/137) 是旧基线上的较宽 Draft，不能当作当前默认产品已修复的证据，也不把它与 #245 重复计算为已完成产品修复。

## 不应为了数量补产品 PR 的项

- #149 的分组、scale 和 dtype 已拆开归类；旧 Q-up scale 行为在最新 MF PyNative 已按 piece 自身 shape 实现，无需在当前 HP 默认中复制旧行为。
- #150 独立 packed 文档应按文档内 shift；当前 HP 已采用此语义，无需向旧 MF 跨文档目标回退。最新 MF 的普通 roll 风险仍需结合上层 mask 判断，本任务没有提交 MF 仓库的修复 PR，不能声称 MF 端已经修完。
- #153、#154、#155、#156 的合法 dtype/融合/位置配置差异，不自动构成必须改默认的缺陷。没有凭“更像旧 MF”创造新产品修改。
- #190、#217 是总报告/历史差异记录，按子问题关联代码，不各自重复提交一个大修复 PR。

## 源码覆盖证据

对实际 200 步训练冻结提交 `dcfdf9b03cd353f79af840f9c65436eecefc3c87` 相对基线 `a1e6c222eab097c9609d6e744a3c5f98aeed6bdc` 的 **19 个改动文件**，与线上 PR #213、#243、#245 的文件集合逐项比较，集合完全相等，`files_missing_pr=[]`。

另从基线依次应用这三个已提交 commit，得到 `fe16c866cba6f40e5fea432e8147e7446fbf1e7f`，Git tree 与实测提交都为 `05d9494e1d9851af165a3a181d546b3542051ab9`。三份任务工作树没有未提交修改。没有将临时训练补丁遗漏在正式源代码之外。

这个证明只覆盖提交来源和可复现性；它不将 #245 自动变成必要产品修改。离线转换工具在训练树之外，已单列 #247 / PR #166，不能由上述 19 文件证明替代其独立验收。

[逐文件覆盖回执](submission-file-coverage.json) · [线上 PR 状态](submission-coverage-pr-states.json) · [完整正确性复核](AUDIT.md)。代码仍是既有 head，本次没有新训练，也没有将待办或无 CI 结果写成已经通过。
