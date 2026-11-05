# GroupedExperts 策略契约及 #1203 可复用性

只读审计：当前 1439 的 Xing4 适配、master `6c8ccfac` 的公共 GroupedExperts，及冻结 #356 组合 `/home/xxl/.cache/codex/xing4-deep-reuse-20261104/after`。本报告未修改产品代码。

## 最小兼容措施

裸 master `GroupedExperts._activation` 在 `components/modules/grouped_experts.py:117-147` 不读取 carrier 的 `limit` / `activation_compute_dtype`，即便模型侧已经验证每个 expert 策略相同也会忽略非空策略。`__init__:272` 把 `_activation` 的返回值保存在 `self.activation_func`，执行时 `:415,419` 实际调用它。冻结 #356 的同名公共实现 `:137-143` 在任意一个策略非空时产生 `partial(swiglu, dim=-1, clamp_limit=limit, compute_dtype=dtype, use_fused=...)`；它使用公共 `activation_swiglu.swiglu`，实际数学、clamp 梯度和 dtype 转换仍由该公共函数实现，不需要复制。

因此在 Xing4GroupedExperts 的 `super().__init__(...)` 后，**仅在至少一项策略非空时**校验实际 `activation_func` 是 `functools.partial`、所绑定函数是当前 GroupedExperts 真正使用的公共 `swiglu`、keyword 中 `dim == -1`、`clamp_limit == carrier.limit`、`compute_dtype == carrier.activation_compute_dtype` 且 `use_fused == bool(getattr(config, "use_fused_swiglu", True))`。不满足则明确报错，提示需要完整 #356 策略支持；两个选项都为 None 时完全保留 master 既有执行路径。验证时应从定义该 GroupedExperts 的同一公共符号解析 `swiglu`，避免不同模块/重复导入造成错误的 identity 判断。

这能识别本轮已经核实的 #356 具体实现，不能证明任意未来版本：等价闭包或其他合法优化实现会被保守拒绝，partial 关键字也不能独自证明内部函数数学没被改坏。长期更稳的是公共 GroupedExperts 明确暴露激活策略/能力契约并为其测输出和梯度；本轮在不改公共接口和不复制计算的约束下，上述检查是可行的最小 fail-closed 选择。不能只检查 `swiglu` 签名，因为安装了扩展 activation API、但仍使用旧 GroupedExperts 时，后者仍可不透传非空选项。构造前可能已经在 `Xing4MLP` 对缺失的扩展函数导入失败，建议给该部分清晰缺依赖错误；不可为了兼容缺失功能而回退默认公式。

建议外部测试：默认策略 + 裸 master；仅 limit、仅 dtype、二者同时 + #356，fused true/false；使用无 #356、仅函数扩展但旧 GroupedExperts、以及伪造错误关键词的负例，均必须 fail closed；真实 #356 与单专家前向/梯度对照，涵盖 clamp 边界；检查参数重用不回归。已有本轮目标是逻辑-only，测试外置。

## #1203 router load 接口

代理只读抓取 PR #1203 快照 [pr1203-readonly.json](pr1203-readonly.json) / [文件差异](pr1203-files-readonly.json)：open，head `73b7169c99743b1d95d4971775e57b296389dff4`。`distributed/expert_parallel/experts.py` 新增 `accumulate_expert_load(module, topk_indices, global_expert_count)`，将全部 top-k bincount 累加到普通 FP32 `_ep_expert_load` 属性；挂接在 `ep_routed_forward` 的 router 之后。`update_expert_bias_from_load` 跨 group all_reduce 后按 sign 更新，**再对 delta 去均值**，并按调用方给定重算次数除计数；Trainer 通过环境变量控制 callback。

该实现**不能直接替换** Xing4 当前 router 计数与更新：Xing 的安全 padding indices 进入 EP 执行但不应计入真实负载；它在 router `moe.py:47-59` 对真实 tokens 计数并以 `is_recomputing()` 排除重算，且非 EP 路径也执行；`moe.py:110-116` 是不做 delta 去均值的 sign 更新。#1203 的采集位置、重算策略、更新数学、跨 rank group 和只支持 EP 的 callback 都不同。可将它作为未来公共路由统计接口的设计参考，但需先明确 real-token mask、非 EP、recompute、并行归约域、bias 算术及 state/dtype，验证完全语义等价后才能复用，不能仅按名称相似替换。
