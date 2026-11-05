# Xing4 router 计数和 bias 精度专项审计

对象：#1439 head 1caf044a，按用户要求只读；原始报告 hp-pr-1439-review-20261105/README.md 的 P1-1 和附带 bias。未修改主库、未启动 NPU。

## 结论与反例

P1-1 的核心问题成立：Xing4Router._token_counts 是 persistent float32 buffer，apply_model_init_dtype 在模型构建尾部对整个 model.to(dtype=target_dtype)，因此 model_init_dtype=bfloat16 时变为 BF16；forward 每个 microbatch 将该批 float32 scatter 结果 add_ 到此 buffer；MoEMonitor 对启用的 router 在 optimizer step 后跨 DP/CP（及需要时 TP/SP）求和、更新 bias 并清零。BF16 不能精确累加普遍的整数计数。

应修正报告中“超过 256 增量全部丢失”“GA>1 必然失效”的说法：重复 +1 在 256 处停住只是一个反例；实际代码按批次加一个整数值，是否丢失取决于当前值和增量以及舍入，并非必然。单批两个专家计数 [256,257] 写入 BF16 时两者均可变为 256，真实计数需要一正一负的平衡方向，但看到的统计得到 [0,0]，无需依赖许多 microbatch。更大批量依然会有不同 ULP 和不可表示整数。FP32 计数也只精确到 2^24 附近；用整数算方向可消除此类舍入。

bias 本身也是持久 floating buffer（官方 Xing4_0TopkRouter），会被相同 model.to 转成 BF16；例如 BF16 bias=0.5、更新率 0.0001，反复一次次加此值仍可能保持 0.5，故单独修计数仍不能保证低精度模型的 bias 更新有效。普通 fp32_main_params 只作用于优化器持有的参数，不能保护此非优化器 buffer。官方 _keep_in_fp32_modules_strict 的声明在当前统一 dtype 构建/验证中没有为此 buffer 提供豁免。

## 本轮真实路径与启用条件

- router 的 enable_expert_bias 来自 moe_router_enable_expert_bias；forward 无论该开关是否启用，只要 training 且非重计算便累加，monitor 只更新启用的对象，调用后零计数。关闭时计数持续累计且仍持久化；因此不能直接假设所有 checkpoint 里的 count 均为零。即便启用，step 边界后的正常 checkpoint 通常为零，但中断、失败、特殊保存调用、老产物及启用前状态需按实际 checkpoint 核实。
- 已有四卡实验使用的 template-cp2dp2、template-fsdp4 及对应性能 yaml 明确 model_init_dtype=float32、moe_router_enable_expert_bias=true、moe_router_bias_update_rate=0.0001，且 fp32_main_params=true。因此这些实验覆盖了该开关开启的 FP32 路径，但不是 BF16 buffer 的修复证据。使用真实 MF 其他源配置时需另核实，不由此代推。
- Xing4 adapter/training.py initialize 对 router weight、count、bias 都初始化；deferred HF finalize 对 missing 模块清除初始化标记后会调用 initialize_weights。若一个 router 在同一模块部分 state 已加载而其他 persistent state 缺失，模块初始化可能对已加载值再次赋零；finalizer 的 loaded snapshot 主要检查身份/形状及标记，并非内容快照。是否在某具体 HF 加载状态触发须实测，不能断言“所有加载后必然清零”。DCP 完整恢复经 live 模型 state_dict skeleton 和 strict load_state_dict，与 HF deferred finalization 是不同路径。
- 现有 DSv41 使用 long + nonpersistent + register_rebuildable_buffer，forward 仅启用 bias 时收集，step 后清零；其更新通过 (sum(counts)-E*counts).sign() 整数比较。该设计是数值参考，不代表可在 Xing4 原样切换持久性。

## 最小修复建议及兼容边界

保留 _token_counts 名称、形状及 persistent 属性，改为 int64；scatter 的目标及 source 都用 int64（不能仅改 buffer 而继续以 float add_）；update 采用整数比较 (counts.sum()-counts*num_experts).sign()，而非对 long 调 mean 或先转 float 比较。enable=false 不再增长是可考虑的独立行为变化，但可能影响已有 checkpoint/启用切换语义，应有明确契约和验证，不能悄悄依赖它处理旧状态。

不得机械设置 persistent=False：这会改变模型 state_dict skeleton，并丢弃旧 checkpoint 中可能非零的累积状态。register_rebuildable_buffer 明确只接受 nonpersistent，因此不能用于一个需要 checkpoint 恢复的持久计数器。CPU/Gloo 单进程实测仓库 hyper_parallel DCP + _ModelStrictLoadPlanner 将 float32 旧同名 count [256,257] 加载到 int64 skeleton，结果 [256,257]；只证明简单同形情况，需再用真实 MF/DCP full restore（含 DTensor/FSDP 与优化器、非零计数）核实，旧 BF16 已丢失的信息不可恢复。HF 部分初始化须另外验证已加载 bias 与 count 未被覆盖。

低精度 bias 需独立闭环：现有 _validate_model_state_dtype 要求所有 floating buffers 是 model_init_dtype，不能只把 buffer 强制回 FP32 而不调整构建、验证和 checkpoint 契约。可探索框架支持的 FP32 状态机制并验证其 buffer/DTensor/FSDP 与同名 checkpoint 兼容，或在启用 expert bias 时对实际非 FP32 bias 及时明确 fail-fast；不得将 fp32_main_params 当成修复。若引入参数化 FP32 bias，需审核状态名字、优化器排除、FSDP placement 和从旧 buffer 的恢复，不能只更改注册类型。

建议外部回归覆盖：256/257 混合计数、累积大整数、enabled/disabled、重计算一次计数、每步跨 DP/CP MAX 非必需但 count SUM 后 bias sign 与 reset、BF16 小步长 bias 更新、meta/to_empty、HF partial 与旧版 full DCP 双向恢复、真实四卡 FP32 基线和 BF16 实际路径。
