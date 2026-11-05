# 公共 MLA CP2 schedule mask 失败分析

失败位置在非 CP reference 的 NPU fusion attention。tests/torch/context_parallel/_test_mla_cp.py 的 _suite_cases 构造 bool [64,64] pairwise allowed mask，batch=2，causal=False；attention_options 只补充 actual_seq_len/is_causal，原 mask 被透传。

公共 MLA CP 的 _validate_mask（mla_context_parallel.py）明确把 [S,S] 视为全局 query/key mask 接受，同时也接受 [B|1,1,S,S]。但 components/functional/npu_fusion_attention.py 的 _prepare_attention_inputs 将任何二维 mask 解释为 [B,S] padding，并扩展成 [64,1,64,64]；这与真实输入 B=2 冲突，kernel 因而拒绝。该失败发生在原 reference，尚未验证对应 CP 结果；不过 CP runtime 自身也把经校验的 2D mask 原样传给相同 backend，所以公共 MLA CP 宣称的二维 pairwise mask 契约存在同一潜在真实缺陷，而不只是 fixture 表面格式问题。此故障与本次 callback 移植无关的判断须以旧冻结对照实测进一步确认。

诊断可在外部测试副本将该 pairwise mask 改成语义等价的 [1,1,64,64] 再运行，检查其余 CP schedule；必须单列诊断结果，不能把原 [S,S] 测试标成通过。完整修复需明确通用 2D padding 与 pairwise mask 的消歧约定，或者在 MLA CP 与 reference 对应边界显式升维；应另行加入代表性 mask 正确性验证，不应盲目改变全部二维 mask 解释。Xing4 训练入口拒绝外部 mask，不能由此断言其隐式因果路径失败。

当前未启动 NPU、未更改仓库，待回归代理保留旧版对照和真实日志。
