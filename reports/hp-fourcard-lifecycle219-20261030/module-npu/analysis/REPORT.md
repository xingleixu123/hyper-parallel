# 独立直接导数复核

本报告由 `y=xW`、`L=sum(y)` 直接得到 `∂L/∂Wᵢⱼ=xᵢ`，使用 Python `Fraction` 和实际人工输入逐行计算 loss 与梯度，再把两 rank 梯度显式求和并按 rank 取连续两行 shard。没有调用 worker 内置的 CPU autograd oracle；这是从录制的合成模块结果得出的结论。

| 用例 | 独立计算的全局行向量（每行重复 4 列） |
|---|---|
| 两 micro 均参加 | `[6, 6, 8, 8]` |
| early 只在 micro0 | `[3, 2, 3, 4]` |
| 双窗口 window0 common / early | `[9, 12, 15, 12]` / `[3, 2, 3, 4]` |
| 双窗口 window1 common / early | `[9, 18, 21, 12]` / `[3, 4, 5, 4]` |

例如 fusion=false、early 只在 micro0 且首 micro no-sync：[rank0](../output/rank-0-fusion-false-micro0_only_false.json) / [rank1](../output/rank-1-fusion-false-micro0_only_false.json) 的首 micro early pending 首列为 `[1,1,1,2]` / `[2,1,2,2]`；末 micro 同步后的 early main_grad 首列为 `[3,2]` / `[3,4]`。

原始独立结果 [independent-math-original.json](independent-math-original.json) SHA256 `f4b1f3707128261c806b3edbffbd00e8ca077f6e0060aee1d2217976e1b3efbf`，列出当次 40 个输出各自的 size/SHA、72 条 backward-phase、4 条 pre-zero 边界和 2 个 rank terminal。原完整档案还已按闭合报告重验107项与 source pre/post；公开包不包含全部107项，不能用公开复算冒充重新验证全档案。

公开适配脚本 [independent_math_public.py](independent_math_public.py) 的数学循环与原 [independent_math-original.py](independent_math-original.py) 字节相同，适配仅更换输入路径、从原独立报告验证40条已封存的 size/SHA、将缺省输出改为指定的新文件，并把107项/源码前后一致标为原复核的历史结论。CPU 复算生成的 [fraction-recomputed.json](../derived/fraction-recomputed.json) 与原报告的数学、状态、40文件和身份共有字段全部相等；`closed_107_rehashed_in_this_portable_run` 为 false。原审计器 CPU 复算 [auditor-recomputed.json](../derived/auditor-recomputed.json) 与原审计器结果逐字节一致。

这里只观察结果状态，callback 实际执行次数未观测；它不覆盖重计算、HSDP partial-only、整网或完整 checkpoint。物理设备映射可与保留的短 SMI、release 文件比对，完整原始安全/进程观测未公开。
