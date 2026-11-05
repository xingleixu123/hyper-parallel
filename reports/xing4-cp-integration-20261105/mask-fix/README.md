# 公共 MLA CP 二维 pairwise mask 修复

master-mask-fix.patch 基于上游 master c75f5f9f1e6d5ae157cbc7911a5beb893aaa2da4，仅包含逻辑文件变更。composite-mask-fix.patch 基于本轮 #1439 候选公共 runtime，供隔离的交叉验证；两个完整源码快照分别以 master- 和 composite- 为前缀。不要把 composite 版本用作独立上游 PR 基底。

CP 的 _validate_mask 已限定二维输入必须为全局 [S,S] bool。其后通过 unsqueeze 将其视图变为 [1,1,S,S]，让通用 NPU attention 将它作为 pairwise mask，而其 2D [B,S] padding 契约保持不变。非正方形、packed 与 mask 混用及 query-partition 的显式 mask 禁止仍由原验证处理。四维输入不变。

test_mla_mask_contract.py 为外部 CPU 探针，不纳入提交：对比 2D 与 4D 传给 attention 的 mask、输出和梯度；验证非方阵及 packed 组合继续拒绝。NPU 真实测试另由 regression_npu 执行，并记录参考端 4D、CP端原始 2D 的组合，结果须以日志为准。

## Issue 草稿（未发布）

标题：MLA context parallel 接受的二维 pairwise attention mask 被当作 padding mask

使用 MLA context parallel 并传入文档允许的 boolean [S,S] global mask，且 batch size 不等于 S 时，NPU fusion attention 将该 mask 作为 [B,S] padding mask 扩展，最终形成 [S,1,S,S]，设备算子因 batch 维不符而拒绝。示例：B=2、S=64 时得到 [64,1,64,64]。

期望：CP 将验证通过的 [S,S] 作为共享 query/key pairwise mask 处理；通用 attention 接口的 [B,S] padding 输入仍按原规则处理。

现象由真实 CP2 schedule 探针的非 CP reference 首先触发；CP runtime 对同一已接受 mask 亦会将二维输入原样传给相同 attention backend。旧冻结版本对照和候选修复 NPU 结果请附实际日志，不以静态推断替代运行证据。

## CPU 执行记录

使用 /home/xxl/.cache/codex/hp-mf-master-sixpr-20261102-venv/bin/python，外部 run_cpu_probe.py 在独立进程载入源码变体。composite 变体 2 passed；master runtime 加匹配 master wrapper 2 passed（pytest 给出已导入 anyio 的 assertion rewrite 警告）。这验证 CPU 合同而非 NPU。第一次将 master runtime 混接当前候选 wrapper 时因新增 callback 参数报 TypeError；匹配 master wrapper 后通过，属于测试 harness 版本错配。
