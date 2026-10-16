# 26 份最终说明增补复核

结论：**PASS_FINAL_DESCRIPTION_ADDITIONS**。仅复核 `body-review-v1` 之后新增的外部参考和当前边界，未修改正文或发布。

26 份正文去掉新增两段后，均可**逐字节恢复上一轮已通过的主体**。13 份 PR 的固定模板标题/checklist、所有原证据、红绿数据和因果边界保持不变。最终 body/JSON/MD SHA 已逐份绑定到 [JSON](final-review-v2.json)。

| 增补内容 | 核对结果 |
|---|---|
| #194 / #199 token 分母 | MF per-token sense/平均rank计数与 Megatron scaling=1/全局token总数的差异准确；同时核对SUM/AVG、TP补偿与MTP分母，不宣称跨框架实测 |
| #199 CP布局 | 连续切片与zigzag helper的物理分配区别准确；要求先还原逻辑序列 |
| #195 / #196 重算 | invocation/replay与grad-enabled差异准确；保留JT自身副作用守卫及实际回归证据边界 |
| #200 dW与GTP | 局部GEMM/累积/归约/main_grad阶段和GTP开关条件准确；普通Linear dW159步失败消融未被当作成功 |
| #201 Muon | MF与固定Emerging-Optimizers依赖的momentum、归一化、dtype差异有源码依据；同实现plain对照未升级成外部数学oracle |
| 其他7组 | 按本地所有权、状态、布局、API或统计契约说明，未强套外部同名实现 |
| 当前验收边界 | 544最新矩阵与186历史证据分开；CPU读回不等于NPU恢复或跨机准入；先主LM，达标后汇报询问MTP/aux；CI空结果不算通过 |

独立读回了 **17 个固定版本源文件、19 个直接行锚**，原始字节SHA与最新参考索引全部一致；201配置SHA一致。固定版本为MF官方GitCode `1c9258ca845e21ddd66fde83e262d5635359bd30`、Megatron `3e2807ce5e787f81b7275be2383092fe0c3bfc71`，实际Muon依赖pin为 `b309e2f01cda75dc96a6dc1a2355a7b3b64b5e16`。

复核中发现的两处轻微行号偏差已修正并读回：DDP per-token缩放指向L250，最终token缩放指向L741。正文事实和适用限制没有其他待修项。

本结论仅适用于JSON绑定的最终候选；发布时仍核对远端旧body/head，不能覆盖并发变更。本次未新增训练、CI、精度或资源收益通过结论。
