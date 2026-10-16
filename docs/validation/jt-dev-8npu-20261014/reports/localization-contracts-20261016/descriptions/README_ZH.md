# 已审阅的中文说明归档

这些正文对应现有13组issue和PR，代码head保持不变。按最终审阅的UTF-8 body字节归档；是否已写入GitHub以总单所关联的实际页面和提交后读回为准。标题未改变，PR模板和checklist保留原文。

| 对象 | 标题 | 正文 |
|---|---|---|
| issue #203 | Trainer 梯度累积按微批有效token归一化，改变整步目标 | [查看](issue-203.md) |
| issue #204 | FSDP 分片初始化后保留完整源Parameter，EP/CP额外占用显存 | [查看](issue-204.md) |
| issue #205 | CP 分块输出损失的支持与多 MTP、TP/SP 回归覆盖 | [查看](issue-205.md) |
| issue #206 | 完整重计算未覆盖 JT 独立 MTP 解码器 | [查看](issue-206.md) |
| issue #207 | JT MoE checkpoint重放重复提交tokens_per_expert计数 | [查看](issue-207.md) |
| issue #208 | 断点续训后累计样本与 token 监控计数重置 | [查看](issue-208.md) |
| issue #209 | 专家权重梯度局部BF16舍入后，FSDP FP32归约无法恢复精度 | [查看](issue-209.md) |
| issue #210 | Muon支持可持久化的NS精度配置并保留BF16默认 | [查看](issue-210.md) |
| issue #211 | 普通非整除 Shard 的重建形状错误及 Partial 归约入口缺少一致检查 | [查看](issue-211.md) |
| issue #218 | [缺陷] CPU 设备索引返回字符串导致 JT 模型构建失败 | [查看](issue-218.md) |
| issue #223 | 【缺陷】并行测试启动器覆盖上层 NPU 选卡，导致用例越界分配设备 | [查看](issue-223.md) |
| issue #225 | Trainer 资源统计遗漏其他 TP/PP rank 的耗时与显存峰值 | [查看](issue-225.md) |
| issue #227 | [缺陷] Trainer 高精度初始化未关闭 NPU HF32，错误属性写入被静默忽略 | [查看](issue-227.md) |
| pr #194 | fix(trainer): 按整步有效 token 数归一化梯度累积 | [查看](pr-194.md) |
| pr #195 | fix(recompute): 完整覆盖独立 MTP 解码器的重计算 | [查看](pr-195.md) |
| pr #196 | fix(jt): 避免重计算重复累计专家路由 token | [查看](pr-196.md) |
| pr #197 | fix(trainer): 从 checkpoint 恢复累计样本和 token 计数 | [查看](pr-197.md) |
| pr #198 | fix(fsdp): 分片初始化后释放完整源参数引用 | [查看](pr-198.md) |
| pr #199 | test(jt): 补齐 CP 分块损失的多 MTP 与四卡回归 | [查看](pr-199.md) |
| pr #200 | fix(jt): 支持专家权重梯度直接累加到 FP32 主梯度 | [查看](pr-200.md) |
| pr #201 | feat(optim): 支持可恢复的 Muon NS 精度配置 | [查看](pr-201.md) |
| pr #214 | fix(dtensor): 在重分布前一致拒绝非整除分片 | [查看](pr-214.md) |
| pr #219 | fix(models): 修复 CPU 模型构建的设备索引 | [查看](pr-219.md) |
| pr #224 | fix(tests): 并行测试继承上层 NPU 可见设备并校验卡数 | [查看](pr-224.md) |
| pr #226 | fix(trainer): 跨所有并行维度归约资源峰值 | [查看](pr-226.md) |
| pr #228 | fix(trainer): 修正 NPU 高精度初始化的 HF32 开关 | [查看](pr-228.md) |
