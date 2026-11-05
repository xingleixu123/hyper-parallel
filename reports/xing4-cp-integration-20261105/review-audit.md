# 1439 / 1425 检视与复用审计

审计快照：1439 head `f96dd5b091f75ba4ec827f73e079b3b21e9174db`，master `c75f5f9f1e6d5ae157cbc7911a5beb893aaa2da4`，1425 head `2fde96553707b4a01b2d6a471650d98f2bc19f4c`。1425 的 reviews、review_comments、discussion 快照各 0；1439 快照 18 reviews、26 inline、7 discussion。静态审查，未运行实验或改变源码；整合完成后需复核新提交。请求的 Terra medium/Fast 开关没有可用设置接口，未假称启用。

## 最新六条 inline

- **4475845264（待整改）**：Muon leaf posthook 可覆盖默认 Q/K 归属；当前仍注册 chain hook，依赖 #1455。
- **4475905678（待整改）**：swap wrapper 直接运行 pipeline，绕过被包装 leaf.step；启用 QKClip 应在实际包装路径显式拒绝 swap。
- **4475925241（待整改）**：当前对全部 attention 投影缩放，无 optimizer ownership 校验；需校验两投影均由 Muon 管理并覆盖冻结和重绑。
- **4476138012（需答复）**：MLA 公共扩展点 _expand_key_value 被公共 forward 与 Xing 覆写复用；避免复制 KV 展开，但是否接受公共 API 抽取属设计评审。
- **4476142089（需答复）**：master batch 先 CP 分片，仅提供局部 shift_labels，不提供逐深度跨片 token；当前 MTP 跨边界交换不能直接删除，未来可作为通用 batch 接口。
- **4476144198（需答复）**：Trainer 负责 loss_fn 与 microbatch 权重；现单个 loss_sum/count 协议不能表达 MTP 每深度不同分母，直接移除 CP 归约会改变目标。

### 直接证据与最小修正

- 默认组参将训练二维非 AdamW 名称分配 Muon（`hyper_parallel/components/optim/parameter_groups.py:141`）；Q/K 两个投影默认符合。冻结、参数排除、模块替换或名称变化不能假设仍归属 Muon；`adapter/optim/muon.py:116`–`:128`目前对所有发现的投影缩放。Muon posthook 应定位实际 leaf，验证 Q/K 都归它管理（考虑 model parameter → FP32 main parameter 重绑），对于不支持的情况明确报错。非 swap 下 chain 在 `core/optimizer/optimizer.py:166` 依次执行 leaf，混精在 `components/optim/mixed_precision_optimizer.py:268` 的整个 chain 返回后复制主权重；leaf hook 发生在 copyback 前。但它早于后续 leaf，若 Q/K 归 AdamW，后续更新可能覆盖缩放。不得仅凭 `requires_grad` 当 ownership。
- `core/optimizer/swap_optimizer_base.py:2002`–`:2016` 由 adapter pipeline 更新而不调用被包装 optimizer 的 step，因此原 leaf hook 确实被绕过；在最终 swap 安装处对启用 QKClip 显式拒绝，并测试包装时机。注册 leaf 可去掉 #1455 hook 依赖，保留回调异常与权重同步语义。
- `adapter/conversion/module_replacement.py:111`–`:125` 仅覆写 RoPE / 拼接而复用公共 `_project_latents` 和 `_expand_key_value`；`components/modules/mla_attention.py:244,279` 公共本身也调用后者。以逻辑复用支持抽取，但 reviewer 的公开扩展边界疑问需解释；不由此认定必须保留某种 API。
- master `data/batching/get_batch.py:297`–`:304,479`–`:502` 先切 CP 再分发本片 shift_labels；`adapter/training.py:245`–`:263` 每 MTP 深度继续位移，`adapter/distributed/context_parallel.py:105`–`:123` 补跨片首 token。现有 batch 尚未提供对应逐深度跨片输入；可以设计公共接口，但不能直接删此逻辑。
- master `trainer/base.py:741`–`:762` 执行 loss_fn 后做 token 加权；`components/losses/chunked_cross_entropy.py:363`–`:385` 的协议只表达一个 sum/count。Xing 在 `adapter/training.py:265`–`:281` 为每深度求自己的有效 token 平均，CP wrapper 同时归约 numerator/count。移到 Trainer 需先表达多个深度分母并证明 DP/CP 梯度归一化，不能只调用现有单 loss 接口。

## 历史在线意见逐条对照

- 4462484211 专家权重转换：已收敛至 `conversion/module_replacement.py` 并执行公共 GroupedExperts；入口 guard 位于 :128–:190，保留其类型、forward、hook、激活、别名和 storage 限制。
- 4462497711 重计算：`adapter/registration.py:27`–`:29` 公共 RecomputePolicy；4462525951、4462531999 官方继承和 reference 命名：`adapter/training.py` 继承原官方类，三个官方文件保留原字节。
- 4466118410 复用 MoE：执行使用公共 GroupedExperts，保留路由训练语义与状态；不是声称所有模型逻辑相同。4466123082 删除 MF 专用 runtime。4466124923、4466130247 无 .gitattributes/setup.py 增量；4466230866 不修改共享 mhc_post；4466419147 固定 config 外置。上述原始作者回复对应 4466599382–4466622291。
- discussion 6628078023 的 conversion 集中已实现；6628331972 和 6631926480 的结构与三处数学问题，在 6646605010 有明确答复。由于本次明确保持官方三文件原样，不能按旧报告建议直接编辑；保留训练与融合差异说明，不宣称官方数值逐位等价。
- fix_router 在 `adapter/distributed/expert_parallel.py:34`–`:46` 显式透传，默认 false；固定路由改变真实 token 目的地，不能代替安全零权重 padding。`adapter/training.py:110` 的 mHC transpose 是 Xing destination-first 到公共 source-first 边界，既往非对称前反向证据仅支持该边界，不支持全局翻转公共模块。
- `xing4-review-0854496d-20261105/0854496d-检视复核与整改.md` 对 0854496d 历史报告逐项裁决仍是旧证据索引：MTP #1386、CP #1425、SwiGLU 可选策略 #356 等依赖必须按最终目标 base 再核实；chunkCE 和 TP loss 为分支、MTP FSDP 排除与显式 wrap 不矛盾、Muon row key 不能仅根据默认配置判死代码。对 clamp 关闭、官方文件不动、UT/ST/配方不入 PR、保留并行与前三提交，遵循本轮明确约束。历史 1000 步和五步结果各属报告指定旧 SHA，不能迁移归属到新提交。

## 必要验证

1. 非 swap 真实 Muon leaf hook：QK 超阈值和未超阈值、FP32 主权重 copyback、多步数值 oracle；冻结、AdamW/不属于 Muon、主权重重绑、不同叶顺序和 hook 异常。swap 在实际注入后启用 QKClip 必须明确失败。
2. CP1/CP2：不均匀有效 token、跨片边界、MTP 多深度（含深度超本片长度）、输出及梯度与整序列参考；DP/CP 不重复归约，TP loss parallel 与 chunkCE 分支；保留原 TP/EP/FSDP/recompute 配置。
3. 整合后复核专家 guard 红绿反例、fix_router 默认及显式透传、非对称 mHC 输出和梯度、公共 MLA 接口及编译/导入；按最终 commit 和实际依赖记录。测试和配方保持外置。
