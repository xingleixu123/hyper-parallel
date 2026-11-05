# #1439 新检视报告逐项复核（不含 P1-1）

对照 `hp-pr-1439-review-20261105/README.md` 全文及本地 HEAD `1caf044aa35f4e69ade9938039f587281f14a3be`，以仓库 `a9fa47d` / master `c75f5f9` 和既有组合证据区分结论。只读审计，未改源码；本任务不处理 P1-1。遵循本轮逻辑代码-only、测试仓外及官方三文件原样限制。静态行号均对应 1caf 工作树，合入前重核。验证与依赖不能从一个 SHA 外推到另一组合。

| 项 | 处置 | 核实与理由 |
|---|---|---|
| P1-2 MTP/SwiGLU 前置 | **采纳明确依赖；不采纳仅惰性 import 即视作解决** | `training.py:29` 无条件 import MTP，因此 bare base 的构建依赖 #1386；`moe.py:130-132,148-153` 仅开启扩展策略构造 `Xing4MLP` 才导入扩展 SwiGLU。#1420 改名与 #356 扩展语义不是同一能力；顺序及实际提交精确声明。若 MTP 可选并要支持未安装 #1386 的无 MTP 模型，延迟导入是另一个接口变更，必须验证其构造路径，不能把导入推迟当作消除启用 MTP 的依赖。 |
| P1-3 缺公共 UT | **范围边界，采纳可复核证据及发布说明** | 用户明确不将 UT/ST/配方提交产品 PR。外部 CPU/NPU probe、回归日志及 hash 可以支撑 review，但不等于库内自动回归或 CI 绿；公共 swap 的既有 hook 由静默忽略改为构造失败是对外契约，需说明。现有外部三个核心测试覆盖范围以日志为准，后续可独立入库。 |
| P2-1 GroupedExperts 策略 | **采纳缺能力必须 fail-closed；不采纳一律拒绝非空/删除 carrier** | 裸 base `components/modules/grouped_experts.py:117-147` 不读取 carrier，非空 limit/dtype 确会丢弃；`module_replacement.py:151-170,265-270` 一致性校验不能证明下游消费。已冻结组合的 `.../xing4-deep-reuse-20261104/after/hyper_parallel/components/modules/grouped_experts.py:137-143` 确实读取 `limit`/`activation_compute_dtype` 并传给公共 swiglu，该组合已支持同一语义。应成套依赖 #356 的 grouped-experts + activation，缺任何能力要提前明确失败；不根据 1caf 裸 base 给本可工作的 #356 组合统一拒绝，不删保留策略的字段。验收需非空 limit 和 dtype、上下界及梯度与逐专家参考一致，验证不支持或仅改名时明确失败。 |
| P2-2 配置校验 | **采纳最小必填/条件检查；不采纳猜测默认/一律拒绝未知字段** | CPU `create_config({use_cache: false})` 实测报告所列 8 属性全缺失，后续直接引用确可报 AttributeError。`moe.py:38-45,168` 的 use_pad_tokens、aux coeff、bias 开关/更新率只在实际 MoE 路径必要；`training.py:50` mHC epsilon 在构建 mHC 必要，`:119` init factor 在对应初始化路径必要；`:263` mtp_loss_factor 只在 MTP 存在时使用，`:275-277` chunk_loss_size 只在全词表 chunk CE 使用。以训练契约明确必填、按路径验证有限数值/正尺寸/布尔，不为缺失字段编造数值；HF config 的合法扩展字段和拼写检测需兼容上游接口，不宜全局白名单。 |
| P2-3 跨模型 eager mHC | **范围边界：保留现有复用，采纳解释** | `training.py:40-70` 确实调用 Transformers DSV4 的 unbound forward 并通过 property 适配，造成版本耦合；训练 builder `builder.py:43-48` 要求最终 fused mHC，eager 非正式目标运行路径。`sinkhorn_knopps` 迭代顺序相似不证明 epsilon、dtype、输入布局、梯度与 fused 全等；为去耦重写数学须有前反向逐值参考，非本轮必要。准确说明 FP32/epsilon/clamp 偏离及 transformers 兼容范围。 |
| P2-4 MTP 形态 | **范围边界；不采纳无证据直接实现非 mHC** | `training.py:162-174,196-209` 的确要求 MTP 使用 mHC 并生成对应参数；报告已指出 HF 发布权重缺少额外 mHC 参数。此轮目标是恢复 MF 完整训练 checkpoint 和特定训练架构；公开 HF 初始化未获验收，不可凭 HF state dict 推断训练应改成无 mHC，也不能称只 expand→mean 即等价消去内部混合。配置为 False 明确拒绝是避免静默替换语义；描述中应列出 HF 权重限制及 MF 目标依据，若要支持无 mHC 需独立规格/权重初始化/数值测试。 |
| P2-5 checkpoint | **范围边界，缺口真实；不采纳仅恢复 ignore 规则** | `registration.py:94-104` 无 checkpoint 映射 provider，`training.py:182` 清空忽略规则；专家转换不等于 backbone layer40→MTP、shared head 和新增 mHC 的完整映射。既有恢复验证针对完整 MF→HP 训练 checkpoint，不证明 HF 通用导入导出。应明确列出不支持的公开 HF checkpoint 路径；恢复 ignore 只会把未消费键藏掉，不能生成缺失的 mHC 参数或建立完整映射。 |
| P2-6 CPU lazy import | **采纳合理性但非 NPU 数学正确性阻塞** | `module_replacement.py:27` 顶层 import torch_npu；`builder.py:25` 使 CPU-only 导入失败。把 import 移到唯一 NPU kernel 调用内部不会引入 CPU kernel 或后端 fallback，原则上与 NPU-only 产品执行边界不冲突；`components/functional/npu_fusion_attention.py:388` 亦有局部 import。应检查其余传递导入与实际路径，避免宣称单此一处便让完整构建支持 CPU；用户此前禁止的是为了 CPU 测试新增产品后端判断/CPU fallback，不等于禁止纯 import 时序优化。当前具体任务只读，不修改。 |
| P2-7 fusion attention | **范围边界，采纳接口契约与依赖披露；暂不扩大公共抽象** | `module_replacement.py:51-88` 利用 `_prepare_fusion_attention_context` 后直接调用 kernel 获取 maxima；与公共接口存在索引、dropout 和固定参数差异，需限制在当前 causal、无 sparse、dropout=0 的目标场景并明确拒绝不支持的输入。复用公共 context 已减少重复，但把 softmax stats 扩成公共 API 需设计返回值、兼容性和各 kernel 版本，不因行数引入。PTA/CANN 的已验组合与 `softmax_layout=TND` 明确声明，最低版本/能力探测需在真实多个版本证实后再写；不能仅凭单个 NPU 成功推断普适支持。 |

## P3 与附加建议

| 项 | 处置 | 核实与理由 |
|---|---|---|
| P3-1 no_decay | **建议后续** | `optim/muon.py:68` 生成 1D 完整名称，公共 `parameter_groups.py:38-61` 按子串匹配，存在同名包含误伤的潜在风险；需先复核当前实际模型有无碰撞，通用按对象身份/ndim 分组属框架设计，不能凭复杂度推断本轮已出错。 |
| P3-2 hook guard | **建议后续** | `module_replacement.py:141-145` 泛扫命名含 hook 的 mapping，对常见 forward/backward/state hooks 为 fail-closed，但命名启发式非严格契约；报告称全是文档化属性并不准确，多数以 `_` 开头。替换须覆盖 PyTorch 版本及所有影响迁移的 hooks，保留旧语义测试。 |
| P3-3 透传 forward / MTP GC | **可独立简化，非功能缺陷** | `module_replacement.py:272-274` 是纯 super 委派，可在确认装饰器/签名契约后删；`training.py:210-211` 设置属性是否无效需对精确 #1386 版本和 HF GC 扫描一起核查，不能仅以容器自身未读取断言无作用。 |
| P3-4 router 临时分配 | **建议不采纳本轮优化** | `moe.py:35-37` 确会先基类构造再替换权重；是短生命周期构造开销，报告并未测量峰值，绕开父类可能漏初始化状态。无实测必要性及全生命周期证明不重写。 |
| P3-5 EP/CP 绑定 | **已覆盖契约，建议保留** | EP forward 影子绑定作用对象是 MoE，不是打包转换所拒绝的 expert 子模块；CP 的空列表为框架明确允许的就地设置。改为 forward 参数会改变接口和并行接线，应有明确收益/覆盖再重构。 |
| P3-6 FSDP 预取顺序 | **建议后续性能验证** | 自定义 MTP wrap 不自动证明必需显式执行顺序；四卡运行验证现有路径，不能证明预取最优。仅在 profiler 证实问题并保持参数包装/梯度正确时优化。 |
| P3-7 QKClip WORLD | **范围边界，需文档声明** | `optim/muon.py:110-113,123-129` 使用 WORLD all_reduce，要求各 rank 模块数及 head 顺序一致；现有 FSDP/EP/TP/CP 组合不等于 PP/异构拓扑受支持。PP 不在本轮能力声明；后续支持前需明确子组/一致性检查并验证不会死锁，当前不能把 PP 缺失当已支持缺陷。 |
| P3-8 eps/scaling 注释 | **采纳说明** | `training.py:89-92` 以配置覆盖 latent norm epsilon，softmax scale 仅非 None 覆盖；相对官方行为可随配置变化。PR 已解释，应让接口说明清楚，但不改用户指定的语义映射。 |
| P3-9 非 EP padding | **建议后续性能核验** | 非 EP shared experts 计算 padding 后裁切的额外成本具合理性，数值正确不等于已测浪费显著；改动前对自然路由/参数/梯度和目标 TP/EP 比较，不能以未测开销改变路径。 |
| 连续 CP / Trainer 接口 | **已覆盖当前契约，未来模式需显式约束** | `CPBatchSharder` 文档及 `batch_parallel.py:240-247` 使用连续 `torch.chunk`；`context_parallel.py:108-134` 按 CP rank 取右邻片首 token，针对该 sharder 正确，重复移位支持深度超过局部片长。mesh 本身不携带分片模式，凭 mesh 加断言无法验证调用者实际 batch；应在未来引入 zigzag 时通过明确 layout metadata/接口校验并加跨边界测试。外部调用者传非连续 shard 不在已声明范围。Trainer 当前缺多深度分母协议，不宜直接删 shift/loss。 |
| #1441 aux scale | **采纳作为启用相关训练语义的正确性依赖，精确核验组合** | master `trainer/base.py:755-758` 仅绑定 foundation loss；`functional/aux_loss.py:30-34,69-85,103-121` 的 per-context 默认 scale 为 1，若 MTP 分支先反传，可能先用默认值；不把特定调度必然错或旧 100 步对比等同证明正确。#1441 所需精确提交及组合版本在 PR 中列明，提供独立分支与重计算梯度对照。 |
| foundation_tokens 复用 | **范围边界，采纳注明与不等权验证** | 仅使用键名及 foundation token 加权并不能表达不同 microbatch 各 MTP 深度的有效目标数；等长样本不自动证明所有 mask/DP 配置严格等价。现有固定配置与已有回归需限定范围，变长/不等 mask 应独立参考并设计通用多深度协议。 |
| 流程/CLA | **外部状态待查证** | 报告的 not_signed、开放状态和门禁是检视时快照，实时状态需主代理按最终 head 查证；不由本静态审计声称已阻塞或已解决。 |

## 建议最小后续验证

外置测试继续遵循逻辑-only：缺 #356 时非空策略明确失败、组合内非空 limit/dtype 的输出与梯度；条件配置缺项的清晰错误；现有 CP 连续片多深度不等 mask 与参考；#1441 多分支 aux 梯度及重计算；HF 权重缺口和已声明 MF 恢复路径。既有 NPU 真实 probe 与四卡数据仅证明其对应源码和配置；性能和跨设备长期行为不凭代码推断。
