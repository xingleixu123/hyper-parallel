# 未答线程回复草稿（未发送）

**4475698652**

已在 fdce1a9a 将配置创建合并进 builder.py，保留 create_config 和真实需要的别名冲突、训练配置及 mHC clamp 边界检查；旧 configuration.py 已删除。

**4475844486**

这个函数由 XING4_ADAPTER_SPEC.replacements 引用，用于延迟加载 replacement provider；保留是为了让模型发现阶段不必提前加载相关实现。已用真实 provider 调用验证。

**4475905775**

已在 fdce1a9a 将 Router、MLP 和 MoE 的训练适配合入 training.py，并更新 replacement 和 CP 导入；官方三个 HF 文件保持原始字节。

**4476024276**

当前保留 training.py 命名，训练 Router/MoE 也已集中到该文件；注册架构名不变，减少无实质收益的接口改名。

**4476285672**

已按建议移除中间 Xing4HyperConnection 包装，在 decoder 初始化时保留原 HF HC 并设置 norm epsilon，再由 plan 替换为 fused HC。参数与 norm 对象身份、真实替换签名及部分加载值保护已检查；去掉重复初始化会改变相同 seed 的冷启动随机序列。

**4476404636**

已删除重复的模型类型判断，保留 HC 的 plan 覆盖检查：如果某个 HF HC 没被 fused replacement 覆盖，构建时明确报错，避免训练时静默进入不同实现。

**4476484034**

当前共享 fused mHC 不提供参考实现的 clamp 语义，本轮训练配置明确关闭 clamp；配置非空时继续拒绝，避免静默改变模型数学。未来补齐公共能力后可再扩展。

**4476511406**

pre 部分仍由 plan 替换；post 在官方 decoder 内联，而本轮要求官方三个 HF 文件不改，所以训练 decoder 保留独立 post 边界，并按共享算子的布局转置 mix。

**4476673821**

初始化函数保留 HF guarded nn.init 调用，确保部分加载时不覆盖已加载的 HC 和 router 状态。已用真实 HF safetensors 的缺项场景验证加载值保持。

**4477449370**

非 EP 路径继续保留 HF ModuleList 循环。EP 当前只承诺已验 grouped 路径，检查改为公共 forward_expert_major 能力而非具体类名；公共非 grouped loop 要求另一套 stacked 权重布局与加载契约，本次不新增未经完整验证的开关。

**4479302597**

目前这部分除打包外还验证来源专家、共享权重、hooks/buffer、shape 和激活策略，并提供可逆 checkpoint 映射；矩阵执行仍复用公共 GroupedExperts。现有公共 EP 能力尚不覆盖全部 ModuleList 转换契约，后续可随公共能力成熟再收敛。

**4479308908**

目前 `_validate_expert_weights` 已从第一组 gate weight 的 shape 读取 intermediate_size 和 hidden_size，再对配置及其他专家权重做一致性检查，避免不一致时静默打包。

**4479501581**

已在 fdce1a9a 回退 `_precision.py` 的文档差异，该公共文件相对本 PR 原基线不再有改动；训练侧继续复用既有 helper。

**4479507617**

已在 fdce1a9a 将 Router、MLP 和 MoE 训练适配合并到 training.py，旧 moe.py 已删除；其余 import 与 provider 已同步调整。
