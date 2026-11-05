# #1439 增量 0.5 / 0.6：激活策略与 FP32 helper 契约

只读核对 1439 `7f4a3fd7cac6da3c465743b57d832042bcc4ae29`，通过已配置代理取到 master `c108cf94bed1c931f2a49becaf289398481745a8`，#1420 `bab14d2c57b269f3571fd39f55088d1fe80926f0`（open），hf1s#356 `ccbd40db029e27088d34ecfb391979b8a25bdda2`（open）。对应 `*-activation-before.json` 与 `master-*-readonly.py` 为读取快照，不是产品变更。本代理继承当前模型；Terra medium 与 Fast 无可验证的设置接口，不声称已开启。

## 0.5：当前没有可直接注入 packed 激活的公共构造接口

master `components/modules/grouped_experts.py:235-244` 构造只接受 `module`、`module_fqn`、`context`，后两者在内部丢弃；`:272` 固定通过 `self._activation(module, config, gated_linear_unit)` 设置 `self.activation_func`。`:117-147` 将 `module.act_fn` 当作 ungated 激活，或在非 fused GLU 情况下仅用于 `act_fn(gate) * value`；fused SiLU 直接选 `partial(swiglu, dim=-1)` 并忽略 carrier.act_fn。自定义 `_apply_gate` 还被显式拒绝（`:129-136`）。因此现有 `module.act_fn` 不是可注入 packed gate+up、同时表达上限与 dtype 的公共构造接口；把 partial 填入 act_fn 会错用调用契约，不能据此宣称已有解决方案。

在途 #1420 只重命名 lazy export 和模块，不增加 clamp/dtype 参数；#356 同时扩展 `activation_swiglu.py` 的 `swiglu`/`swiglu_from_projections`、GroupedExperts 和 SwiGLUMLP：GroupedExperts 有策略时读取源 `limit` 和 `activation_compute_dtype`，通过公共 `partial(swiglu, dim=-1, clamp_limit=..., compute_dtype=..., use_fused=...)` 消费。冻结组合同样包含该一整套实现。故不能只合 #1420、只检查 swiglu 函数签名、或删除 carrier 来保证策略；#356 尚未合入时不应宣传非空策略受支持。

### 可行选项及取舍

1. **当前受锁定版本约束的检查**：`models/xing_4/adapter/conversion/module_replacement.py:262-286` 将原策略放在 carrier，并在构造完成后比对 `partial.keywords`。它不复制数学，空策略走原 master，配套 #356 时实际透传；缺失依赖则失败。其缺点确实是依赖私有实现细节且可能误拒未来等价闭包；只核关键词也不能证明函数身份及分割轴。它是对**已核实版本**的 fail-closed 过渡，不是通用 capability API。组合测试必须覆盖 base 默认、#356 非空 limit/dtype、以及缺公共策略时失败。当前模型正常非空路径先由 `adapt_mlp` 构造 `Xing4MLP` 并导入 #356 的 `swiglu_from_projections`（`adapter/moe.py:119-153`），裸 master 上可能先报 ImportError，因此报告所谓所有 grouped 路径都必定到达该 ValueError 不严谨。
2. **模型侧显式覆盖或重绑**：技术上可覆写 `Xing4GroupedExperts._activation`（父构造经虚调用）或在 `super().__init__` 后设置 `self.activation_func = partial(public_swiglu, ...)`；仍须依赖 #356 的公共 swiglu 参数及在缺 API 时提前明确失败。这不复制激活数学，但在模型侧重复 #356 的通用策略选择，绕开公共模块未来的处理、检查和优化，并依赖 `_activation` 私有扩展点或直接修改已构建字段。不能说这是现存公共构造注入 API，也不能只因少量代码就忽略兼容、梯度、端点语义和重复维护。
3. **公共显式 API**：给 GroupedExperts 新增 activation callable/策略对象入参、或者在公共层加入明确能力契约，是长期更清晰的架构；需定义 packed 输入维度、dtype/clamp、source gate 的拒绝、默认行为与替换框架签名，并协调 #356 防止同时引入两套机制。需要公共回归和模型交叉验证，当前并非已存在的零改动选项。直接将 #356 的两个公共文件作为整体前置也是已实现的公共复用途径；不是“改一处 GroupedExperts 即可”，因为公共 swiglu 本身也要新增相应参数和数学。

**判定**：接受 reviewer 对内省耦合的可维护性关注；不接受把仍有明确前置、已冻结组合使用且有历史运行证据的适配称为死代码并整块移除。若本轮继续交付依赖 #356 的非空策略，保留已声明版本边界和当前防静默护栏可行；若决定拓展成稳定通用接口，应单独协调公共 PR，不应在本 PR 无验证地照搬一套。

## 0.6：名称是可简化问题，不是行为错误

master `core/multicore/modules/_precision.py:22-53` 的旧 `preserve_fp32_parameters` 使用 `getattr`、对象身份和通用 Tensor 操作，buffer 也可按同一逻辑保护。当前 1439 的 `:22-53` 函数体与 master 一致，仅改名并更新文字；`:56-57` 保留旧名 alias，mega_gate / mega_mhc 原调用继续可用，Xing4 Router 采用新名。新名字准确描述参数和 buffer，旧名别名保留既有入口；额外一个别名本身没有已证实行为回归。

如果评审目标是缩小共享模块 API 差异，**保留旧名并更新 docstring，Xing4 也引用旧名**是功能等价且改动较少的方案；因为原签名与代码不变，不应删除已有外部调用所依赖的旧入口。当前新名+别名也自洽，可读性更好，不能据此断定必须修改或将其作为阻塞。反之只改新名并删除旧别名会破坏仓内两个现有调用；除非同步迁移所有消费者并验证外部兼容，不建议。

两种方案都应由现有外部 FP32 buffer、meta materialization、dtype/device 和共享既有参数测试支撑；名称选择不会改变保护机制，也不能由此推断 checkpoint 恢复路径已经通过。
