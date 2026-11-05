# 被忽略缺失项的初始化与已加载同模块状态

## 结论

值得单独提出公共 loader 问题并考虑独立 PR：当前 HP _finalize_model_loading 先通过 _adjust_loading_keys 从 missing_keys 去掉匹配 _keys_to_ignore_on_load_missing 的项，随后才由过滤后的集合构造 initialization_targets；这些被忽略的 persistent tensor 因而不会由该路径初始化。meta → to_empty 之后相应存储可能是任意值。忽略只应决定严格检查/报告，不应隐含“已经有有效值”。

但公共修复不能只把原始 missing 集合用于 initialization_targets：若调用模型的 module 级初始化，已加载的同模块 tensor 可能被用户实现的原位操作覆盖。Xing4 adapter.initialize 正存在该问题，需要模型专属修复或更强的保护机制。

## 真实运行证据

本目录 hf-init-real-probe.py 使用当前已安装的 Transformers PreTrainedModel、真实 Xing4Router、真实 adapter.training.initialize、HP _finalize_model_loading；以加载 weight=7、bias=3、缺失 count、内存 sentinel count=123 构造等效加载报告：

1. HP strict 且忽略 missing count：成功返回，count 保持 [123,123,123,123]，bias 保持 3；说明没有运行 count 初始化。
2. 去掉 ignore，HP strict=False 走现有 missing 初始化：count 变零，但先前已加载的 bias 从3变0；weight 仍为7。
3. 该模型还用真实 save_pretrained/safetensors 写入临时目录、删 count，并调用真实 Transformers from_pretrained：missing 返回空集合（被 ignore 规则过滤），count 初始化为零，但 bias 从3变0。实验为小模型实际框架路径，不是完整官方 Xing4 checkpoint。临时目录由 TemporaryDirectory 自动清理。

Transformers 当前 modeling_utils 的加载顺序是先 _move_missing_keys_from_meta_to_device 与 _initialize_missing_keys，再 _adjust_missing_and_unexpected_keys；因此 ignored missing 的普通语义是静音报告而非免初始化。

Xing4 initialize 的 nn.init.normal_(module.weight,...) 被 HF initialize_weights 的 guard_torch_init_functions 按 tensor._is_hf_initialized 保护，所以 weight 保持；对 module.e_score_correction_bias.zero_() 和 module._token_counts.zero_() 的直接 Tensor 方法不经 nn.init guard，因此 sibling bias 被清零。HF 自己的 from_pretrained 同样不能自动保护这类自定义直接写入。这不改变公共 HP 缺初始化问题的独立性，也不表示全部自定义模型都受影响。

## 安全处理

公共 PR 先用真实 Transformers 模型和 ignored persistent buffers 建立回归，分开保存 raw_missing 与 reporting_missing；对实际缺失 tensor 做初始化，但必须明确安全契约，不可假定所有 _init_weights 采用受保护的 nn.init。应设计按目标初始化/显式模型补全的接口，或在缺失项引起模块初始化时保护同模块已加载 tensors 的数值并正确处理 DTensor、别名和大参数内存；仅校验对象身份、shape 和 _is_hf_initialized 无法检测原地覆盖。并覆盖 nonpersistent 通过 rebuildable 配方、strict 其他缺键仍报错、已加载值不变、meta/to_empty 和完整 DCP restore。

Xing4 单独修正 adapter.initialize，使其对每个 tensor 尊重 _is_hf_initialized 标志（或通过受守护的 transformers/torch nn.init 函数），并验证 missing count 为零、已加载 bias 与 weight 原值不变；按实际 meta/DTensor 与初始化流程测试。这样模型侧是必要保护，与公共 loader 维护报告过滤和初始化目标分离互补。避免把共享行为变更直接塞入 #1439；公共 PR 的设计验证完成前，#1439 应明确官方 HF cold start 当前不受支持或采用局部可验证的定点补全方案，不能声称两行 nonpersistent 无条件更优。
