# Router bias FP32 协议及 FSDP 审计

审查对象：#1439 head 1caf044a 的 Xing4 路由；代码只读。没有启动 NPU。

## Parameter 方案

仓库 core/multicore/modules/_precision.py 的 preserve_fp32_parameters 与 MegaGate 的 fp32_parameter_names / _apply 配套；model_builder._model_fp32_parameter_ids 和 _validate_model_state_dtype 对这些局部参数有 FP32 豁免。独立 CPU 最小模型证实：将同名旧 buffer state strict load 到 requires_grad=False Parameter，构建转 BF16 后权重 BF16、bias 保持 FP32，0.0001 更新保留，Muon/AdamW 参数筛选排除该 frozen Parameter。脚本 fp32-router-probe.py 位于本目录；仅证明单进程协议，不证明整个模型。

但更改 buffer 为 Parameter 改变运行时所有权：它进入参数分类及 fully_shard 管理；默认可以按 dim0 分片，router 的 optimizer step 后直接 bias.add_(整专家 direction) 没有 DSv41 update_modality_bias 对 DTensor 的布局适配。保存时使用 ignore_frozen_params 的路径也可能滤掉该参数，而原 buffer 不在此过滤集合。普通 Trainer full DCP 的 _model_state_dict 使用 model.state_dict，默认键相同，但同名不保证分片、layout 和 optimizer/checkpoint 生命周期兼容。

FSDP 混合精度是另一约束：HSDPParam.init_dtype_attrs 按 custom_params 或全局 param_dtype 设置 unshard dtype，all_gather_inputs 可能将持久 FP32 shard 暂时转 BF16；_apply 保持存储 FP32 不足以保证 router forward 的 bias 数值仍是 FP32。若选该方案，需明确指定参数 replicated/ignored 与 per-parameter FP32 policy 或正确处理 DTensor 和 forward 精度，并以实际 TP/FSDP、full restore 测试证明；不能仅凭 MegaGate 可类比认定安全。

## 保留 buffer 的扩展方案

按主代理的新方向，更稳妥的是保留原 e_score_correction_bias 的 persistent buffer 身份、名称和加载契约。在 model_builder 中增加与 fp32_parameter_names 平行的局部 fp32_buffer_names 身份集合及 dtype 验证，Xing4Router 的 _apply 用通用化 preserve_fp32_tensors 保留该 buffer；保留旧 preserve_fp32_parameters 名称和原调用语义，以免影响 MegaGate 和 MegaMHC。_token_counts 独立改 int64、保持 persistent、保持同名。

静态追踪 HP fully_shard：HSDPState._move_states_to_device 对模块 buffers 仅调用 buffer.to(device)，不指定 dtype；param_dtype/自定义精度配置用于 HSDPParam 管理的参数及其 all-gather，未见对普通模块 buffer 按混合精度统一 cast。FSDP 对输入或输出的 cast 不等价于对模块 buffer cast。由此该方向避免把 bias 加入 FSDP 参数分片生命周期，但这只是仓库代码静态结论，仍需设备实测。

实现注意：声明必须按每个模块的本地 buffer 名称解析且支持 alias，避免误豁免同名别的模块；验证需检查声明目标确为存在的 FP32 buffer，非声明浮点 buffer 仍服从 target dtype；保持 integer buffer 行为。helper 原理是先执行 transform，再对受保护 tensor 沿目标 device 使用原始 FP32 内容，故必须覆盖 meta/to_empty、CPU/NPU 迁移、DTensor layout、共享别名和转换时对象身份；不得先把 BF16 舍入结果重新转 FP32。若源 buffer 已为 BF16，重新 cast FP32 无法恢复已丢信息。原 HF 加载和 DCP restore 后分别验证 bias 值、dtype、state_dict key 与实际选择路由和小步长更新，尤其加载带非零 bias 的旧 checkpoint。

建议测试：未包 FSDP 的 dtype/重入转换、device/meta materialization 和 shared state；BF16/FP32 model_init_dtype；FSDP param_dtype=BF16 的 CP/DP/TP forward 和 step 后实际 buffer dtype/值；旧/新完整 DCP 同名恢复以及 HF 加载；原 MegaGate/MegaMHC 的 parameter 协议回归。训练中断在非 step 边界时 count 恢复与禁用状态另按既有 persistent 契约验证。
