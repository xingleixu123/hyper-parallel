# Xing4 路由计数冷加载复核

基线：#1439 head 7f4a3fd7；新检视第 0.4 节。只读主库、未使用 NPU。可运行 CPU 实证 cold-load-probe.py；该探针是合成模型，不代表官方全模型加载通过。

## 结论

第 0.4 节的核心 cold HF 缺键机制成立：HF build_load_targets 收集所有 persistent 参数及 buffer，_token_counts 在其中；若预训练来源不提供它，最终 _finalize_model_loading(strict=True) 对 missing_keys 报错，位于初始化目标构造前。合成模型用同一 build_load_targets 和 finalizer 实测重现该错误。官方 checkpoint 的原始模型不含此训练专用字段，但完整官方权重还涉及该 PR 已披露的 MTP 命名和新增参数问题；解决本项不等于宣称完整 HF cold start 已打通。

“persistent=False + register_rebuildable_buffer 两行严格更优”不成立。CPU/Gloo 使用仓库真实 DCP 和 _ModelStrictLoadPlanner 测得：旧 checkpoint 持久 count=13，载入 nonpersistent 模型 skeleton 时 count 未请求，非零值丢弃；反向用旧 persistent skeleton 载入新 checkpoint 时报 missing model._token_counts。真实全量/多 rank 行为仍需实测，不能把此合成证据扩展到所有格式，但它足以否定无代价的断言。register_rebuildable_buffer 要求 nonpersistent，并在 materialization 时根据配方重建；它不保存任何旧计数，不能代替有状态 resume。

正常 Trainer 成功 step 的顺序为 optimizer.step → MoEMonitor.on_step_end 全局同步、bias 更新及 zero_ → Trainer.on_step_end checkpoint；正常 epoch/train_end 保存也处于成功 step 边界。因而启用时常规成功步保存为零有代码依据，不必泛泛声称没有清零。当前 disable 分支不累加，新运行常规值也为零；但中途自定义保存、旧版 enable=false 曾持续累加、外部调用和既存非零 checkpoint 不属于此不变量，训练失败路径也不等同正常 checkpoint。已有 perfnew-fsdp4/rank-0/restored-state.json 的两个旧 float32 [64] count SHA256 均为 5341e6b…，等于 256 字节全零；只证明该特定 rank 的 step5 记录。

## 现有恢复证据的范围

MF1500 是转换后的完整状态恢复，既有报告记载 65 模型项、91 优化器槽、61 FP32 主权重别名及调度/样本游标核对；FSDP step5 是原生 DCP checkpoint 恢复后四布局续训并逐 rank 核验。它们验证各自转换/原生 resume 契约，不是官方 safetensors 的 HF strict cold-load 实验，也不反驳 HF 缺键分析。保留同名 persistent state 有助于延续已有 DCP 路径；转换的具体缺键策略及非零恢复仍需独立核验。

## 建议

应作为本 PR 的真实模型冷加载缺口处理，但不要通过直接删除持久状态来掩盖。较小的语义正确方案是：仅在 HF pretrained 冷加载时识别实际缺失的 Xing4Router 计数 FQN，以确定性零值初始化该缺失项，并使严格校验仍对其他缺失项生效；已提供的计数必须保持原值；DCP full resume 和保存 schema 保持不变。实现应挂在合适的已有模型适配/加载扩展点；若公共 finalizer 暂无这种按项初始化接口，可能需要独立公共补丁或协调完整冷加载支持，不应贸然扩大全局 strict 豁免。

尤其不可只添加 _keys_to_ignore_on_load_missing：_adjust_loading_keys 在构造 initialization_targets 之前移除被忽略项；而 meta + to_empty 可能令 persistent buffer 未初始化。现有 register_materialized_state_hook 不能擅自修改 persistent buffer，严格 rebuild 会检查并拒绝持久状态变更。应对每一个被豁免的缺失计数明确初始化与设备/layout，并证明不改已加载 bias/weight；不能无条件调用整 router 初始化来覆盖加载状态。

建议外部验证覆盖：模拟缺 count 且其他参数齐全的 strict HF 冷加载、官方实际 checkpoint 的其余独立缺键清单、meta materialization 后 count 确定为0且 bias不被改、已有非零 count 的 HF 恢复、旧/新完整 DCP（真实多 rank、DTensor、优化器与采样游标），以及成功步 checkpoint count 的记录。当前结果无需重跑已完成的 MF1500/FSDP step5，仅因代码方案将改变才重新验证受影响路径。
