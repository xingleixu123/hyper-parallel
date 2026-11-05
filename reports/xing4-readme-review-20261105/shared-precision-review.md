# 六文件候选精度与公共测试复核

审查基于本轮工作树和隔离 candidate 的精度 helper/model_builder；未修改产品源码或启动 NPU。

候选最初 create_config 判断 MoE 使用 moe_layer_freq；官方 Xing4_0DecoderLayer 按 layer_idx >= first_k_dense_replace 构造 MoE，未使用该频率。已报告主代理，工作树随后改为层数与阈值比较；此处不把旧问题记作未修复。

精度改动以现有 helper 别名保持原导入，model_builder 以每个 module 的本地 named_buffers 中声明名构造 FP32 buffer 身份集合，并只对声明项要求 FP32；其他浮点 buffer 仍按 target dtype 校验。Xing4 的 correction bias 仍为持久 buffer，integer count 不被 model.to 的浮点转换影响。原 helper 保留现有 MegaGate 参数调用契约。共享跨模块 buffer alias 的转换顺序、meta/to_empty 及设备迁移、HF/DCP 旧 state 恢复仍需针对最终代码验证；仅声明 FP32 不会恢复已由旧 BF16 存储丢失的精度。

隔离 candidate 运行既有测试（拷贝测试文件以避免原目录 conftest 带来的伪 torch_npu 污染；使用候选优先的 PYTHONPATH）：
- MegaGate module：2 passed。
- deferred pretrained：7 passed、14 subtests。
- build options：15 passed。
- DSv41 MegaGate replacement：collection error，当前 CPU venv 无 PIL/Pillow，未执行，不能记通过。
日志保存在 related-cpu/*.log。初次混合收集因原测试目录 torch.npu stub 干扰失败，已经改用独立进程隔离；失败日志保留 related-cpu-tests.log。

现有 MegaGate 两个测试覆盖实例初始化和 meta to_empty，但不单独覆盖参数 BF16 往返及所有 FP32 alias；主代理另行执行的 router probe 负责本轮新增 buffer 行为。不能由本次 CPU 套件得出 FSDP/NPU 或完整 checkpoint 通过。
