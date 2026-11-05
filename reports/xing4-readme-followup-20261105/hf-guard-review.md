# Guarded 初始化候选 CPU 验证

在 /home/xxl/.cache/codex/xing4-readme-review-20261105/hf-init-stage 独立复制此前 candidate，再复制主工作树四个候选文件；旧 candidate 未覆盖，未改产品代码，未使用 NPU。

真实 Transformers PreTrainedModel.from_pretrained + safetensors + 真实 Xing4Router/HyperConnection/adapter.initialize（小规模 CPU）对照旧 candidate 与新 stage：

| 情况 | 旧 candidate | 新 stage |
|---|---|---|
| count 缺失；bias=3、weight=7 已加载 | count=0、bias被清0、weight=7 | count=0、bias=3、weight=7 |
| count 和 bias 同缺失 | 两者0 | 两者0 |
| count=19 已提供 | count=19 | count=19 |
| hc_fn 缺失；hc_scale=2、hc_base=4 已加载 | scale被重设0.125，base被清0 | scale=2，base=4 |

seed=909 的随机初始化全部 state tensor 内容 SHA256 逐项一致。运行 hf-guard-regression.py；日志 hf-guard-candidate.log、hf-guard-hf-init-stage.log。该验证是 Transformers 原生加载，不代表 HP 的 ignored-missing 路径已修复，更不代表完整官方模型、FSDP 或 NPU 验证。

既有 CPU 用例以 stage 为导入源且隔离运行：MegaGate module 2 passed，deferred pretrained 7 passed、14 subtests，build options 15 passed；日志 hf-init-test*.log。先前 DSv41 MegaGate replacement 受 CPU venv 缺 Pillow 限制，此轮未计通过。

## 对独立公共 loader 窄改的风险

保留 raw missing 用于初始化、过滤后的 missing 用于 strict/report，方向与 Transformers 顺序一致；不能越过 tensor 级 guarded nn.init 模型契约。本次 Xing4 将直接 .zero_/.fill_ 改为 nn.init，恰好使 sibling tensor 的初始化标志生效，已验证上表。其他模型自定义 Tensor.zero_/copy_ 等不会自动受 guard 保护，不能对其宣称无条件安全。

建议公共 PR 补既有 deferred pretrained 测试：ignored persistent missing 被初始化、已加载同模块 sibling 保持值、非忽略缺键 strict 仍失败、非 strict 与 reporting、tied alias 不重复初始化、meta/to_empty 与 nonpersistent rebuild。明确 DTensor missing 在现有 _validate_missing_initialization_targets 下仍拒绝；此次把被忽略缺失纳入初始化会让原先静默的 DTensor 场景开始报错，应记录契约并测试。测试 composite PreTrainedModel、忽略多个 keys 及正常完整 checkpoint，防止 _is_hf_initialized 标记跨模块破坏。对于不遵循 guarded initializer 的自定义模型，不应无证据自动初始化并承诺不改变已载 sibling。
