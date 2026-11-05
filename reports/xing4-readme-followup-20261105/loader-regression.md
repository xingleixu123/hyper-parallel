# 公共 checkpoint loader：独立 CPU 回归

对照 master `c108cf94bed1c931f2a49becaf289398481745a8` 的原 loader 与候选文件，使用该 commit 的 `hyper_parallel/` 和 `tests/` 建立独立 CPU stage，仅在 stage 内替换候选 loader，不改主仓。Python 3.12、Torch 2.12.0+cpu、Transformers 5.22.1。源码及脚本哈希见 [manifest](loader-regression-manifest.json)；各日志包含退出码。

用真实 `transformers.PreTrainedModel`、guarded `nn.init.constant_` 与 `safetensors` 文件，通过实际 `load_pretrained_weights` 分别调用 legacy 和 DCP 单进程 loader。模型在 meta 构造后 `to_empty(cpu)`；将缺失位置预置为 71 以稳定观察未初始化存储，nonpersistent buffer 经 `register_rebuildable_buffer` / `rebuild_materialized_state` 显式重建为 17。持久 parameter、持久 buffer 都设置 missing ignore，同一子模块另含完整加载的 parameter 和 buffer。

| 场景 | 原 loader | 候选 loader | 两种加载后端 |
|---|---|---|---|
| 被忽略的 parameter / buffer 缺失 | 保持 71 / 71，旧路径红 | 正确初始化 3 / 5 | legacy、DCP 均复现 |
| 同子模块已加载 parameter / buffer | 分别保持 7 / 8 | 分别保持 7 / 8 | 均通过 |
| nonpersistent buffer | rebuild 后保持 17 | rebuild 后保持 17 | 均通过 |
| 所有持久项均在文件中 | 两个相关值保持 9 / 10 | 两个相关值保持 9 / 10 | 均通过 |
| 非忽略的 parameter 缺失且 strict=True | 明确报错 | 明确报错 | 均通过 |

[原版红例输出](loader-regression-before.log) 与 [候选绿例输出](loader-regression-after.log) 的 `exit_code=0` 代表探针断言符合各自预期；原版红例并非“功能通过”。[既有回归](loader-regression-existing.log)：`test_huggingface_load_planner.py`、`test_hf_external_targets.py`、`test_deferred_pretrained.py` 共 **16 passed、18 subtests passed**，6 条 pytest mark 警告（独立 stage 未带原项目 pytest 标记注册文件），exit 0。

候选在 ignore 过滤前保存 missing keys，用于 persistent 初始化；报告和 strict 检查仍采用过滤后集合。测试表明它对遵循 HF guarded initializer 契约的模型解决该缺口；测试不支持声称对模型 `_init_weights` 直接使用 `zero_` / `fill_` 等绕开 HF guard 的自定义写入自动保护已加载兄弟张量。Xing 对该类自定义初始化另行调整为 `nn.init`。本测试未验证多进程 DCP、DTensor、NPU 或大模型内存行为，也不对产品初始化引入全量权重快照。

## 初始化集合边界补测

[补测脚本](loader-boundary-probe.py) 和 [真实输出](loader-boundary.log) 使用候选 SHA `8f623e58927b1923fecf92330fc3a0bc414e1a5e8d37d9111051b18a8f463385`、CPU Gloo 单进程、真实 HP `DTensor.from_local` 与实际 safetensors 的 legacy / DCP 加载路径，退出码 0。

- tied alias：实际 checkpoint 只有 `src.weight`，`dst.weight` 为同一参数、声明在 `all_tied_weights_keys`。两个后端均将缺键 alias 正确 resolve，报告缺失为空，参数值保持 7，模型初始化调用次数为 0；未重复初始化共享数据。
- ignored DTensor：真实 HP DTensor 参数 `ignored` 被 missing-ignore 规则匹配，两个后端均在初始化前报 `Missing distributed parameters require DTensor-aware initialization; first keys: ignored`；另一已加载参数值保持 8、初始化调用次数为 0。这是候选相对原版静默跳过的有意 fail-closed 行为，并非声称已有 DTensor 初始化支持。

上次 6 条 pytest 警告经原日志确认**均为 `PytestUnknownMarkWarning`**，分别对应 `cpu_linux`、`cpu_windows`、`cpu_macos`、`level0`、`allcards`、`essential`；独立 stage 仅归档了源码和测试，未带项目 pytest 配置，属于标记注册/收集环境噪声，不需要修改产品。

补测仅覆盖 rank 1 的 CPU Gloo；不推广到多 rank、NPU、不同 alias 布局或 DTensor 分布式初始化。
