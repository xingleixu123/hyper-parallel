# Standards 轴只读审查

审查日期：2026-10-18。未修改仓库、issue 或 PR，未启动训练。审查使用缓存中的固定 base/head，未将继承代码当作新增缺陷。

| PR | base | head |
|---|---|---|
| 180 | 3d381781589cd49680ab575535c6ccac30c81787 | 5de3cceab094bbed18915d1a9c9231ca220629ca |
| 192 | ad1c55379767d63ece7e084b0623340b37b83d93 | 8e0722b5bfd209f5143d42afce1614f5f5c5b4b7 |
| 213 | 521d3a8a9aa62353327093ab1258ca2c22006e8b | 6f04715a9be134b1e544dcf3c8487bbd8051ffcf |
| 241 | a1e6c222eab097c9609d6e744a3c5f98aeed6bdc | 633ed7d435594bc05b10cc0a3aee9833c53ac927 |
| 243 | a1e6c222eab097c9609d6e744a3c5f98aeed6bdc | 22c2e70969af168d594c5302dbbe200a1b85a173 |
| 245 | a1e6c222eab097c9609d6e744a3c5f98aeed6bdc | 4ccb63a84304b4d968fb483f5d86b4a385be23df |
| 232 | e21302f9e7b47d41b61f5f4987fc34c000f7c978 | d790be978d263b7f5ee31fab3c11937f8b9a5ea0 |

规则源：`.agent/rules/code-style.md`（边界校验、公开接口契约、避免冗余和死逻辑）、`.agent/rules/distributed.md`、`.agent/rules/testing.md`、`.agent/rules/unit-test.md`、`.agent/skills/code-review/review-checklist.md`（API BC、显式状态、依赖初始化、合理抽象）。Fowler smell 仅作为判断性提示。

## 可行动问题

### [P2] PR245：拒绝被归档子类静默忽略的原生优化器选项

文件：`hyper_parallel/models/jt_deepseek_v3/mf_archive/optimizer.py:107–114,128–176,242–244`。

新增构造器将 `**kwargs` 原样交给公共 Muon；Builder 又通过公共 Muon 的 `ADDITIONAL_CONFIG_KEYS` 接受 `momentum_update_fn`、`reshape_fn`、`ns_transform_fn`、`zeropower_fn` 和 `zero_rms_scale_mode`。但归档子类覆盖 `_update_muon_momentum`、`_compute_batched_ns_updates` 后，前四种回调从未被读取；缩放始终直接执行 `sqrt(dimension) * matched_adamw_rms`，也忽略 `zero_rms_scale_mode='use_lr'` 在 RMS=0 时的原生含义。因此配置被成功接受、记录却运行另一算法。固定配方不支持这些选项是合理选择，静默接受它们不是。

建议：在归档构造器及 `load_state_dict` 的参数组恢复处统一校验，拒绝非空回调及不支持的缩放策略；或明确复用并保留其语义。配方固定 LR folding 的覆盖已明确记录，不与本项混为一谈。

分类：可观察的配置契约问题，违反 `code-style.md` 的边界校验及公开 API 契约要求；同时是可能的 Refused Bequest（判断性 smell）。静态确认这两个覆盖方法访问的实例属性分别只有 `state`、`model_config`。

### [P2] PR245：公开 BaseTrainer 子类入口绕过了子类初始化

文件：`hyper_parallel/trainer/text_trainer.py:40–54`，特别是第 50 行。

新增公开参数仅验证 `issubclass`，随后执行 `base_trainer_class.__new__(base_trainer_class)`，完全不调用子类 `__init__`。普通子类若在 `__init__` 定义字段，在 `_setup` 或专用阶段读取，就会在通过类型校验后因属性缺失失败；自定义需要参数的 `__new__` 也没有合适的创建契约。既有默认 BaseTrainer 手动编排阶段的模式不等于新增“任意子类”的初始化契约。

建议：把无副作用的状态初始化分成明确公共/受保护阶段，并让 TextTrainer 调用；或把该 seam 收窄成明确协议/工厂，清楚规定并校验不调用 `__init__`。测试应覆盖带实例状态的真实子类，而非只用 `pass` 子类。

验证：在不导入 Torch、不连接设备的隔离 AST 探针中执行本次实际新增构造方法；`Specialized.__init__` 设置 `initialized_stage_state`、`_setup` 读取它。TextTrainer 路径得到 `AttributeError: 'Specialized' object has no attribute 'initialized_stage_state'`；普通 `Specialized(config)` 的 `_setup()` 返回 `ready`。

分类：新增公开 API 契约缺口；依据 `code-style.md` 的显式实例状态规则，以及 checklist 的 API BC/清晰状态要求。

### [P2，范围校验风险] PR245：安装目录版本不等于实际装载的 CANN

文件：`hyper_parallel/models/jt_deepseek_v3/mf_archive/model.py:228–230`；关联 `legacy_gemm.py:30`。

固定 schedule 工厂读取硬编码 `/usr/local/Ascend/cann/opp/version.info` 来判定受支持 runtime，但实际 ACL 编译库通过 `ct.CDLL('libacl_op_compiler.so')` 交给动态装载器选择，TorchNPU 同样使用进程装载环境。存在并行 CANN 安装或其它安装根时，检查文件与实际执行库可以不同：正确环境被拒绝，或错误环境通过。本包明确依赖特定 CANN 累加行为，该检查会制造未经验证的支持结论。

建议：以目标环境/实际装载库对应的安装根和 runtime 版本核验，并将实际版本写入复现元数据；无法可靠确认时拒绝运行或收窄为用户核验的先决条件，避免把固定路径文件当作实际 runtime 检查。此项是边界验证风险，不宣称已有 200 步实测跑错 CANN。

## 未列为新增问题的事项

- `model.py:236–237` 的全局 JIT/determinism 设置与基线 `adapter/jt_builder.py:124–125` 相同，未归为 PR245 新缺陷。
- `legacy_gemm.head_weight_gradient` 对编译/格式开关确实使用 `try/finally` 恢复；只允许单训练线程的限制已明示。并发使用是支持边界，不作为已发生的竞态缺陷。
- 固定 910B3/形状、主机参与归约、旧 packed MTP 跨文档目标、旧 shape scale 均已被明确描述为显式归档选项；不能仅因接近 MF 判定通用正确性，也不能仅因它们非通用就伪造数学 bug。
- `ArchivedModel.compute_jt_losses`、`ArchivedMoE.route/local_routed_forward` 与原生路径存在大量平行实现，构成长期 Duplicated Code/维护耦合风险。固定调度本身有说明；是否保留约 2309 行产品归档包应由维护者确认支持寿命、边界和回归责任，200 步通过只能证明这一配方的数值复现。
- 新 UT 模块经导入链依赖 TorchNPU，但同基线已有多组 JT UT 依赖同一链，未把全仓已有 CPU collection 局限重复报为本 PR 的新缺陷。

## 其他 PR 范围

PR213 的新增分支只消除单分片 update 打包舍入，多分片策略没有被声称一并解决；PR243 只拆存储融合导致合并的 Q/latent-KV 和 expert gate/up，并明确保留更宽旧 MF schema 为另一个配方；PR241 已明确区分 CLI 诊断与独立 200 步训练因素；PR192 仍是与已合入 PR232 不同的显式环境策略提案，当前 Draft 范围说明没有声称它独立完成全部对齐。上述固定 diff 未发现需追加的高优先级 Standards 新缺陷。

未运行完整测试或训练；以上是固定源码及现有声明的只读审查，具体复跑证据由父任务独立核对。
