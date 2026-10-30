# Muon assignment 的单次观察迁移：固定源码依赖审计（待 root 认可）

这里只判断是否存在条件等价关系，不把 restore-only 缓存当作此前 1501 两次执行的现场录像，也不从 raw 值变化选 owner。此文本形成时，owner run 尚无已归档完整输出。全部引用行号均来自固定 commit 的 Git 字节；原稿误用了当前工作树中的 Muon 行号，已在本版更正。

本机从 Git 1a39572fef23819a2d2651b4bd53047a05a4892f 读取下列路径的原始字节，其 SHA 均等于本次 actual source manifest 0f168e2ce3279cb315fc27d60d06693a17a5626a228fbf777146dda9da3e977d 对应项。唯一 scheduler overlay 不改这些路径。

| 路径 | SHA-256 |
| --- | --- |
| hyper_parallel/core/optimizer/optimizer.py | 9e24e38e602e929ef8cc3a214710602452aa99abb2f94980a26afa29d7bf63c6 |
| hyper_parallel/core/optimizer/muon.py | 27e81dc93bcbbc4c3a288977ff3cab81f8e27c4f537d466d41c2830566aa4f10 |
| hyper_parallel/core/optimizer/sharding_category.py | d418c2dbdb18dbaef9197342aba40430dbd90e6a352d5c08277ca9f9cf0a32ae |
| hyper_parallel/core/optimizer/muon_shard.py | b33e6ce958e8633c148e0e3fc209f4aebafbf4f3786a0e91ffe8b92ce5c079c4 |
| hyper_parallel/core/optimizer/utils.py | 0772696eadaf03c5a9c6de1df0d62222a3980b33b5577bed22be265267ce8157 |
| hyper_parallel/components/optim/mixed_precision_optimizer.py | 5dce5f65c1f7a3fa6873c6ee25398d16d4f36b535f29056cd1e535b784e61400 |
| hyper_parallel/core/dtensor/device_mesh.py | d36229a78ca50f193f0afc565da37e85f0cc2db7f477991a96e13177db20cf5d |

两臂原 config 与各自 owner-v2 config 字节完全相同；sync SHA d779f298fe8e335a4bd9cfec081705eb6dbf177a2a9371d14a9a30bdce5fea7b，deferred SHA acf412733e3bed399510709a6aeb9456bc7c3fd6c24ec4993e69adc1891a3481。本机 YAML 递归比较仅有 fsdp_config.requires_grad_sync 的布尔 True/False 一项差异。已回读的原两臂 4 rank target 在 tensors、bindings、groups（含成员顺序）、extra、loader、initial_sampler、mesh、source contract 均逐结构相等；每 rank 的已记录 57 个 model 梯度 trainable、无缺失。训练现场前者实际为 model=main。这些事实不是 owner 现场记录。

## assignment 实际依赖

1. Muon.reset_optimizer_parameters（muon.py 322–334）按 group 顺序分类参数、建 shard metadata，推断 replica count、构建 batch 与广播缓存，再建立 step 用的 assignment；muon_shard.py 129–203 为 shard PG 做 shape all-gather，重建 record 时保留 param 和 index，要求真实 PG 与各 rank 局部 shape 匹配；mixed_precision_optimizer.py 147–160 重绑 main 后再次 reset。原 worker 与 owner-v2 worker 在 restore 和 258-role gate 前的代码一致（除 stop-step 校验）；但此静态对照不证明训练 step 前无人或无动态插件改缓存。
2. group_parameters_for_hsdp（sharding_category.py 225–329）按原 param_groups 的成员顺序枚举，按 DeviceMesh 的 rank_list/shape、placement、规范化 shard 轴及 ndim 所确定的类型分组；group 字典插入顺序来自参数首见顺序。内部取 device_mesh.get_group(dim)，所以各轴 PG 的真实成员与次序是输入。不能仅凭 shape 相同替代 PG。
3. optimizer.py 547–621 对组总 param.numel() 和组内每个 param.numel() 降序稳定排序，按 replicate_broadcast_max_bytes（属性缺省 512 MiB）和生效 replica count 构造 sub-batch；同尺寸 tie 依赖前述首见/成员顺序。不可忽略 runtime 自定义 cap、record index、dtype/layout、绑定或同尺寸组顺序。固定源中未发现其他对该 cap 的赋值，但这不排除外部运行时修改；owner-v2 也没有直接输出该属性。
4. optimizer.py 623–678 从 mesh 取 replica 坐标/尺寸；sharding_category.py 437–489 先过滤 param.requires_grad，按 (-param.numel(), record.index) 稳定排序，再按 (当前负载, 坐标) 贪心分配。此构造的源码未读取 FSDP requires_grad_sync、reduce_grads、梯度数值、数据或随机数。两臂记录的 preclip trainability 和 model=main 强化一致性，但未直接记录历史缓存构建时每个 main 的 requires_grad。
5. optimizer.py 393–435 的自动 replica count 使用当前 get_device_count() 及 replicate PG size；utils.py 54–61 在 NPU 可用时读取 torch.npu.device_count()。因此“都映射物理 4–7”不能单独替代在构建时有相同有效设备数量与 PG 证据。owner-v2 会报告有效 count、PG、实际 cache 和 callback；原历史执行是否能从其 raw preflight/physical 记录约束相同值，需要 root 核实。
6. muon.py 400–465 在普通 step 中，no_comm 本地更新，assignment 对 owned_params 更新 momentum，再 flush 参数广播。自定义 momentum_update_fn、optimizer swap 或外部可变 callback 改变生命周期时须另行证明；新观察中的属性和实际旧运行构造都需复核。

单次 sync restore-only 结果只有在 4 rank owner/target/terminal/audit 齐全、旧两臂及新观察的 source/config/完整初态/组序/layout、真实 rank 与 PG 组、软件/物理卡匹配，并且上述 requires_grad、cap、有效 device count、缓存从恢复至 step 未重置或变更的条件经 root 基于实际工件和固定调用链认可时，才能有条件地推及旧两臂。比较器须把 root 认可与精确工件哈希绑定；缺失条件即 unknown/incomplete。老 raw 值只能用于独立交叉验证，不能反推 owner。

本审计不证明设备环境、PG 成员、cap、构建时 grad 标志或旧 step 的缓存身份已经实测；也不证明 model/slot 全量正确。若这些条件不能约束，应由 root 决定是否需要第二次 restore-only 或别的现场证据，而不是在比较器中放宽 Muon slots。没有对 0.005 训练验收阈值作任何变更。

逐行摘录 [source-excerpts-1a.txt](source-excerpts-1a.txt) 由完整 Git blob 与 0f manifest 的 SHA 和长度核实后生成；[source-excerpts-verification.json](source-excerpts-verification.json) 记录逐文件完整字节校验和引用范围。
