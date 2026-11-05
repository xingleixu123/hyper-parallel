# 正式 7dd 源与合入依赖增量核验

以 source-manifest SHA256 `7dd9613c60843e646e15f6bb9879f477c201d63824793859e5386bd8059d8f4f` 的 1513 条文件记录逐条与 `fdce1a9a2285f1ab76f4c73d4e68b620b134eaea` Git tree 比较；对差异再核已知 PR head 和 master `435ccc0287056af621c3f5fda0182e27284967e5` 的 blob。该复合源码不是单个 Git tree；下表证明文件身份或给出代码范围，不以差异存在推定路径实际触发。

| 对 fdce 的差异 | 归属与适用边界 |
|---|---|
| `components/checkpoint/base.py`, `dcp_checkpointer.py` | 精确等于 #1431 `e5978afd` 对应文件；#1431 仍 Draft/open，未合入 master。 |
| `trainer/callbacks/checkpoint_callback.py` | 复合文件，与 #1431 不逐字相同；包含 pending synced optimizer、持久化完成后 drain/cleanup、恢复后 cleanup；另含 HF export、定时保存等其他后续逻辑。单卡没有非 owner rank；四卡 native Muon 保存/恢复会进入相关生命周期，但本次没有独立证明移除此补丁就发生数值失败。建议 #1431 / #1430 列 **四卡状态生命周期与内存条件项、运行中已包含**，勿列为无条件数值 P0。 |
| `components/checkpoint/weight_conversion.py` | 精确等于 #1442 `5fef2cf8`；专家反向转换条件路径，不等同普通续训必须。 |
| `functional/__init__.py`, `activation_swiglu.py`, `modules/grouped_experts.py`, `modules/swiglu_mlp.py` | 首项等于 #1420，后三项精确等于 hf1s#356 `ccbd40db`；本次策略关闭，勿把运行源已包含扩展功能误记成该功能已在主仓合入或条件被触发。 |
| `functional/grouped_matmul.py` | 精确等于 #1362 `6066caa7`，但当前 GroupedExperts 的 `cumsum` 产生 tensor group list，故不能据此认定 list/tuple 修复被触发。 |
| `losses/__init__.py`, `losses/mtp.py`, `modules/__init__.py`, `modules/mtp.py` | 等于比较时 master；来自已合入 #1386，无需再次合入。 |
| `core/shard/_op_dispatch.py`, `core/shard/ops/yaml/torch_element_wise.yaml` | 精确等于 #1470 `60a2dc21`，实际 FSDP 后 guarded constant 初始化需要。 |
| `distributed/activation_checkpoint.py` | 复合文件；含 #1358 的 PTA GMM/attention 注册，同时有其他差异，不是 #1358 blob 逐字相等；SAC 条件策略不由一般重计算使用自动证明。 |
| `trainer/base.py`, `config/trainer.py`, `config/training.py`, `runtime/loss_normalization.py`, `text_trainer.py`, `vlm_trainer.py` | 归一化复合 delta：#1410 `7a55338d` + dz-ssk#7 `3eee09cb`、与本轮基线融合；详见 `normalization-runtime-commit-binding.json`，不能用单一 PR blob 声称全树相同。 |
| `trainer/callbacks/environ_meter_callback.py`, `tqdm_callback.py` | 与 fdce 和所核最新 master 不同，等于多个较旧 PR head 的共同 blob；属于冻结基线差异，不能据此指定独立 PR 或称为本轮活跃新修复。 |
| `models/README.md`, `models/xing_4/README.md`, `models/xing_4/recipes/train.yaml` | 文档/配方文件身份差异，不等于本次正式私有配置或额外产品能力。 |

#1431 修复复用原有清理接口，目标是 checkpoint 导出/恢复后释放非 owner replicated momentum 副本，属于资源及状态生命周期；并非 Muon 更新公式变更。`#1430` 仍 open。运行时保留副本的精确规模及未带修复时本次能否完成，现有验收不构成对照证明。

对比脚本 `compare-fdce-runtime.py`、`compare-master-runtime.py`、`compare-pr-runtime.py` 保留以便复核；完整源 manifest 是权威身份。GitHub 只读来源：[#1431](https://github.com/mindspore-ai/hyper-parallel/pull/1431)、[#1430](https://github.com/mindspore-ai/hyper-parallel/issues/1430)。
