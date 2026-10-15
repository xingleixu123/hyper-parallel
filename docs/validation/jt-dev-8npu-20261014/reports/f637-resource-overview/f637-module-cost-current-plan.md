# 当前固定源的最小模块重算成本复测计划

本计划尚未执行。固定源为 `18602c4524dc664b81b2bad9a7b624455ce1494f`；目标是定位值得进入 200 步验收的模块策略，先复用已有包装入口，不增加产品 API。当前 FSDP8 关闭／完全／选择重算各 200 步精度均通过；完全重算比选择重算再少 0.482707977 GiB，步时仅差约 0.2491% 的单次观测，不能判定速度优胜。资源总览独立审计为 PASS，EP2／CP2／TP2 仍为精度 FAIL，不能纳入交付推荐。

依据为 [重算灵活性 v2 审查](f637-recompute-flexibility-gap-review-v2.md)。该报告已区分 MindFormers PyNative 固定提交 `5089da308ebcc146d0cbe32180c209c4f1d39d77` 与 Megatron `3914483fd325e053797aff3fa431d4a3141139aa`；本计划不引入它们的算法或静态图配置。旧 `ec2e16b` 模块实验仅用于选择候选，不能替代当前配方实测。

## 最小矩阵：一个匹配基线加三个候选

先各 40 步，共四组，不重跑旧十组矩阵，也不重复已经完成的 full/selective 200 步。顺序为 off、norm、dense MLP、完整 MoE；40 步只用于筛选，不构成用户要求的 200 步验收。

| 新 case 建议名 | activation_checkpoint.mode | JT_RECOMPUTE_TARGETS | 期望命中数 |
|---|---|---|---|
| `r129-f637-module-off-40-v1` | `off` | 不设置 | 0 |
| `r129-f637-module-norm6-40-v1` | `full` | 下列 N6，逗号连接、无空格 | 6 |
| `r129-f637-module-dense1-40-v1` | `full` | `model.layers.0.mlp` | 1 |
| `r129-f637-module-moe2-40-v1` | `full` | `model.layers.1.mlp,mtp.layers.0.transformer_layer.mlp` | 2 |

N6 的固定 FQN：

```text
model.layers.0.input_layernorm
model.layers.0.post_attention_layernorm
model.layers.1.input_layernorm
model.layers.1.post_attention_layernorm
mtp.layers.0.transformer_layer.input_layernorm
mtp.layers.0.transformer_layer.post_attention_layernorm
```

这些路径已与当前基线 [模块 inventory](evidence/candidate.json) 逐个核对。N6 只含三个 decoder 内的 RMSNorm，不含 MTP 外层 `enorm/hnorm`，不能与旧八个 norm 的结果同名混用。dense1 是一个 `DeepseekV32MLP`；moe2 是两个完整 `JTDeepseekV3MoE`，包含路由、routed experts 和 shared experts，避免用 `*.mlp` 将不同计算混为一种成本。

若 moe2 的整步收益值得继续定位，下一轮才单独测 `model.layers.1.mlp.shared_experts` 与 `mtp.layers.0.transformer_layer.mlp.shared_experts`，不与父 MoE 同时包装。首轮不直接包装 `.experts`：实际 [routed 调用](source/hyper_parallel/models/jt_deepseek_v3/modeling_jt_deepseek_v3.py#L545) 是 `forward_expert_major(...)`，绕过普通模块 `forward`，仅替换 `forward` 的 wrapper 会空测。attention／MLA 子投影及 MTP 整块留到有证据需要时再展开。

## 安装点与固定配方

直接复用 [install_target_ablation](../f637-cp2-200-complete/raw/runs/r129-f637-fp32-cp2-200/module_profile.py#L20) 和 [capture_training 入口](../f637-cp2-200-complete/raw/runs/r129-f637-fp32-cp2-200/capture_training.py#L124)：在 `TextTrainer(config)` 前，以精确 FQN 列表替换实验进程内的 `model_builder._apply_activation_checkpointing`。builder 的实际顺序是 [plan/layout → activation → FSDP](source/hyper_parallel/models/_transformers/model_builder.py#L497)，因此保留正常 pre-FSDP 安装位置。候选设 `mode=full` 是为了进入该入口，[off 会跳过调用](source/hyper_parallel/models/_transformers/model_builder.py#L390)；实际策略是“仅对列出的模块做普通 non-reentrant checkpoint”，并非默认三个 decoder 的 full 策略。默认 early_stop 保持不变。

从当前 off-r4 的 [resolved config](../f637-fsdp8-r4-complete/config.resolved.json)／同目录原 YAML 克隆；仅改 train_iters=40、本表目标、save_ckpt=false、restore_from=null 及独立输出路径。保留以下边界：

- `lr_scheduler_iters=1000`，constant LR=0.0001；GBS=8、micro=1、token_weighted、seed42；固定初始权重、数据顺序和 packed 边界。
- FSDP8，TP/CP/EP/PP=1；BF16 参数／FP32 reduce、Muon FP32 NS、FP32 main params、expert FP32 wgrad；HF32 关闭；chunk loss=4096。
- selection.source=default、swap_inputs=false，compile/activation_swap 关闭；原 reshard、预取、通信参数不变。四组统一关闭 checkpoint 保存；最终保存时间不能计为重算收益。

四组都取消 `JT_MODULE_PROFILE`、`JT_EXPERT_PROBES`、`JT_NS_FP32`、`JT_WGRAD_FP32` 等旧 override，保留匹配的输入身份采集。各 case 新建输出目录、独立端口和 manifest，固定源码、镜像、入口及依赖 SHA，核验 editable 实际导入路径。旧 helper 在进程内作全局 monkeypatch 且不恢复，因此各组必须使用新 worker 进程，不能在同一 Python 进程轮换策略。

运行前精确核对目标 manifest 的 FQN 集合为 6/1/2，off 未安装自定义目标；并比较规范化参数布局、FSDP unit、初始权重、输入摘要。现有 helper 只检查“至少一个命中”和“无父子嵌套”，不能代替完整集合检查。FSDP 后用 HP wrapper 类身份的 MRO 核验真实区域，不能依赖已知会漏计动态派生类的旧类名统计；不为修报告静默切换训练源码。先用低成本 CPU 检查新边界反向／重放副状态；短跑前 3 步检查有限 loss/grad 与路由计数。诊断 observer 或检查器有新增时，单独冻结其 SHA。

## 测量与扩展条件

1. 从 raw 8rank 标量逐步取 MAX；四组统一用第 21–40 步时间中位数，以及 `sum(监督token一次计数) / sum(逐步MAX时间)`。candidate40 与新 off40 同窗比较，不与旧 200 步中位数相减。
2. 显存报告本组 1–40 步累计 allocated/reserved MAX，并记录峰值阶段。这不是 200 步峰值：既有 off-r4 的 reserved 首次最高值在第 41 步。模块进入／退出 allocated 差不是保存激活；不同模块单独节省量不可相加。
3. 同时给出节省 GiB、额外秒、有效 token/s、实际区域数与 loss/aux/路由状态。噪声内时间差不算“无限性价比”；如需要速度结论，仅对有希望的候选与 off 做配对重复。
4. 只有成本归因仍不清楚时，再加 off 与最有希望候选各 12 步 events 诊断，与 40 步计时分开。最多筛出一个有价值的策略／组合进入完整 200 步独立对照，按既有严格门禁验还原系数后的 LM/MTP/aux 逐步绝对差 ≤0.005；短诊断不得记为该门禁 PASS。

## 旧工具复用边界

| 工具 | 可复用 | 本轮限制 |
|---|---|---|
| `module_profile.py:20–40` | FQN 匹配、父子嵌套拒绝、checkpoint_wrapper、pre-FSDP 接入 | 新进程、新目录；另外核对完整命中集合 |
| `capture_training.py:124–156` | 目标安装、prepare_step 透传、输入身份与 scalar 记录、可选事件 callback | 关闭其他数学 override，冻结入口与依赖 SHA |
| `module_profile.py:43–114` | 限窗 NPU events、HP replay 状态、always_call 早停记录、正常结束移除 hooks | 只用于额外短诊断；每步同步会扰动计时，inclusive 父子时间不可相加，allocated_delta 不是激活存量 |
| `prepare_module_experiments.py:10–44` | 克隆配置的思路 | 不直接运行：它固定旧配置、七种策略加三种事件 case、旧端口与覆盖式输出 |
| `collect_module_evidence.py:9–36` | 白名单与 SHA manifest 的思路 | 不直接运行：固定旧十组及旧报告/压缩包路径；改用当前 fair collector 收集新 case，不能覆盖旧证据 |

旧 events callback 按 step_begin 的 global_step 选窗，实际记录第 6–10 步。仅分析本轮精确目标 FQN 的 forward/replay，保留 `completed=false` 的 early-stop 记录。`lm_head` 的 chunk 重放来自 [独立 Torch checkpoint](source/hyper_parallel/components/losses/chunked_cross_entropy.py#L297)，HP `is_recomputing()` 不保证标记该路径，不能用旧 head phase 解释其重放；本轮固定 head 数学及分块，不用 head 事件归因成本。

排队沿用现有 fair guard：读取最新全局 release 状态，满足释放后 1200 秒墙钟／monotonic 门槛并取得空闲卡后运行；出现外部占卡按既有规则让卡。保留退出、归属、释放和完整输入摘要，仅在自有 worker 退出、证据复制且 SHA 核验后清理本任务临时文件。本报告没有启动实验。

## 固定输入 SHA256

| 文件 | SHA256 |
|---|---|
| `tools/plot_f637_resource_overview.py` | `7fe5ba2f4755623ab00acdf745da1bd6f29d60f6086ded12e68c4beeabf6f138` |
| `evidence/f637-current-resource-overview.json` | `a34e51647d4546fc5ca630e00c0d7d61d7fcb6ebb965548a8204c1c692f8914c` |
| `evidence/f637-recompute-flexibility-gap-review-v2.md` | `b2fb17e2d3244417439c48f1ee8500a54497771df8f72a7af52549fb3ef52efc` |
| `tools/module_profile.py` | `b01dcf73a23ec967b04b5b5358f2f8cdae028227cb87e28ac465c0bb7e722c9c` |
| `tools/capture_training.py` | `aef5a45ef770da810f07a59a3d97e3363368e7bb7076525d8e95f05ac9c6d717` |
| `tools/prepare_module_experiments.py` | `b7373a8b85972538503dab119321e13d0cfc841210f79f6b0467b79aeac8b05b` |
| `tools/collect_module_evidence.py` | `411b0fb514a45b399dcb3a6b8d713f9665f84642525e057a8911c721f19bd76d` |
| `runs/r129-f637-fp32-fsdp8-0-200-r4/config.resolved.json` | `dfb276d6b985e0d6197888d4480c340611bc2ab42a3d58c89121a1319457f782` |
| `runs/r129-f637-fp32-fsdp8-0-200-r4/integration/inventory/candidate.json` | `c369274e8e697c77445d33556a60dacb94bf3a8dd58320d96084c39e363794bc` |
