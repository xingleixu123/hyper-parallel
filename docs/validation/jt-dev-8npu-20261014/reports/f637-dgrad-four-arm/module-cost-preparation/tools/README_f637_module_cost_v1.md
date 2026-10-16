# 模块成本四组实验：准备稿，框架与设备验证尚未执行

仅新增本任务文件。生产源固定为 `18602c4524dc664b81b2bad9a7b624455ce1494f`，镜像固定为 `sha256:5f9e677f439d11ab4e7c888f54b142335a27eee7411cbd2ce894a4798a70146d`。没有启动容器、CPU 合约检查或 NPU 训练，也没有修改现有 helper、fair guard、C/D 请求或公共报告。

## 文件与作用域

- `f637_module_cost_spec.py`：唯一四组 case、端口和 FQN；配置允许差异及冻结依赖检查，不导入框架。
- `f637_module_cost_experiment.py`：复用既有 `module_profile.install_target_ablation`，在 builder 的正常 pre-FSDP 阶段安装；先拒绝漏命中、重复、别名和父子嵌套，再检查 0/6/1/2 的精确集合。
- `capture_f637_module_cost_training.py`：调用原 `capture_training.main()`，保留初始状态、输入身份、prepare_step 和标量采集；使用原 `install_ns_batch_trace` 保存相同 NS 元数据，不替换数学实现。
- `preflight_f637_module_cost.py`：真实 `parse_training_args` 与 JT config 构造，逐项对照基线 resolved config，核验 editable、源码、patch 与全部依赖 SHA，不构造训练模型。
- `check_f637_module_cost_cpu.py`：待执行的真实 CPU checkpoint 合约测试。
- `run_fair_f637_module_cost.sh`：单 case 八卡入口，840 秒内部 timeout，外层仍需既有 fair guard 的 900 秒/20 分钟让卡约束。
- `run_f637_module_cost_cpu.sh`：仅供已启动的无设备环境调用，不创建容器、不改 editable。
- `../configs/r129-f637-module-*-40-v1.yaml`：由当前 off-r4 原 YAML 克隆的独立配置。
- `../scheduling/f637-module-cost-v1.manifest.json`：源码/tree/image pin、完整外部 helper/config/检查器依赖 SHA 和未执行状态。
- `../scheduling/f637-module-cost-local-checks-v1.json`：本机静态语法、仓库 pylint、YAML 类型与允许差异、固定目标数的结果；不是框架或设备验证。
- `../scheduling/r129-f637-module-*-40-v1.request.draft.json`：嵌套 planning-only 草稿，不能直接作为现有 guard 请求；没有进入队列。

| arm | 端口 | HP 模块 wrapper 数 | 精确目标 |
|---|---:|---:|---|
| off | 30371 | 0 | 无 |
| norm6 | 30372 | 6 | 两个主干 decoder 与 MTP decoder 各 input_layernorm、post_attention_layernorm |
| dense1 | 30373 | 1 | `model.layers.0.mlp` |
| moe2 | 30374 | 2 | `model.layers.1.mlp`、`mtp.layers.0.transformer_layer.mlp` |

全部 FQN 在 spec 中逐字列出；没有通配符。off 依然保留当前 `enable_chunk_loss=true` 的独立 Torch head/CE checkpoint。moe2 包含整个 MoE，不直接包装绕过普通 forward 的 `.experts.forward_expert_major`。

候选设 `activation_checkpoint.mode=full` 只是为了进入自定义精确区域入口，不冒称默认整层 full。其余差异只有 train_iters=40、禁用保存及独立 checkpoint 输出路径；固定 constant LR、GBS=8、micro=1、FSDP8、TP/CP/EP/PP1、chunk4096、BF16 参数/FP32 reduce、FP32 NS/main params/expert wgrad 等。所有旧数学 override 和 module-profile/head/expert 高频观察均显式关闭。

## 覆盖与生命周期

构造后按 `isinstance(CheckpointWrapper)` 和真实类 MRO 识别 FSDP 动态派生 wrapper，精确集合不符即失败。每 rank 保存 coverage/runtime 文件。计数在所有四组采用相同九个观察边界，只记录前 3 步 forward/replay 与完整/早停次数，不读 Tensor 值、存激活、加 NPU events 或同步。单区域单 phase 每步上限 64，记录仅三行。

TextTrainer 的 step_begin 看到已完成步数，因此 observer 用 `global_step+1`；step_end 使用已完成的新步数。第 3 步删除所有自有 hooks，替换 callback 列表移除自身，保持当前循环后续 EvidenceCallback 仍执行；第 4 步起没有该计数 callback。原 NS 元数据记录与基线输入采集继续保留，四组一致；资源结果仍不是无插桩性能。

Builder patch 仅存在于构造 context；成功、失败均恢复，所有权被外部改动则拒绝覆盖对方。worker finally 清理自有 hook 与 capture/NS 临时 patch。每组独立进程与输出目录；文件排他创建，既有 helper 的覆盖式 manifest 输出也先检查不存在。EXIT_CODE 通过 EXIT trap 保存，含 preflight 失败路径。

## 本机已检查与尚未执行的边界

1. 本机已通过 5 个 Python 文件静态语法、2 个 Bash 入口 `bash -n` 和仓库配置 pylint（结果 `[]`，exit 0）。PyYAML 6.0.3 对四份草稿的 `rms_norm_eps: 1.0e-06`、`ns_epsilon: 1.0e-07`、`adamw_eps: 1.0e-08` 均解析为 float；草稿与基线 raw YAML 仅有计划差异。静态目标数为 0/6/1/2，总共九个互不重叠观察边界。使用当前 Torch 顶层导入规范，四个涉及框架的文件仅禁用冻结插件已过时的 Platform 导入规则；使用 TestCase.enterContext 管理测试目录的一行单独说明资源清理，未关闭其他新增诊断。真实框架验证和运行状态仍为 NOT_RUN。
2. 在无设备、正确 editable 的现有验证镜像中执行真实 parser preflight：四组 resolved 配方必须逐项等于基线加允许差异；初始化加速器即失败。
3. CPU 7 个测试方法：四组 × aux/total 的真实前后 loss、全部参数梯度、路由累计计数；前三步计数与第四步删除 callback；off 的 head 反向仍重放；无 aux 图残留。每轮保存反向前计数快照，要求反向后逐专家值不变、全局计数等于四轮内累计真实 token×top-k，且计数没有 grad_fn；最终消费累计计数后，bias 更新与参考相同并清空计数。覆盖整段 MoE 的持久副作用，而不仅比较 loss。
4. CPU 负例：空/通配符/重复/嵌套/缺失/模块别名、预先存在 wrapper、额外或漏掉 wrapper、跳过安装、未知 arm、安装失败恢复及外部 patch 所有权冲突、部分 hook 安装回滚。
5. 使用真实 `_extend_module_with_hsdp_interface` 做 CPU 类扩展身份检查，无 FSDP 状态/进程组。CPU 的 NPU-only GMM 使用可微替代，但 JT 路由、aux、MTP、chunk CE 与 checkpoint 均是真实实现；不宣称验证 NPU kernel。
6. 审查并执行后再决定 guard 是否加入此固定 runner/source allowlist，以及是否将草稿转换为正式请求。本轮不改守卫，不跳过全局 1200 秒冷却，不复用 C/D 的目录或端口。预留端口尚未作远端占用检查。

可在后续获准的无设备容器中使用（以下命令尚未执行）：

```bash
bash /work/tools/run_f637_module_cost_cpu.sh \
  /work/evidence/f637-module-cost-parser-v1.json \
  /work/evidence/f637-module-cost-cpu-v1.json
```

CPU 与 preflight 通过也不代表 NPU 或 200 步通过。四组 40 步资源须用新 off40 同窗比较，计时第 21–40 步逐步全 rank MAX，监督 token 一次计数；显存为 1–40 步累计峰值，不能替代 200 步。只有候选真实生效且短诊断满足精度后，才评估一组 200 步确认。

采集使用既有完整目录 collector，可包含新增 coverage、runtime、calls、cleanup、preflight、manifest 和冻结脚本；清理仅在任务自有进程退出且证据复制/哈希核验后进行。本准备稿不发出任何运行或清理指令。
