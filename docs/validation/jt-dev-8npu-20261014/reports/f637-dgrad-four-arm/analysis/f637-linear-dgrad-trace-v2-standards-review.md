**PASS：两项 P2 均已关闭，本轮没有新增阻断问题。**

审核时间：2026-10-16T04:35:00.607306+08:00。本轮属于独立诊断工具的 Standards／行为与证据复核，不是生产 PR 审批，也不并入 TP2 资源审计。硬规范问题 0，需处理的 Fowler 启发式发现 0。

检查范围包括观察层、训练入口、CPU 边界用例及被委托的旧数学实现；只读核对 CPU v2 结果、原始日志、部署与退出回执。适用 [code-style.md](<D:/01-work/hyper-parallel/.agent/rules/code-style.md>) 的边界校验、显式异常处理和局部副作用规则，以及 [code-review 技能](<C:/Users/90803/.agents/skills/code-review/SKILL.md>) 的 Standards 轴；启发式不替代仓库规则，未重复报告自动工具可检查的格式项。

- **bias 子类旁路闭合。** [eligibility](<D:/Obsidian Vault D/hyperllal/jt-dev-8npu-20261014/tools/linear_dgrad_trace_experiment.py:110>) 将 bias 为 None 或精确本地 Tensor／Parameter 纳入条件，bias 为 Tensor 子类或 DTensor 时不进入干预；类型和 dtype 也写入元数据。[真实 bias 子类用例](<D:/Obsidian Vault D/hyperllal/jt-dev-8npu-20261014/tools/check_linear_dgrad_trace_cpu.py:113>) 对比原生 forward、input/weight 梯度，并要求事件保持 native。
- **安装和 ownership 异常清理闭合。** [安装回滚](<D:/Obsidian Vault D/hyperllal/jt-dev-8npu-20261014/tools/linear_dgrad_trace_experiment.py:75>) 在部分注册失败时移除已注册 handles 后原样抛错；[close](<D:/Obsidian Vault D/hyperllal/jt-dev-8npu-20261014/tools/linear_dgrad_trace_experiment.py:157>) 遇其他 owner 替换 F.linear 时保留对方函数，移除自己的 hooks、记录 ownership 丢失，再报告异常。CPU 用例分别触发第二个 Linear 注册失败与 ownership 冲突，验证没有残留 hooks 或覆盖其他 owner。
- **数学和 autograd 契约保持。** [四输入 Function](<D:/Obsidian Vault D/hyperllal/jt-dev-8npu-20261014/tools/linear_dgrad_trace_experiment.py:30>) 委托旧 forward/backward；第四个非 Tensor observer 对应第四个 None 梯度。旧 backward 的 once_differentiable 经委托保留。旧数学文件与 v1 冻结副本逐字节一致，新观察层未重写 forward 或 wgrad 算法。
- **hook 与记录生命周期明确。** pre-hook 入栈，always_call=True 的 forward hook 在正常返回及 non-reentrant checkpoint 早停异常中出栈。输出梯度 hook 隐式返回 None；事件保存名称、shape、dtype、布尔值与身份字符串，没有激活或梯度 Tensor。Counter 不积累逐步张量；每 rank 最多写入 20,000 个事件，超限即失败。事件追加写和 summary 写入以每臂使用独立、全新输出目录为运行前提。

已独立核验 CPU v2 **22 case PASS，Torch 2.12.0+cpu，devices=[]，npu_initialized=false，exit=0，removed=true**：native／FP32 两臂各 8 种冻结梯度组合、1 个 FP32 dtype 旁路、2 个 non-reentrant checkpoint 对照及 3 个新增边界例。两个 checkpoint 用例分别记录 plain_calls=2、checkpoint_calls=4，各自梯度完全一致。结果中的执行源码 SHA 与当前文件、部署清单一致；原始日志中的完整结果与 JSON 相同。本审查未自行启动这些测试。

训练入口在审核期间新增 arm 元数据，已按下表最终 SHA 只读复核；安装 observer 后写元数据失败仍会进入 main 的 finally 清理。该入口及其元数据没有被上述 22 个 CPU 用例执行覆盖，production_source_unchanged 等声明仍需后续 runner／guard 的源码证据绑定，不能单凭布尔字段证明。

范围限制：aarch64 CPU 日志原样保留 MKLDNN BF16 不支持后切换 BLAS gemm 的 warning。这些结果不证明 NPU 原生大矩阵 forward/wgrad 选核等价，也不证明真实 FSDP／CP 四臂运行通过。真实 DTensor 分布式计算、高阶梯度、NPU 流及显存行为没有由这 22 case 实测覆盖。native 的 output_gradient_observed 仅表示输出梯度到达；forward_during_backward 是可选 graph-task 提示，不单独证明某个 checkpoint 或内核完成。额外 autograd replay 和记录开销不支持性能收益声明。后续四臂 runner、guard 扩展及 NPU 验收不在本轮结论内。

受审源码与证据 SHA256：

| 文件 | SHA256 |
|---|---|
| [linear_dgrad_trace_experiment.py](<D:/Obsidian Vault D/hyperllal/jt-dev-8npu-20261014/tools/linear_dgrad_trace_experiment.py>) | `62a18e143b1a8af826cc8ebdaf201e7be442c8179362caa102e02a73bea8ab87` |
| [check_linear_dgrad_trace_cpu.py](<D:/Obsidian Vault D/hyperllal/jt-dev-8npu-20261014/tools/check_linear_dgrad_trace_cpu.py>) | `bb8d24f5e7d9e7024174a5410473a3d9a0a4e65c01144f0945474d420a7db39b` |
| [capture_fair_dgrad_training.py](<D:/Obsidian Vault D/hyperllal/jt-dev-8npu-20261014/tools/capture_fair_dgrad_training.py>) | `0ee23ecae60977dda9dd7efee91c34de1e809acb428e10da82850dc8b306f97b` |
| [linear_dgrad_experiment.py](<D:/Obsidian Vault D/hyperllal/jt-dev-8npu-20261014/tools/linear_dgrad_experiment.py>) | `373b5ed016604a92dbfec9afeb2c76edc8b456cf427ade592dfab2aa8098f925` |
| [run_linear_dgrad_trace_cpu.sh](<D:/Obsidian Vault D/hyperllal/jt-dev-8npu-20261014/tools/run_linear_dgrad_trace_cpu.sh>) | `796f111f6082b1035cd5dd3a04c1d9aae83688b22154c01a443d8c5e3b90d054` |
| [f637-linear-dgrad-trace-cpu-v2.json](<D:/Obsidian Vault D/hyperllal/jt-dev-8npu-20261014/evidence/f637-linear-dgrad-trace-cpu-v2.json>) | `b15a2fe57500c5f5857f3bf71540808968a2f1b1242270839cbf86352fcdcce2` |
| [f637-linear-dgrad-trace-cpu-v2.receipt.json](<D:/Obsidian Vault D/hyperllal/jt-dev-8npu-20261014/evidence/f637-linear-dgrad-trace-cpu-v2.receipt.json>) | `b2c326c79d8925dd6312844f4e74d07972b486f36c5f94001443f4088270b95d` |
| [f637-linear-dgrad-trace-cpu-v2.container.log](<D:/Obsidian Vault D/hyperllal/jt-dev-8npu-20261014/evidence/f637-linear-dgrad-trace-cpu-v2.container.log>) | `84ef807447322f8cc0e48e8857e4190ffbb855ab831f69e826f02edaad505a4a` |
| [f637-linear-dgrad-trace-cpu-v2-deployment.json](<D:/Obsidian Vault D/hyperllal/jt-dev-8npu-20261014/evidence/f637-linear-dgrad-trace-cpu-v2-deployment.json>) | `ddfba2ea26e2e747ab5169c59786a21f4d160ca4062a20ace54e3c6e0dd4d9b7` |

规范指纹：code-style.md 为 `fb951b583e7319cf23edffc184fbbbbc830e1092a5f9a0bdcc1561b65046b46e`；code-review 技能为 `3abbdfe52540f9242e16589fae2cc423b3d98994c37683944462b29bab89495c`。

本次仅本机只读检视与证据解析；没有启动容器或设备任务，没有修改源码、诊断工具或原始证据，没有 GitHub 操作。唯一写入是本审核文件。
