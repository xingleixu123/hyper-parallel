# 重算覆盖报告漏计 FSDP wrapper：验收约束

基点：`hf1s/hyper-parallel:JT_dev` 的 `f637ef7ee4d4b416800716f1ebaaae7709b8a06a`。

真实八卡案例 `r129-f637-fp32-fsdp8-full-200-r3` 在所有 rank 打印安装 3 个重算区域，最终 inventory 记录两个 `abc.HSDPCheckpointWrapper` 和一个普通 `CheckpointWrapper`，但 `activation_checkpoint.json` 的 count 仅为 1。该训练随后因外部卡争用中断，0 个完整优化器步；只能作为报告漏计证据，不能作为 200 步或资源收益验收。

要求：

1. `collect_activation_checkpoint_wrappers` 应识别 HP `CheckpointWrapper` 及 FSDP 动态派生类，函数本身不新增 Torch 或执行后端导入。包级既有导入不属于本补丁的修复范围。
2. 同名但无 HP wrapper 继承关系的类不属于重算区域；保留返回结构、遍历顺序、FQN 和直接包裹对象的类型。
3. 使用真实 checkpoint wrapper 与真实 FSDP 类扩展函数在 CPU 复现两种组合；无需构造 HSDP 状态、分布式进程组或 NPU/GPU。
4. 同一新增 UT 在基点失败、修复后通过；覆盖三处区域、根 wrapper、关闭重算/无关模块以及无后端导入。
5. 只修复报告识别，不改变重算、FSDP、loss 或训练数学。不将此修复声称为 CP/EP 精度修复。
6. 一个中文子问题、一个单提交草稿 PR，与总单 #202 双向关联；保留真实中断日志及测试证据。
