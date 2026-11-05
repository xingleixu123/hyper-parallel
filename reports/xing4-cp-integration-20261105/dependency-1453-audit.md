# #1453 新旧 head 与四卡回归依赖审查

## 结论

#1453 从 eead893cb8e593ded11d98f2480184550c1be0e7 更新到 ac4aea64c0a938265b790a049a9dfaeb11630ef6 后，两个完整 head 的生产代码完全相同；差异仅是三个测试文件。就这次依赖更新而言，基于 d7802906ce13b0615844b692a4e2a4b7a0065d8e 的已完成四组回归无需仅为更换 #1453 head 而重跑。这个结论不覆盖其余源码、配置或环境的差异，也不能扩大已完成实验本身的场景和数值覆盖。

两边提交哈希发生变化（a40b3bbd/eead893c 对应 6860e74e/ac4aea64），但不能称为完全相同的提交重排：head 的测试树有真实变化，生产树不变。

## 证据

- 本地精确双头 git diff eead893c ac4aea64 仅列出 tests/ut/trainer/test_aux_loss_scaling.py 和 test_base.py 修改，以及 test_metrics.py 删除。hyper_parallel 目录下没有差异。
- 旧 #1453 的两个提交涉及四个生产文件：hyper_parallel/trainer/base.py、trainer/runtime/metrics.py、trainer/text_trainer.py、trainer/vlm_trainer.py。d780 与新 head 在这四个文件逐字节相同。
- d780 与 #1439 原 HEAD f96dd5b0 在 Xing4 adapter/training.py 和 adapter/distributed/context_parallel.py 无差异；冻结候选对 CP adapter 的工作树变化为 MLA 调度参数与文档，未修改 CP shift 或 per-depth mean_loss 方法。
- 测试差异包括在 aux-loss 测试中合并独立分支并加入空微批覆盖、增加 main_grad 在 clipping 前归一化的断言，以及删除原 test_metrics.py。生产行为相同不意味着新旧测试覆盖完全相同。未运行新增测试，不作通过断言。

## 训练语义

Trainer 对每微批基础 loss 以有效 foundation token 数构造 numerator；同步及累积后，以完整 optimizer step 的 DP/CP 有效 token 总数归一化梯度和 loss 指标，再执行梯度检查及 clipping，最后 optimizer step。若单卡各微批有效 foundation token 数相等，各微批等权；若不等则按该计数加权，并非无条件等权。此处不主张 MF 的其他实现或配置必然等价。

Xing4 每个 MTP depth 先用该 depth 的有效目标 token 数归一化，CP 路径将该 depth 的 numerator 和 count 在 CP 上聚合；各 depth 求和并乘 MTP 系数。Trainer 随后仍按基础 foundation token 数对整个微批目标跨微批加权；它不是将所有微批每个 depth 的 token 全部合并后再求单一 depth 均值。该区别对不同微批有效长度不一致时的对比很重要，不能仅由固定长度单卡结果推断一般情况。

本审查使用精确两个 Git 对象，未采用 diverged API compare 的 merge-base 文件列表；未修改主库或运行 NPU。
