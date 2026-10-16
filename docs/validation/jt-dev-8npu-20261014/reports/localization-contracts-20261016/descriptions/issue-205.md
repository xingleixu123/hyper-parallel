## 最新四卡回归：完整通过（2026-10-15）

独立提交 `e9fc94d376ca7380c13a5e09c0020b34a75dc1d2` 的129四卡原用例已完整结束，未修改阈值：四个rank各 `1 passed, 21 warnings`（44.78–44.86秒），容器退出0、总占用73.99秒、正常释放且无本任务设备PID残留。该入口运行两种CP策略下的TP1、TP2、TP2+SP，以及chunk48的TP1/TP2+SP，共8组合各2步，另含packed FA输出/梯度/统计量对照。

16条rank0记录的LM/MTP/aux日志最大绝对差为0.000682830810546875、0.00011944770812988281、9.103678166866302e-8。既有ST使用相对容差；短步结果不替代总单要求的200步绝对误差≤0.005，不用于宣称吞吐或显存收益。CANN9.2.0-beta.2、Torch/TorchNPU2.12、editable导入路径与补丁SHA已核验，17份证据全部通过SHA校验，重复压缩包已清理。

[分析及16条数值](https://github.com/xingleixu123/hyper-parallel/blob/4432472f2667832980835fb2bff9a815e38570a4/docs/validation/jt-dev-8npu-20261014/reports/pr199-f637-npu-st/pr199-f637-npu-st-analysis.json) · [完整采集清单](https://github.com/xingleixu123/hyper-parallel/blob/4432472f2667832980835fb2bff9a815e38570a4/docs/validation/jt-dev-8npu-20261014/reports/pr199-f637-npu-st/pr199-f637-npu-st-evidence-manifest.json) · [释放凭据](https://github.com/xingleixu123/hyper-parallel/blob/4432472f2667832980835fb2bff9a815e38570a4/docs/validation/jt-dev-8npu-20261014/reports/pr199-f637-npu-st/r129-pr199-f637-st4-r2.receipt.json)。此前67.78秒中断轮是历史未完成尝试，原证据保留，不再作为当前ST状态。

总单：#202；对应独立PR：#199。

## 问题与当前状态

旧 JT 示例默认配置 `loss_chunk_size=4096`，但 CP 安装过程拒绝任何正值，阻断分块输出投影与交叉熵损失。相同目标分布到 CP 分片后，有效监督 token 数可能不相等；不同 MTP 深度移位后也有不同有效数目，必须各自按全 CP 目标数归一化，并避免 checkpoint 重放再次执行计数 collective。

**上游 #186 已实现生产功能。** 当前 JT_dev `f637ef7` 使用 `enable_chunk_loss` 和正整数 `chunk_loss_size`，保留原局部损失并按 CP 有效 token 比例归一化。#199 已基于该版本去除重复生产改动，改为补齐此问题的独立回归保护；没有将他人的上游实现写成本 PR 的成果。

## 当前独立 PR 的范围

单提交 `e9fc94d376ca7380c13a5e09c0020b34a75dc1d2`，仅两个测试文件：

1. CPU 用例：两层未来目标的有效 token 总数分别4、3，分别分成两个 CP 局部分片，使用同一输出头。全部目标先构图再共同反向，对照独立完整序列 CE；两组 hidden 梯度和共享 head 梯度都必须匹配，反向后计数 collective 总次数仍为4。
2. 四卡 ST：保留原六种 expanded_ulysses/latent_kv_head 与 TP/SP 组合各两步、FA 检查和全部容差，再增加 TP1+CP4、TP2+SP+CP2 的 chunk48，各两步。记录输出头输入长度，确认实际进入分块路径；观测只保存整数，hook 异常退出也会移除。

## 复现和回归命令

```bash
export HYPER_PARALLEL_TEST_DEVICE_TYPE=cpu
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export OMP_NUM_THREADS=1
python -m pytest -q tests/ut/models/jt_deepseek_v3/test_context_parallel.py tests/ut/models/jt_deepseek_v3/test_jt_packed.py tests/ut/auto_models/test_projected_cross_entropy.py
```

在新基点的隔离 CPU 容器中，20项测试、25项子测试通过，31.45秒；20条警告来自框架弃用、无 NPU 和既有 pytest 标记。Python3.12.14、Torch2.12.0+cpu、transformers5.13.1、pytest9.1.1，editable 导入路径正确。未挂载 NPU，不能据此声明 NPU 通过。

```bash
python -m pytest -q tests/torch/context_parallel/test_jt_mla_cp.py
```

上述新基点四卡用例已在第二轮完整通过，见顶部证据。规范/验收条件两轴检视均通过；修改文件的静态 pylint 基点与补丁均零告警，因本机缺 Torch 排除 import-error/no-member；空白检查通过。

## 历史证据与数值边界

旧版 PR 的生产实现曾通过13项 CPU 测试、2项子测试，以及47.19秒的真实四卡 ST。六项修复集成源 `ec2e16b` 的 CP2 chunk4096 完成200步，但相对 FSDP 的 LM 最大差0.022237778、日志加权 MTP 最大差0.005699873，仍未通过0.005。这些证据保留于[固定历史报告](https://github.com/xingleixu123/hyper-parallel/blob/2f5ba55838338dcb5c37c9cd44a958586da2e89a/docs/validation/jt-dev-8npu-20261014/REPORT_ZH.md)，不能移用为新基点完整精度或性能结果。

#198 的源参数释放把 CP2 峰值 allocated 从31.183506降至21.058210 GiB，属于独立显存修复。#199 当前没有生产代码变化，没有吞吐或显存收益声明。ST 的相对容差与总单要求的全rank、200步、加权及未加权分量最大绝对差≤0.005分别披露。

当前公开证据：[详细中文总报告](https://github.com/xingleixu123/hyper-parallel/blob/3781c38e91aa73baf84942ff3d4cd459ead758a2/docs/validation/jt-dev-8npu-20261014/REPORT_ZH.md)、[本次 CPU 原始日志](https://github.com/xingleixu123/hyper-parallel/blob/3781c38e91aa73baf84942ff3d4cd459ead758a2/docs/validation/jt-dev-8npu-20261014/reports/pr199-f637/pr199-f637-cpu-ut.log)、[隔离容器凭据](https://github.com/xingleixu123/hyper-parallel/blob/3781c38e91aa73baf84942ff3d4cd459ead758a2/docs/validation/jt-dev-8npu-20261014/reports/pr199-f637/pr199-f637-cpu-receipt.json)、[源码与单提交重写记录](https://github.com/xingleixu123/hyper-parallel/blob/3781c38e91aa73baf84942ff3d4cd459ead758a2/docs/validation/jt-dev-8npu-20261014/reports/pr199-f637/pr199-f637-rewrite.json)。以上链接固定到证据提交，不随分支后续更新漂移。

## 验收清单

- [x] 上游已支持显式开启 CP 分块输出损失。
- [x] 多 MTP 目标独立分母、共同反向和共享 head 梯度的 CPU 回归通过。
- [x] 当前补丁两轴检视、静态检查通过。
- [x] 当前 f637ef7 基点的四卡分布式用例通过。
- [ ] 新基点完整200步跨并行验收；精度、显存和性能由总单统一记录。
- [ ] 维护者评审、当前提交完整 CI 及最终合入确认。

本单保持开启，PR保持草稿；不把上游功能合入或 CPU 通过当作完整200步验收。

## 具体定位、触发条件与两侧正确性

固定本PR提交 `e9fc94d376ca7380c13a5e09c0020b34a75dc1d2`。相关位置：[hyper_parallel/models/jt_deepseek_v3/adapter/distributed/context_parallel.py::projection_loss:301](https://github.com/hf1s/hyper-parallel/blob/e9fc94d376ca7380c13a5e09c0020b34a75dc1d2/hyper_parallel/models/jt_deepseek_v3/adapter/distributed/context_parallel.py#L301)；[tests/ut/models/jt_deepseek_v3/test_context_parallel.py::test_chunked_mtp_depths_keep_independent_denominators:127](https://github.com/hf1s/hyper-parallel/blob/e9fc94d376ca7380c13a5e09c0020b34a75dc1d2/tests/ut/models/jt_deepseek_v3/test_context_parallel.py#L127)；[hyper_parallel/models/jt_deepseek_v3/adapter/distributed/context_parallel.py::reduce_losses:271](https://github.com/hf1s/hyper-parallel/blob/e9fc94d376ca7380c13a5e09c0020b34a75dc1d2/hyper_parallel/models/jt_deepseek_v3/adapter/distributed/context_parallel.py#L271)。

触发条件：原 CP 路径拒绝正 loss_chunk_size；#186 已提供生产能力。当前 PR 仅覆盖 CP2 多 MTP 深度共享 head 的分母独立性及 CP4/TP2+CP2 分块路径。

正确性参考：未分片完整序列 CE 独立参考；两 MTP 深度有效监督总数 4/3，联合建图后 backward，比较 hidden 与共享 head 梯度。

修复侧或并行侧核验：当前基点 CPU 20 passed、25 subtests；真实 4 卡 ST 1 passed，8 组合×2 步，保留 backward 后 collective 数量检查。

覆盖边界：ST 使用其相对容差，16 条 rank0 数值记录；不是 8 rank×200 步绝对误差验收。旧 ec2e 配方 CP200 的失败必须保留；当前 PR 不再声称修改生产实现。

### 最新官方实现对照与适用范围

核对时点为2026-10-16，固定MF官方GitCode master `1c9258ca845e21ddd66fde83e262d5635359bd30`、Megatron官方main `3e2807ce5e787f81b7275be2383092fe0c3bfc71`；MF GitHub镜像当时仍为旧`5089da3`。

最新MF官方master的per-token路径先反传原始分子，累积完成后按全局token计数缩放梯度；其loss sense已经包含1/(DP×TP×CP)，因此最终除数采用每rank平均计数。Megatron相应路径将DDP scaling设为1，再除全局token总数。两侧应重建相同的sum(loss)/sum(valid)目标，并核对SUM/AVG、TP补偿及每个MTP深度的有效分母，不能只照搬某个倍数。依据：[MF反传入口](https://gitcode.com/mindspore/mindformers/blob/1c9258ca845e21ddd66fde83e262d5635359bd30/mindformers/pynative/trainer/trainer.py#L1998)、[MF最终梯度归一化](https://gitcode.com/mindspore/mindformers/blob/1c9258ca845e21ddd66fde83e262d5635359bd30/mindformers/pynative/trainer/trainer.py#L2084)、[Megatron DDP缩放设置](https://github.com/NVIDIA/Megatron-LM/blob/3e2807ce5e787f81b7275be2383092fe0c3bfc71/megatron/core/distributed/distributed_data_parallel.py#L250)、[Megatron最终token缩放](https://github.com/NVIDIA/Megatron-LM/blob/3e2807ce5e787f81b7275be2383092fe0c3bfc71/megatron/core/distributed/finalize_model_grads.py#L741)。本次是源码语义核对，未执行跨框架训练对照；主LM的局部回归不自动覆盖每种MTP/aux定义。 CP还须比较还原后的逻辑序列：[MF连续序列切片](https://gitcode.com/mindspore/mindformers/blob/1c9258ca845e21ddd66fde83e262d5635359bd30/mindformers/pynative/distributed/context_parallel.py#L339)与[Megatron CP块映射](https://github.com/NVIDIA/Megatron-LM/blob/3e2807ce5e787f81b7275be2383092fe0c3bfc71/megatron/core/utils.py#L2634)的物理rank分配不同，直接比较同编号rank并不是有效的跨框架oracle。

## 当前整体验收边界

本问题的定向回归与上文按版本标注的历史集成证据分别保留。最新训练源544526f7目前只有201的单次FSDP8完整200步及checkpoint CPU读回，共享机同源参考、跨机精度、最新续训和重算/并行矩阵尚未验收。当前先完成全200步、8rank主LM loss最大绝对差≤0.005；主loss全矩阵通过后先汇报，由用户决定是否继续MTP/aux专项。最新进度、曲线和资源约束见总问题单[#202](https://github.com/hf1s/hyper-parallel/issues/202)。本次仅补充说明，代码提交未改变，现有CI空结果不构成通过。
