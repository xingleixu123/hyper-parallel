<!-- correctness-review-20261018 -->
## 2026-10-18 正确性复核：token 权重：按目标函数推导的恒等梯度修复

**正确性依据**：令微批有效 token 数为 n、全局为 N，局部 loss 是 mean，DP×CP 域平均梯度的大小为 D。回传应为 `local_mean * (n/N) * D`，平均后得到全局 token 平均梯度。D=1、一个微批时，比例恰为 1；应避免分开计算 `1/N` 和 n 所引入的不必要舍入。

**行业对照**：[Megatron 全局 token 归一化](https://github.com/NVIDIA/Megatron-LM/blob/1effc75a46b26c6b93478355ff9fc3bb8ac1fafc/megatron/core/distributed/finalize_model_grads.py#L738) 先累计 token loss 总和的梯度，再汇总全局 token 数并乘倒数。HP 的入口是局部 mean，两条完整链路可实现同一目标，不能只抄其中一行。该修复不是为了匹配 MF 的算术顺序。

**状态与范围**：PR #180 已合入当前 JT_dev。已有用例覆盖单微批恒等、多微批不同 token 数、DP×CP、SP 与命名目标；本轮只复核公式和固定 diff，不将其写成新增多卡实测。不改数据 mask、损失系数或优化器状态。


本次固定参考：MF GitCode master `409019aac0c6579bb4fb8a9b04a75c259d98a96e`（只核查 `mindformers/pynative/`，未用旧 training_graph 冒充），Megatron main `1effc75a46b26c6b93478355ff9fc3bb8ac1fafc`，当前 JT_dev `a1e6c222eab097c9609d6e744a3c5f98aeed6bdc`。源码路径/行号/SHA-256 见[完整正确性复核](https://github.com/xingleixu123/hyper-parallel/blob/xxl-dev/jt-correctness-review-20261018/AUDIT.md)与[来源清单](https://github.com/xingleixu123/hyper-parallel/blob/xxl-dev/jt-correctness-review-20261018/source-catalog.json)。本轮不新增训练结果，不以 MF loss 距离代替独立正确性判据。


<details><summary>原问题与历史实验记录（适用旧基线；当前判断以上方复核为准）</summary>

# 【子问题·P1】Trainer 单微批 token 权重先乘后除破坏恒等梯度

## 检查清单与关联

- [x] 已检索现有问题；#179为AdamW完整消融总单，#203为不等微批目标归一化问题，均不替代本单的浮点恒等不变量。
- [x] 已核对实际`mean_global_loss`和真实回归。
- [x] 最小数值复现和真实函数回归均有修复前后证据。

独立修复：[PR #180](https://github.com/hf1s/hyper-parallel/pull/180)，提交`5de3cceab094bbed18915d1a9c9231ca220629ca`，已合入当前JT_dev。历史完整消融在#179；本单补齐“一个问题对应一个PR”的追踪粒度，关联本轮[200→400步总问题 #217](https://github.com/hf1s/hyper-parallel/issues/217)。保留问题开放状态，不因补建链接自动关闭。

## 问题和预期

一个rank的一个microbatch拥有整步全部有效token、DP=CP=1时，从局部mean loss到训练loss的反传权重应为1。旧计算先`loss * n`再除`n`，浮点反向可能先计算`1/n`再乘`n`，导致权重偏离1。FP32、n=164时，得到`0.9999999403953552`。

```python
import torch
loss = torch.tensor(1.0, requires_grad=True)
n = torch.tensor(164.0)
((loss * n) / n).backward()
print(loss.grad)  # 用loss.grad.item()读取完整值
assert loss.grad.item() == 1.0
```

上述例子隔离算术现象。真实产品回归见PR180中的`tests/ut/trainer/test_loss_metrics.py`：只模拟collective，调用实际`mean_global_loss`，避免只测试一段复制公式。

## 修改及正确性

先计算`current_tokens / global_step_tokens * dp_size * cp_size`，再乘loss。全局日志均值、通信调用次序、各目标token计数和SP额外缩放按原契约保留。局部均值乘有效token比例是数学依据，不能简单删掉一般场景的归一化。

[Megatron的全局梯度归一化](https://github.com/NVIDIA/Megatron-LM/blob/3bb482c487cd447a2b40a3bf076ab14cad514e13/megatron/core/distributed/finalize_model_grads.py)从token loss总和出发，HP此处从局部均值出发，两者要比较完整公式。不能直接照搬其中一行，也不声称Megatron保证全部浮点恒等式。

## 逐项消融与影响边界

|场景|结果|结论|
|---|---|---|
|真实CPU函数回归|旧实现1失败/6通过；修复后含相邻Trainer用例9通过|恒等、3/7不等微批、DP×CP=4、SP=2、独立命名目标、空目标和非法零计数被覆盖|
|历史AdamW固定组合，仅还原旧公式|200步MAE0.0676338065、最大误差0.517194748、147步超0.005；共同对齐组合MAE2.30073928833e-7、最大误差9.53674316406e-7|这是#179固定组合内的条件影响，不能移植成当前Muon收益|
|当前Muon200个真实token计数|NPU中28个非恒等计数修复后全部恒等；首个受影响计数出现在206步|当前201首步梯度不受影响；不能归因为首步偏差|
|当前包含PR180的完整200步|后续多组仍失败|此PR不能单独解决总问题|

P1依据：接口要求的恒等权重被破坏，有实际最小反例及独立修复；当前剩余排查不再把已合入问题当作未修复主因。

## 验收与环境

执行：

```bash
python -m pytest -q tests/ut/trainer/test_loss_metrics.py \
  tests/ut/trainer/test_text_trainer.py \
  tests/ut/trainer/test_distributed_runtime.py
```

验收要求：单微批全token反向系数严格1；不等微批/DP/CP/SP保持正确比例；原有日志和通信次序不变；独立目标计数不串用。真实多卡回归未在该PR中声明。

当前复核硬件为Ascend910B3单卡，HP torch/torch_npu2.12、CANN9.2.0-beta.2；MF MindSpore2.7.2、CANN8.5.0。历史AdamW与当前Muon+AdamW使用各自原始记录，不混合指标。PR已合入与当前全窗口仍失败同时成立。


</details>
