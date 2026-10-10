# 训练 loss 与吞吐日志

使用 YAML Trainer 时，保留已有启动方式，只需设置日志间隔：

```yaml
training:
  logging_steps: 1
```

由现有 LoggingCallback 在 rank 0 输出；已有进度条和远端日志回调消费同一份 `step_env_metrics`。不需要新增回调或手工填写 FLOPs 常数。

## 字段和口径

| 字段 | 含义 |
| --- | --- |
| `training/total_loss` | 原有参与反传的训练目标总和，保持原算法 |
| `training/foundation_loss...` | 原有分项目标；名称由 loss 模块提供 |
| `training/lr`、`training/grad_norm` | 学习率、裁剪前梯度范数 |
| `training/mtp_1_loss`、`mtp_2_loss` 等 | 模型提供时打印：每个 MTP 深度乘训练系数前的诊断均值 |
| `training/aux_loss`、`training/indexer_loss` | 模型提供同名标量时打印；不从总 loss 反推，不替模型增加目标 |
| `performance/step_time` | 秒；等待本步设备工作完成后，取所有 rank 的最大耗时 |
| `performance/tokens_per_second` | 原有有效监督 token/s（优先使用 Trainer 的 token_count） |
| `performance/input_tokens_per_second` | 输入 tensor 的 token/s，包含 padding；与监督 token/s 分开 |
| `performance/samples_per_second` | 全局逻辑样本/s |
| `performance/throughput_tflops_per_device` | 每卡估算模型 TFLOP/s，公式见下 |
| `data/step_tokens`、`data/step_input_tokens`、`data/step_samples` | 本步对应计数 |
| `data/consumed_tokens`、`data/consumed_samples` | 原有累计计数 |

诊断 loss 是各微步、DP/CP rank 的算术均值，与 MF PyNative tracker 的聚合约定一致。它们用于观测，**不加入训练目标**。不要把这些均值再相加重构 `total_loss`：SFT 有效标签数不等时，Trainer 的训练目标可能采用不同的 token 加权。

## FLOPs 与并行

`TFLOP/s/device = 本步全局模型 FLOPs / 最慢 rank 的步时(s) / world_size / 1e12`。

使用 MF 的 DeepSeek MLA 模型计算口径：前向加反向，包含 dense、激活的 routed/shared experts、MLA、causal attention、执行的 MTP 层及输出 head。按输入形状计算，忽略标签 mask；CP 还原完整序列长度，先在 DP+CP 内求和，CP 样本只算一次，TP 副本不重复计数。MTP 层数取实际模型模块，不把 checkpoint 配置里未执行的 MTP 层算进去。

当前估算器支持 DeepSeek V2/V3 和 JT V3 的 SiLU/MLA 结构。其他模型仍有 token/s 和样本/s，省略 TFLOP/s。该值不计优化器、通信、重计算和内核额外 padding；不是硬件实测 FLOPs 或 MFU。不同公式得到的 TFLOP/s 不能直接横比。PP 的分段模型不在本次 FLOPs 支持范围内。

参考：MindFormers master `d3f5dc943f6bf50cfa238db9c320ba4ff75ac52e` 的 [loss_callback.py](https://gitcode.com/mindspore/mindformers/blob/d3f5dc943f6bf50cfa238db9c320ba4ff75ac52e/mindformers/pynative/callback/loss_callback.py) 和 [models/utils.py](https://gitcode.com/mindspore/mindformers/blob/d3f5dc943f6bf50cfa238db9c320ba4ff75ac52e/mindformers/models/utils.py)。

## 模型接入诊断 loss

模型在原有 output 之外提供可选 `loss_metrics` 标量字典。Trainer 在 backward 前 detach 后收集，设备标量只在步结束时读取：

```python
# 原来的 loss 字段和反传公式保持不变。
output.loss_metrics = {"mtp_1_loss": raw_mtp_loss.detach()}
```

key 必须是非空、不含斜杠的名称；同一优化器步内及 DP/CP rank 间必须一致，值必须是 TP 上已复制的本地标量均值，不能直接传 TP 局部和。保留的训练目标名称（如 total_loss）不可覆盖。没有对应目标就省略，不打印伪造的 0。

共享 `calculate_mtp_loss(..., loss_metrics=metrics)` 可填入各深度标量；调用者把 metrics 放入模型 output 后才会记录。模型原有 `aux_loss` 和 `indexer_loss` 也可直接读取；`aux_loss` 不自动改名为 MF 的按层平均 load_balancing_loss，因为来源模型的系数和归一化可能不同。
