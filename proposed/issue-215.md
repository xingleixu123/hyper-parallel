<!-- correctness-review-20261018 -->
## 2026-10-18 正确性复核：单分片写回：保护已有 FP32 更新，触发范围必须明确

**精确触发条件**：分片组大小为 1，写回方法实际收到 FP32 update。此时无广播需求，BF16 pack 会丢弃已经算出的有效位；直接保留输入 dtype 是合理的本地写回改进，与 MF 是否一致无关。

**收紧默认影响**：固定 PR #213 的原生 Muon 动量路径会转 BF16，NS 输出也写回输入 dtype；没有证据证明原生默认 JT 的每个更新都带 FP32 位数。历史“24 份 FP32 update”来自覆盖动量/NS 的归档适配路径，新增 UT 则直接给写回函数输入 FP32。两者不能冒充完整默认 `Muon.step` 的普遍触发。

**实现与判据**：PR #213 保留 layout 切片、weight decay、两种 LR folding、原有最终 foreach 路径，多分片 BF16 通信不在本单修改范围。独立期望是相同 update 直接执行 FP32 加法与衰减，不是 MF loss。BF16 输入对照和无原地改 update 也需保持。

**行业对照**：[MF master PyNative NS](https://gitcode.com/mindspore/mindformers/blob/409019aac0c6579bb4fb8a9b04a75c259d98a96e/mindformers/pynative/optimizer/muon.py#L1195) 明确预归一化/迭代精度边界；[Megatron 可选 QKV 拆分](https://github.com/NVIDIA/Megatron-LM/blob/1effc75a46b26c6b93478355ff9fc3bb8ac1fafc/megatron/core/optimizer/emerging_optimizers.py#L458) 也将正交化、shape scale 和参数写回分成步骤。这支持分别检查每个 dtype 边界，不能证明所有实现的最终 update 都是 FP32。

**推荐**：保留这个小范围条件修复，按条件收益定优先级；若要宣称默认 JT 同样受损，需补固定版本完整 step 产生 FP32 update 的独立证据。既有 212 UT 与真实 NPU 结果是历史已执行证据，本轮源码审查未重新执行这些测试。


本次固定参考：MF GitCode master `409019aac0c6579bb4fb8a9b04a75c259d98a96e`（只核查 `mindformers/pynative/`，未用旧 training_graph 冒充），Megatron main `1effc75a46b26c6b93478355ff9fc3bb8ac1fafc`，当前 JT_dev `a1e6c222eab097c9609d6e744a3c5f98aeed6bdc`。源码路径/行号/SHA-256 见[完整正确性复核](https://github.com/xingleixu123/hyper-parallel/blob/xxl-dev/jt-correctness-review-20261018/AUDIT.md)与[来源清单](https://github.com/xingleixu123/hyper-parallel/blob/xxl-dev/jt-correctness-review-20261018/source-catalog.json)。本轮不新增训练结果，不以 MF loss 距离代替独立正确性判据。


<details><summary>原问题与历史实验记录（适用旧基线；当前判断以上方复核为准）</summary>

# 【子问题·P1】Muon 单分片写回将 FP32 更新额外转为 BF16，损失已有精度

## 检查清单

- [x] 已检索现有问题：#149 汇总旧Muon分块/缩放差异；本单限定为当前实现的单分片通信打包边界。
- [x] 已核对当前代码、精度契约和真实单卡训练证据。
- [x] 有最小失败用例、修复前后对照、实际NPU验证；这是数值错误，没有崩溃堆栈。

## 关联与范围

独立修复：[PR #213](https://github.com/hf1s/hyper-parallel/pull/213)，提交`6f04715a9be134b1e544dcf3c8487bbd8051ffcf`，基于`JT_dev@521d3a8a9aa62353327093ab1258ca2c22006e8b`，一个提交、两个文件。总问题：[单卡MF200→MF/HP400完整排查 #217](https://github.com/hf1s/hyper-parallel/issues/217)。历史关联#149第三节，不能把本修复视为解决#149的全部算法选择。

## 实际问题与预期行为

`Muon._fused_broadcast_and_apply`在分片组大小等于1时也创建BF16通信打包缓冲。已有FP32 update被写入这个缓冲后，再写到FP32主参数；即使没有广播需求，也发生FP32→BF16→FP32。真实训练记录到24份二维更新在打包前为FP32、写回时已发生BF16往返。

预期：单分片本地写回应保留调用者已计算的update精度。学习率已折叠时直接相加；未折叠时沿原路径乘`-lr`；仍使用原有layout切片、解耦衰减和foreach写回。这里不要求NS本身改用FP32，也不改变多分片通信策略。

## 最小复现

在基线版本安装本仓库后运行以下代码。故意构造已计算好的FP32 update，将问题隔离到写回，不依赖模型、NS、数据或MF。

```python
from types import SimpleNamespace
from unittest.mock import patch
import torch
from hyper_parallel.core.optimizer.muon import Muon

optimizer = object.__new__(Muon)
optimizer._param_shard_metadata = {}
parameter = torch.nn.Parameter(torch.tensor([[0.01, 0.02], [0.03, 0.04]]))
update = torch.tensor([[0.1234567, -0.2345678], [0.3456789, -0.456789]])
group = {"lr": 1e-4, "weight_decay": 0.0, "apply_lr_in_update": True}
expected = parameter.detach().clone().add_(update)
with patch.object(optimizer, "_get_shard_info", return_value=((1,), (0,), (None,), 1)), \
     patch.object(optimizer, "_batched_relay_broadcast") as broadcast, torch.no_grad():
    optimizer._fused_broadcast_and_apply(
        [parameter], {parameter: update}, {parameter: (0,)},
        group, SimpleNamespace(layout_spec=None))
broadcast.assert_not_called()
torch.testing.assert_close(parameter, expected, rtol=0, atol=0)
```

旧实现失败；FP32更新、学习率已折叠且衰减为0的NPU样本最大绝对误差为`0.0002422630786895752`，4/4元素不同。完整回归还覆盖FP32/BF16、两种学习率折叠、零/非零衰减、多参数、update不被原地修改及多分片打包对照。

## 消融与影响优先级

|证据层级|结果|解释|
|---|---|---|
|单分片边界|原更新FP32，打包后恰等于BF16往返|直接可证伪的精度损失|
|固定MF状态的真实首步|BF16模型差异元素877586→182；所测dense更新相对误差约0.00165269→1.33e-6|只改打包精度的条件收益，不是全模型质量结论|
|生产修复与上述消融|实际更新后48个模型张量SHA256全相同；恢复、源基线、输入和指标相同|证明独立PR确实覆盖已定位边界|
|完整200步控制63→64|total最大误差0.430796623→0.144495010；MAE0.013333042→0.008620534；超限步均87/200|收益明显，但仍未达到全窗口目标|

优先级P1依据是明确的本地写回精度契约和真实影响；不以“MF更像正确答案”作为唯一理由。其它Muon分块、缩放、NS归约树不混入此PR。

## 哪个实现正确及行业参考

本问题应保留已有update的dtype和精度。FP32主参数不能恢复已经在BF16打包中丢掉的信息；单分片也没有通信压缩的必要性。BF16 NS是独立的算法/性能选择，与NS之后额外降低已有FP32更新精度不同。

[现代MF PyNative Muon](https://gitcode.com/mindspore/mindformers/blob/d3f5dc943f6bf50cfa238db9c320ba4ff75ac52e/mindformers/pynative/optimizer/muon.py)和[NVIDIA Muon参考](https://docs.nvidia.com/nemo/emerging-optimizers/0.4.0/_modules/emerging_optimizers/orthogonalized_optimizers/muon.html)用于区分NS计算、缩放和参数写回边界。不同实现允许不同NS精度，不能据此要求所有中间值FP32；本修复的依据是单分片写回不应再无条件压低已算好的更新。

## 修复验证和验收

- CPU最小回归修改前4个FP32组合失败；修改后覆盖组合通过。
- Ascend910B3/CANN9.2.0-beta.2：修改前4个FP32组合失败，修改后8个dtype/折叠/衰减组合逐元素精确；BF16对照保持一致。
- `python -m pytest -q tests/ut/core/optimizer`：212项通过。
- 修改文件Pylint、`git diff --check`通过；生产方法真实首步验证48/48模型hash一致。
- 多分片只有模拟通信边界回归，未进行多卡训练；没有性能基准，不声称多卡或性能收益。
- 服务端CI尚无当前提交的正向结果，不写“门禁通过”。PR开放不等于总任务回归完成。

验收：上述精度组合无额外转换、update不变、layout/衰减/学习率语义不变、多分片分支回归通过。完整MF/HP200步loss验收由总问题跟踪，不作为本局部修复已完成的声明。

## 环境与证据

HP：torch/torch_npu2.12，CANN9.2.0-beta.2，Ascend910B3单卡；MF参考：用户归档源码、MindSpore2.7.2、CANN8.5.0。共同配置为4K Arrow、2主层+1MTP、Muon+AdamW，MF200完整模型和优化器状态恢复。

证据文件：`pr-muon-single-shard-npu-red-green-final.json`、`pr-muon-single-shard-real-training.json`、对应首步模型逐参数报告、完整实验表63/64。可执行回归已经随PR213上库。


</details>
