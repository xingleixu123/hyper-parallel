## 问题与范围

总问题：[单卡 MF/HP 对齐 #217](https://github.com/hf1s/hyper-parallel/issues/217)。对应已有 [PR #166](https://github.com/hf1s/hyper-parallel/pull/166)，当前 head `c9134cb852010bb179686ceea163ac100963f2f3`，目标 `JT_dev`，一个提交、三个文件。该 PR 已存在，但原描述仅两段，没有明确的专门子 issue；本单补全提交关联、完整恢复要求和验收边界，不宣称新增代码已验证。

仅转换模型权重，不能复现同一次续训。AdamW 的下一次更新依赖一阶矩、二阶矩和 step；Muon 依赖动量和 step；下一个 batch 还依赖数据游标。恢复后的模型 forward 相同，也不能证明这些状态已经恢复。

本单跟踪**指定旧 MF safetensors 与两层 JT 配方的离线状态转换和原生 HP checkpoint 物化能力**。不要求算术上模仿 MF，不改变损失目标、默认优化器或训练精度。最新 MF PyNative 源码采用不同模块名时必须另行适配，不能把此映射当成所有 MF master 的通用转换器。

## 当前实现

| 文件 | 当前作用 |
|---|---|
| `scripts/checkpoint_conversion/convert_mf_ckpt_to_hp.py` | 按既定 JT 参数映射、Q/latent-KV 拼接、gate/up 去交错及专家张量转置，生成 `model.npz`、`optimizer.npz`、`state.json`、`CONVERSION.json` |
| `scripts/checkpoint_conversion/materialize_hp_native_checkpoint.py` | 构建指定 HP Trainer，载入 `adam_m/adam_v/muon_m`，设置 optimizer step、常数 LR 调度计数、顺序数据游标，再调用原生 CheckpointerCallback 写 DCP；不执行训练 |
| `scripts/checkpoint_conversion/README.md` | 记录两阶段命令、dtype 和固定网络/批量限制 |

源代码固定于 [PR166 head](https://github.com/hf1s/hyper-parallel/tree/c9134cb852010bb179686ceea163ac100963f2f3/scripts/checkpoint_conversion)。模型和 optimizer 的布局变换应逐项对应；不只转换权重，不靠重新初始化 optimizer 掩盖缺失状态。

## 支持边界与正确性判据

- 旧源键名、layer0 dense、layer1 MoE、一个 MTP 层；不是任意层数/模型的通用 mapper。
- materializer 当前要求 `global_batch_size=1`，按顺序采样和常数 LR 配方重建计数；不能宣称恢复任意随机 sampler、动态 loss scaler 或非恒定 LR 调度历史。
- `--step N` 与源 `state.json.global_step` 必须一致。`2*N` 是当前脚本的训练长度设置，不是 checkpoint 格式要求。
- 源模型/optimizer 状态约定为 FP32。转换过程保留数组 dtype；不支持的 BF16 safetensors 明确拒绝。表参数 storage dtype 与后续 forward compute dtype 分开。
- 真正的判据是：参数、moment、step、调度器与数据游标映射和恢复正确；不是“恢复后 loss 更像 MF”。

## 最小使用方法

先准备匹配的 MF checkpoint、JT 配置和空输出目录，配置中的 `reference_weights` 指向转换目录。

```bash
mkdir -p /your/hp-offline-state /your/hp-native-checkpoint
python scripts/checkpoint_conversion/convert_mf_ckpt_to_hp.py \
  --ckpt /your/mf-step200.safetensors --out-dir /your/hp-offline-state
python scripts/checkpoint_conversion/materialize_hp_native_checkpoint.py \
  /your/train-4k.yaml --state-dir /your/hp-offline-state \
  --output /your/hp-native-checkpoint --step 200
```

以原生 callback 实际生成的目录为准，不能把脚本打印路径当作落盘成功的唯一证据。

## 已有证据与尚未完成的验收

本任务后续续训的第一次 forward 前，独立观测验证了 **48 个模型数组、28 个 Muon 动量、18 个 AdamW 一阶矩、18 个二阶矩**，shape/dtype/值精确，step、scheduler 和 sampler 游标均为 200；说明该次使用的锚点完整，不是零状态重启。

[实际恢复记录](https://github.com/xingleixu123/hyper-parallel/blob/3f1afbb1ff6b8a4d4c36cd1880a64d1e0b982410/runs/hp-archive200-r2/RESTORE.json) · [独立最终审计](https://github.com/xingleixu123/hyper-parallel/blob/3f1afbb1ff6b8a4d4c36cd1880a64d1e0b982410/evidence/clean-final-validation.json)。

这不等于 PR166 当前 head 的完整独立回归。本次只核对了源码、提交与已有恢复证据，没有重新运行该 head 的转换工具。该 PR 当前没有随代码提交 UT/ST 文件，不代勾测试或维护者评审。

合入前还需明确并验证：

1. **缺槽应被拒绝**：当前 `load_optimizer_npz` 对不存在的 key 使用 `continue`，`expected` 也只统计已找到的 key；因此 `assigned == expected` 单独不能证明所有必需槽齐全。需补缺失 Adam/Muon state 的失败用例，不能静默保留初始化槽。当前那次真实锚点的独立 64 数组审计已排除此问题，但不能替代工具自身保护。
2. **占位与真实状态区分**：只跳过另一 optimizer family 的合法占位；不能把真实标量状态或不匹配形状的必需槽当作占位。验证 shape、dtype、未消费非占位状态和部分转换输入。
3. **恢复计数的范围**：常数 LR、batch1 和顺序游标条件需与配置核对；对不支持的采样/调度策略不能声称通用恢复。源身份元数据和实际输出路径应准确，不以固定示例文件名证明来源。
4. **独立回归**：同一有效源状态先转换、物化、重新加载，逐张量比较参数和 optimizer，并确认下一批数据及下一次更新；至少覆盖缺失/错误 shape/dtype/不支持输入的失败路径。

以上是明确的待完善项；“已有 PR 已提交”不等于所有输入保护和合入验收已完成。本单保持开放，PR166 的原始代码 head 不在本次关联补充中改写。
