<!-- correctness-review-20261018 -->
## 2026-10-18 正确性复核：共享 down 归约：保留诊断工具，不认定普通 HP norm 错误

**当前判断**：固定 CANN 8.5 归约与 HP 普通 FP32 norm 的差异，可以解释旧训练轨迹分叉，但不证明普通 HP 的数学实现错误。旧样本里 MF norm 标量更近 FP64，而 92,955 个不同 BF16 normalized 值全部是普通 HP 更近 FP64；必须同时看标量、最终值和训练轨迹。

**最新行业依据**：[MF master PyNative NS](https://gitcode.com/mindspore/mindformers/blob/409019aac0c6579bb4fb8a9b04a75c259d98a96e/mindformers/pynative/optimizer/muon.py#L1195) 使用 FP32 norm/clamp 后 BF16 NS；当前 Megatron 的 EO 开发依赖 v0.3.0 使用 FP32 输入和 normalize，不要求旧 DVM 固定 tile/core。`clamp(norm,eps)` 与 `norm+eps` 是不同定义，不能混称一个 epsilon 实现。

**PR 处理**：保留 PR #241 作为 examples 下的只读诊断工具，报告实时输入的普通路径、归档路径和独立 FP64 误差；不更新训练模型，不声称修复了公共 optimizer。PR #245 对旧归约的训练接入仅是 Draft 归档复现候选。

**验收与限制**：旧 #99→#100 单因素影响、完整 200 步和 16 项输入/统计/hash 检查仍按原硬件、shape、版本成立。CLI 诊断不是训练修复；有限样本不证明任一路径对所有输入更准确。本轮未运行最新 MF 数值实验。


本次固定参考：MF GitCode master `409019aac0c6579bb4fb8a9b04a75c259d98a96e`（只核查 `mindformers/pynative/`，未用旧 training_graph 冒充），Megatron main `1effc75a46b26c6b93478355ff9fc3bb8ac1fafc`，当前 JT_dev `a1e6c222eab097c9609d6e744a3c5f98aeed6bdc`。源码路径/行号/SHA-256 见[完整正确性复核](https://github.com/xingleixu123/hyper-parallel/blob/xxl-dev/jt-correctness-review-20261018/AUDIT.md)与[来源清单](https://github.com/xingleixu123/hyper-parallel/blob/xxl-dev/jt-correctness-review-20261018/source-catalog.json)。本轮不新增训练结果，不以 MF loss 距离代替独立正确性判据。


<details><summary>原问题与历史实验记录（适用旧基线；当前判断以上方复核为准）</summary>

总问题：[#217](https://github.com/hf1s/hyper-parallel/issues/217)。本单只跟踪共享专家 down 的归一化边界，不把 397 步传播到其它梯度的差异重复算成新问题。

## 🚀 问题与功能描述

第99组独立200步中，总loss/Aux已达标，原始MTP仅398步绝对误差0.009670257568359375超过0.005。定位到396步主干 `model.layers.1.mlp.shared_experts.down_proj.weight` 的Muon更新（MF对应 `decoder.layers.1.mlp.shared_experts.linear_fc2.weight`，形状5120×3072）。

需要独立可执行的诊断复现和完整正确性说明，避免只凭“更接近旧MF loss”就更改HP产品默认。**当前证据归类为旧MF编译图兼容差异，没有证明HP普通归约是数学缺陷。**

## 具体位置与因果

- MF归档 `mindformers/core/optim/muon.py::zeropower_via_newtonschulz5_2d` 的 `x/(x.norm()+1e-7)`，实际DVM `Cast_Mul_ReduceSum_split-op0`；9776元素/块、1609块、尾8832、16 lanes、40 vector cores。
- HP受控实验 `muon_jit_probe.py::graph_jit_ns` 配套 `hp_ns_epsilon229_factor.py`：FP32普通norm、同一epsilon、除法后BF16。它是第99组控制，不能混称原生JT_dev所有默认行为。
- 395完整目标权重/动量精确，396全部46项梯度/平方和/范数精确，目标裁剪后梯度及新动量精确；两端原更新replay完整复现自身训练，归一化前输入完整原生哈希一致。
- 396归一化产生92,955个BF16差异，更新后9,556,275个FP32位差、117,828个BF16差；只改变实时归约顺序，15,728,640个更新后权重全部精确等于MF，动量不变。
- 新因素所有步、主干/MTP两处均执行，无step、元素坐标、目标loss或MF续训参考值特判。

## 完整200步单因素影响与优先级

| 指标 | #99最大误差 | 仅新增共享down归约的#100最大误差 | #100首步误差 | #100超0.005步数 |
|---|---:|---:|---:|---:|
| 总loss | 0.00276947021484375 | 1.9073486328125e-6 | 9.5367431640625e-7 | 0/200 |
| 原始MTP | 0.009670257568359375 | 3.1789143886129523e-6 | 7.947285975973273e-7 | 0/200 |
| 原始Aux | 0.001382586546242237 | 1.2973323464393616e-6 | 5.8673322200775146e-8 | 0/200 |

本阶段P1，因为这是唯一改变三项验收结论的阻塞因素；原始MTP最大误差下降约99.967%。该收益仅在第99组背景成立，不声称原始HP单加此因素即可通过，不把不同背景消融收益相加。所有200步全局梯度范数精确一致。独立197步观测逐值复现完整运行前缀；396/397全部46项梯度、平方和/范数及48项模型FP32/BF16哈希精确。

## 两边正确性与最新行业参考

HP普通norm=0.22601467370986938，MF=0.2260146588087082，FP64=0.22601466135891418。MF norm更近FP64，但92,955个不同normalized位置全部是HP的BF16结果更近FP64。需要区分标量norm精度、最终BF16精度与归档轨迹兼容。

2026-10-16重新核实：

- [MF master PyNative 1c9258ca](https://gitcode.com/mindspore/mindformers/blob/1c9258ca845e21ddd66fde83e262d5635359bd30/mindformers/pynative/optimizer/muon.py) 第1195–1199行：FP32 norm，除以clamp(norm,min=eps)，之后BF16。
- [Megatron-LM 3e2807ce](https://github.com/NVIDIA/Megatron-LM/blob/3e2807ce5e787f81b7275be2383092fe0c3bfc71/megatron/core/optimizer/muon.py) 已委托Emerging Optimizers。
- [NVIDIA EO 4bbed19e](https://github.com/NVIDIA-NeMo/Emerging-Optimizers/blob/4bbed19e681caeaeb630f7ae91d0518d363404b8/emerging_optimizers/orthogonalized_optimizers/muon_utils.py)：FP32输入，默认functional.normalize，可选FP64 norm；BF16迭代受matmul precision控制。此处是上游参考，不声称所有Megatron安装锁定该EO提交。

当前行业实现不要求旧DVM固定tile/core调度；clamp(norm,eps)与norm+eps也不是同一定义，此次没有改变epsilon语义。通用精度API复用#210，不重复扩展本单。

## 环境、复现与验收

125实际910B3，单卡、4K Arrow、batch1、seed42，2主层+1MTP。MF用户归档不改，MS2.7.2/CANN8.5.0；HP torch/torch_npu2.12/CANN9.2.0-beta.2，JT_dev e21302f9加PR192，集成181fab5f。48模型、28Muon动量、18+18AdamW矩、step/data=200均完整恢复。MTP系数0.3/Aux1e-4保持，原始口径分别除系数，逐步检查而非平均。

完整训练按公开启动脚本重新生成201–400，再运行 `compare_shared_down396_full200_v2.py` 和 `check_loss_acceptance.py`。局部诊断输入为实际 `(postclip_gradient + updated_momentum*0.95).bfloat16().float()`，不得拿未经前置舍入的梯度冒充NS输入。对应PR提供单独CLI：

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python examples/torch/alignment/compare_shared_down_normalization.py   --input /your/diagnostic/ns_input.npy --output /your/diagnostic/normalization.json
```

验收：同一真实输入对齐MF原生统计的normalized哈希；同时输出普通路径与FP64误差，不只报“更像MF”；不改输入/模型、不覆盖报告、拒绝不支持输入。实际16项真机验证已通过。总loss首步≤0.0005，原始MTP/Aux首步≤0.001，三项200步每一步≤0.005已独立通过。

## 现有替代方案与DFX

原始monkeypatch用于诊断，已完整公开执行源码。独立CLI使复现不依赖整套实验入口，使用统计/hash输出；固定调度含CPU归约与同步，仅适合本已测内核的诊断兼容，不承诺性能、其它形状/芯片或多卡。通用默认采用现代精度设计时另作正确性/性能评审。

## API与补充信息

不改变HP公开API、默认优化器、数据或系数。新增诊断CLI仅接受实际输入和报告路径。当前训练基点固定用于单因素对照；上游后续PR232针对K-shift的合入不能冒记为本次200步代码版本。

归档原源码SHA256：43fee7992bc12281a2c4586799fb99f927cc15d0a9c085fe016ab44c588bdf49。396目标数组及CLI中间输入完成取证后已清理约755MB；重新replay先按采集入口再生，保留hash不能代替完整数组。

[完整因果链](https://github.com/xingleixu123/hyper-parallel/blob/f4825d1d03888ef17042e487b7efbecb05d7437d/evidence/frontier396-causal-chain.json) · [200步审计](https://github.com/xingleixu123/hyper-parallel/blob/f4825d1d03888ef17042e487b7efbecb05d7437d/evidence/frontier396-shared-independent200-audit.json) · [独立原生统计](https://github.com/xingleixu123/hyper-parallel/blob/f4825d1d03888ef17042e487b7efbecb05d7437d/evidence/frontier396-shared-native-statistics-comparison.json) · [具体位置与正确性](https://github.com/xingleixu123/hyper-parallel/blob/f4825d1d03888ef17042e487b7efbecb05d7437d/阶段记录-396归一化正确性与行业对照.md) · [CLI真机验证](https://github.com/xingleixu123/hyper-parallel/blob/f4825d1d03888ef17042e487b7efbecb05d7437d/evidence/frontier396-pr-diagnostic-validation.json) · [完整报告](https://github.com/xingleixu123/hyper-parallel/blob/f4825d1d03888ef17042e487b7efbecb05d7437d/总问题-三项loss达标.md)


## 独立PR

[PR #241：共享专家Muon归一化对照工具与正确性说明](https://github.com/hf1s/hyper-parallel/pull/241)，head `633ed7d435594bc05b10cc0a3aee9833c53ac927`，一个提交、两个example文件。16项真实输入检查通过；当前PR开放，CI尚无检查结果，未宣称已合入或CI绿色。总问题仍为#217。


</details>
